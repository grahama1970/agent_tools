import json
import os
import asyncio
import aiohttp
import pyperclip
import regex as re
import subprocess
from tqdm import tqdm
from typing import Optional, List
from loguru import logger
from asyncio import Lock
from collections import defaultdict
from huggingface_hub import snapshot_download
import openai
import asyncssh
from src.generate_schema_for_llm.shared.env_utils  import get_project_root, load_env_file

# Setup logging
logger.add("lora_manager.log", rotation="10 MB", level="DEBUG")

class LoraManager:
    def __init__(self, config: dict):
        """
        Initializes the LoraManager.
        """
        self.docker_container_name = config.get("container_name")
        self.server_url = config.get("server_url")
        self.huggingface_adapters_dir = config.get("huggingface_adapters_dir", "/root/.cache/huggingface")
        self.hf_token = os.getenv("HF_TOKEN")
        self.ssh_username = config.get("ssh_username")
        self.ssh_client_keys = config.get("ssh_client_keys")
        self.use_ssh = config.get("use_ssh", False)
        self.lora_locks = defaultdict(Lock)
        self._ssh_connection = None  # Persistent SSH connection
        self.is_remote = self.use_ssh  # Set context once during initialization
        self.lora_registry = {}  # {lora_name: lora_path}
        self.base_model = config.get("base_model")
        
        logger.info(
            "LoraManager initialized with Docker container: {}, Server URL: {}, Use SSH: {}",
            self.docker_container_name, self.server_url, self.use_ssh
        )

    
    async def _initialize_environment(self):
        """
        Initializes the environment (local or remote) based on configuration.
        """
        if self.is_remote:
            await self._initialize_ssh_connection()

    async def _initialize_ssh_connection(self):
        """
        Initializes the SSH connection if it is not already established.
        """
        if self._ssh_connection is None:
            try:
                host = self.server_url.split("://")[1].split(":")[0]
                client_key = os.path.expanduser(self.ssh_client_keys[0])
                logger.debug("Establishing SSH connection to {}", host)
                self._ssh_connection = await asyncssh.connect(
                    host=host,
                    username=self.ssh_username,
                    client_keys=client_key,
                    known_hosts=None
                )
                logger.success("SSH connection established to {}", host)
            except Exception as e:
                logger.exception("Failed to establish SSH connection: {}", str(e))
                raise

    async def _execute_command(self, command: str) -> str:
        """
        Executes a command in the appropriate environment (local or remote).
        """
        if self.is_remote:
            return await self._execute_remote_command(command)
        return self._execute_local_command(command)

    async def _execute_remote_command(self, command: str) -> str:
        """
        Executes a command remotely via SSH.
        """
        try:
            await self._initialize_ssh_connection()
            result = await self._ssh_connection.run(command)
            if result.stdout is None:
                raise RuntimeError("No output from remote command")
            return result.stdout
        except Exception as e:
            logger.exception("Error executing remote command via SSH: {}", str(e))
            raise

    def _execute_local_command(self, command: str) -> str:
        """
        Executes a command locally using subprocess.
        """
        try:
            logger.debug("Executing command locally: {}", command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Command failed: {}", result.stderr)
                raise RuntimeError(result.stderr)
            return result.stdout
        except Exception as e:
            logger.exception("Error executing local command: {}", str(e))
            raise

    async def _close_ssh_connection(self):
        """
        Closes the SSH connection if it exists.
        """
        if self._ssh_connection:
            logger.debug("Closing SSH connection.")
            self._ssh_connection.close()
            self._ssh_connection = None

    def __del__(self):
        """
        Destructor to ensure SSH connection is closed.
        """
        if self._ssh_connection:
            asyncio.run(self._close_ssh_connection())

    
    def _get_lora_path(self, lora_name: str) -> Optional[str]:
        """
        Retrieves the path for a given LoRA adapter.

        Args:
            lora_name: The name of the LoRA adapter.

        Returns:
            The path to the LoRA adapter or None if not found.
        """
        path = self.lora_registry.get(lora_name)
        if not path:
            logger.error("LoRA adapter {} not found in registry.", lora_name)
        return path

    def _infer_base_model_from_name(self, lora_name: str) -> str:
        """
        Infers the base model name from the LoRA adapter name.

        Args:
            lora_name: The name of the LoRA adapter.

        Returns:
            The inferred base model name.
        """
        # Add logic here based on naming conventions or configuration
        if "Meta-Llama" in lora_name:
            return "meta-llama/llama-3.1-8b"
        logger.warning("Unable to infer base model for LoRA: {}", lora_name)
        return "unknown-base-model"

    async def ensure_lora_loaded(self, lora_name: str, hf_token: Optional[str] = None, base_model: str = "base-model-name") -> bool:
        """
        Ensures a LoRA is loaded and ready for inference.
        Returns True if successful, False if failed.
        """
        try:
            async with self.lora_locks[lora_name]:
                # Check if the LoRA is already loaded
                if await self._is_lora_loaded(lora_name):
                    logger.debug("LoRA {} is already loaded", lora_name)
                    return True

                # Download the LoRA adapter if not already available
                if not await self._is_lora_available(lora_name):
                    success = await self._download_lora(lora_name, hf_token)
                    if not success:
                        logger.error("Failed to download LoRA {}", lora_name)
                        return False

                # Load the LoRA adapter
                success = await self._load_lora(lora_name)
                if not success:
                    logger.error("Failed to load LoRA {}", lora_name)
                    return False

                # Verify the LoRA is ready for inference
                if not await self._is_lora_ready_for_inference(lora_name):
                    logger.error("LoRA {} is not ready for inference after loading.", lora_name)
                    return False

                return True
        except Exception as e:
            logger.exception("Critical error in load process for LoRA {}: {}", lora_name, str(e))
            return False


    async def _is_lora_ready_for_inference(self, lora_name: str) -> bool:
        """
        Checks if a LoRA adapter is currently loaded in vLLM by checking loaded adapters.
        """
        try:
            logger.debug("Checking if LoRA {} is loaded", lora_name)

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/v1/loaded_loras") as response:
                    if response.status != 200:
                        logger.warning(
                            "Failed to check loaded LoRAs. Status: {}, Response: {}", 
                            response.status, await response.text()
                        )
                        return False
                    
                    loaded_loras = await response.json()
                    if lora_name in loaded_loras:
                        logger.debug("LoRA {} is loaded and ready.", lora_name)
                        return True
                    
                    logger.debug("LoRA {} is not currently loaded.", lora_name)
                    return False

        except Exception as e:
            logger.exception("Error checking if LoRA {} is loaded: {}", lora_name, str(e))
            return False

    async def _is_lora_available(self, lora_name: str) -> bool:
        """
        Checks if a LoRA adapter is available in the execution environment.

        Args:
            lora_name: The name of the LoRA adapter.

        Returns:
            True if the LoRA adapter exists, False otherwise.
        """
        try:
            lora_path = self._get_lora_path(lora_name)
            command = f"test -d {lora_path}"  # Shell command to check if directory exists
            await self._execute_command(command)  # This will handle local or remote execution
            return True
        except RuntimeError:
            logger.debug("LoRA adapter {} is not available at {}", lora_name, lora_path)
            return False


    async def _is_lora_loaded(self, lora_name: str, test_prompt: Optional[str] = "test") -> bool:
        """
        Checks if a LoRA adapter is currently loaded in vLLM by performing a test inference.

        Args:
            lora_name: The name of the LoRA adapter to check.
            test_prompt: A simple prompt to test inference.

        Returns:
            True if the LoRA is loaded and responds to inference, False otherwise.
        """
        try:
            logger.debug("Verifying if LoRA {} is loaded by performing inference.", lora_name)

            # Construct the payload for the /v1/completions endpoint
            payload = {
                "model": lora_name,
                #"prompt": "Ping",
                "messages": [{"role": "user", "content": "Health check"}],
                "max_tokens": 5,
            }
            headers = {"Content-Type": "application/json"}

            # Create a curl command for debugging
            curl_command = (
                f"curl -X POST {self.server_url}/v1/chat/completions "
                f"-H 'Content-Type: application/json' "
                f"-d '{json.dumps(payload)}'"
            )
            logger.debug("Generated curl command for debugging:\n{}", curl_command)

            # Send a request to the vLLM server
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/v1/chat/completions", json=payload, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(
                            "LoRA {} test inference failed with status {}: {}",
                            lora_name, response.status, await response.text()
                        )
                        return False

                    # Parse the response
                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        logger.info("LoRA {} is loaded and functional.", lora_name)
                        return True

                    logger.warning("LoRA {} did not respond to inference as expected.", lora_name)
                    return False

        except Exception as e:
            logger.exception("Error verifying if LoRA {} is loaded: {}", lora_name, str(e))
            return False


    async def _download_lora(self, lora_name: str, hf_token: Optional[str] = None) -> bool:
        """
        Downloads a LoRA adapter from HuggingFace Hub using snapshot_download.
        Handles execution context implicitly (local, Docker, or remote).

        Args:
            lora_name: The name of the LoRA adapter.
            hf_token: Hugging Face API token.

        Returns:
            True if the download was successful or already exists, False otherwise.
        """
        try:
            logger.info(f"Starting download of LoRA {lora_name}")

            # Use the provided token or a default one
            token = self.hf_token or os.getenv("HF_TOKEN")

            # Define the writable directory for the Hugging Face cache
            cache_dir = self.huggingface_adapters_dir

            # Prepare the snapshot_download command/script
            snapshot_command = f"""
            export HF_HOME={cache_dir}
            python3 -c "from huggingface_hub import snapshot_download; snapshot_download(
                '{lora_name}', cache_dir='{cache_dir}', token='{token}')"
            """

            # Use remote or local execution to run the command
            if self.is_remote:
                logger.debug("Executing snapshot_download remotely.")
                await self._execute_remote_command(snapshot_command)
            else:
                logger.debug("Executing snapshot_download locally.")
                os.environ["HF_HOME"] = cache_dir
                snapshot_download(repo_id=lora_name, cache_dir=cache_dir, token=token)

            # Register the downloaded adapter
            adapter_path = os.path.join(cache_dir, "snapshots", os.listdir(os.path.join(cache_dir, "snapshots"))[0])
            self.lora_registry[lora_name] = {
                "path": adapter_path,
                "base_model": self._infer_base_model_from_name(lora_name)
            }

            logger.success(f"Successfully downloaded and registered LoRA {lora_name} at {adapter_path}")
            return True

        except Exception as e:
            logger.exception(f"Error downloading LoRA {lora_name}: {str(e)}")
            return False

    async def _load_lora(self, lora_name: str) -> bool:
        """
        Loads a LoRA adapter into vLLM using the /v1/load_lora_adapter endpoint.
        """
        try:
            # Retrieve the LoRA metadata from the registry
            lora_config = self.lora_registry.get(lora_name)
            if not lora_config:
                logger.error("LoRA metadata for {} not found in registry.", lora_name)
                return False

            # Get the active base model
            active_base_model = lora_config.get('base_model', None)
            if active_base_model is None:
                logger.error("Active base model not found for LoRA {}", lora_name)
                raise Exception("Active base model not found for LoRA {}", lora_name)
            
            # Construct the payload according to vLLM documentation
            payload = {
                "lora_name": lora_name,
                "lora_path": lora_config["path"]
            }
            headers = {"Content-Type: application/json"}

            # Generate curl command for debugging
            curl_command = (
                f"curl -X POST {self.server_url}/v1/load_lora_adapter "
                f"-H 'Content-Type: application/json' "
                f"-d '{json.dumps(payload)}'"
            )
            logger.debug("Generated curl command for debugging:\n{}", curl_command)

            # Log the request details
            logger.debug("Sending POST request to: {}", f"{self.server_url}/v1/load_lora_adapter")
            logger.debug("Request payload: {}", json.dumps(payload, indent=2))
            logger.debug("Request headers: {}", headers)

            # Send the request to the vLLM server
            timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/v1/load_lora_adapter", 
                    json=payload, 
                    headers=headers,
                    timeout=timeout
                ) as response:
                    logger.debug("Response status: {}", response.status)
                    logger.debug("Response text: {}", await response.text())

                    if response.status != 200:
                        logger.error(
                            "Failed to load LoRA {}. Status: {}, Response: {}",
                            lora_name, response.status, await response.text()
                        )
                        return False

                    logger.info("Load request successful for LoRA {}. Waiting for adapter to be ready...", lora_name)
                    
                    # Wait for the LoRA to be ready with timeout
                    max_retries = 10
                    retry_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        if await self._is_lora_ready_for_inference(lora_name):
                            logger.success(
                                "Successfully loaded and verified LoRA {} onto base model {}", 
                                lora_name, active_base_model
                            )
                            return True
                            
                        logger.debug(
                            "LoRA not ready yet, attempt {}/{}. Waiting {} seconds...", 
                            attempt + 1, max_retries, retry_delay
                        )
                        await asyncio.sleep(retry_delay)
                    
                    logger.error("Timed out waiting for LoRA {} to be ready", lora_name)
                    return False

        except Exception as e:
            logger.exception("Error loading LoRA {}: {}", lora_name, str(e))
            return False



    async def _unload_lora(self, lora_name: str) -> bool:
        """Unloads a LoRA from vLLM."""
        try:
            command = (
                f"curl -X POST {self.server_url}/v1/unload_lora_adapter "
                f"-H 'Content-Type: application/json' "
                f"-d '{{\"lora_name\": \"{lora_name}\"}}'"
            )
            await self._execute_remote_command(command)
            logger.info("Successfully unloaded LoRA {}", lora_name)
            return True
        except Exception as e:
            logger.exception("Error unloading LoRA {}: {}", lora_name, str(e))
            return False


    async def list_available_loras_in_directory(self) -> List[str]:
        """
        Lists all available LoRA adapters in the standard Hugging Face cache directory
        and updates the registry with metadata.

        Returns:
            A list of available LoRA adapter names.
        """
        try:
            # Use the standard Hugging Face cache directory
            cache_dir = "/root/.cache/huggingface/hub"

            # List all directories in the cache directory
            command = f"ls -1 {cache_dir}"
            logger.debug(f"Executing command: {command}")

            output = await self._execute_command(command)
            if not output:
                logger.warning("No LoRA adapters found.")
                return []

            # Update the registry with metadata
            self.lora_registry = {}
            for line in output.splitlines():
                if line.startswith("models--") and line.endswith("_adapter"):
                    # Convert directory name to LoRA name
                    lora_name = line.replace("models--", "").replace("--", "/")
                    lora_path = os.path.join(cache_dir, line)

                    # Add to the registry
                    self.lora_registry[lora_name] = {
                        "path": lora_path,
                        "base_model": self._infer_base_model_from_name(lora_name)
                    }

            if not self.lora_registry:
                logger.warning("No valid LoRA adapters found in the standard Hugging Face cache directory.")
                return []

            return list(self.lora_registry.keys())

        except Exception as e:
            logger.exception("Error listing available LoRA adapters: {}", str(e))
            return []



async def perform_inference(lora_manager: LoraManager, prompt: str) -> str:
    """
    Performs inference using the currently loaded LoRA adapter via the OpenAI-compatible API.

    Args:
        lora_manager: An instance of LoraManager.
        prompt: The prompt to use for inference.

    Returns:
        The inference result as a string.
    """
    try:
        # Set up the OpenAI client to point to the vLLM server
        openai.api_key = "dummy"  # API key is not required for local vLLM server
        openai.api_base = lora_manager.server_url  # vLLM server URL
        messages = [{"role": "user", "content": "Ping"}]

        # Perform inference using the OpenAI-compatible API
        client =openai.AsyncOpenAI(api_key="dummy", api_base=lora_manager.server_url)
        response = await openai.chat.completions.create(
            model=lora_manager.base_model,  # Replace with your base model name
            messages =messages,
            max_tokens=10,
            temperature=0.7,
        
        )

        # Extract and return the generated text
        return response.choices[0].text.strip()

    except Exception as e:
        logger.exception("Error during inference: {}", str(e))
        return f"Error: {str(e)}"


async def use_lora_adapter(lora_manager: LoraManager, current_lora: Optional[str], new_lora: str, prompt: str) -> str:
    """
    Unloads the current LoRA adapter, loads a new one, waits for it to load, and performs inference.

    Args:
        lora_manager: An instance of LoraManager.
        current_lora: The name of the currently loaded LoRA adapter (or None if none is loaded).
        new_lora: The name of the new LoRA adapter to load.
        prompt: The prompt to use for inference.

    Returns:
        The inference result as a string.
    """
    try:
        # Step 1: Unload the current LoRA adapter (if any)
        if current_lora:
            logger.info("Unloading current LoRA adapter: {}", current_lora)
            success = await lora_manager._unload_lora(current_lora)
            if not success:
                logger.error("Failed to unload LoRA adapter: {}", current_lora)
                return "Error: Failed to unload current LoRA adapter."

        # Step 2: Load the new LoRA adapter
        logger.info("Loading new LoRA adapter: {}", new_lora)
        success = await lora_manager.ensure_lora_loaded(new_lora)
        if not success:
            logger.error("Failed to load LoRA adapter: {}", new_lora)
            return "Error: Failed to load new LoRA adapter."

        # Step 3: Wait until the new LoRA adapter is loaded
        logger.info("Waiting for LoRA adapter {} to load...", new_lora)
        while not await lora_manager._is_lora_loaded(new_lora):
            await asyncio.sleep(1)  # Poll every second
        logger.success("LoRA adapter {} is now loaded.", new_lora)

        # Step 4: Perform inference using the new LoRA adapter
        logger.info("Performing inference with LoRA adapter: {}", new_lora)
        inference_result = await perform_inference(lora_manager, prompt)
        return inference_result

    except Exception as e:
        logger.exception("Error in use_lora_adapter: {}", str(e))
        return f"Error: {str(e)}"


async def main():
    # Load environment variables Globally
    project_dir = get_project_root()
    load_env_file(env_type="backend")
    
    # Initialize the LoraManager
    config = {
        "container_name": "vllm-openai",  # Docker container name
        "server_url": "http://192.168.86.49:30002",  # vLLM server URL 
        "ssh_username": "graham",
        "ssh_client_keys": ["~/.ssh/id_ed25519"],
        "use_ssh": True,
        "base_model": "meta-llama/llama-3.1-8b"
    }
    lora_manager = LoraManager(config)

    # Download the LoRA adapter
    await lora_manager._download_lora("grahamaco/Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter")

    # List available LoRA adapters
    available_loras = await lora_manager.list_available_loras_in_directory()
    if not available_loras:
        logger.error("No LoRA adapters found in the Hugging Face cache directory.")
        return

    logger.info("Available LoRA adapters: {}", available_loras)

    # Example: Select the first LoRA adapter for inference
    new_lora = available_loras[0]
    prompt = "What is the capital of France?"

    # Perform inference with the selected LoRA adapter
    result = await use_lora_adapter(lora_manager, None, new_lora, prompt)
    print("Inference Result:", result)

async def lora_test_usage():
     # Load environment variables Globally
    project_dir = get_project_root()
    load_env_file(env_type="backend")
    
    # Initialize the LoraManager
    config = {
        "container_name": "vllm-openai",  # Docker container name
        "server_url": "http://192.168.86.49:30002",  # vLLM server URL 
        "ssh_username": "graham",
        "ssh_client_keys": ["~/.ssh/id_ed25519"],
        "use_ssh": True,
        "base_model": "meta-llama/llama-3.1-8b"
    }
    lora_manager = LoraManager(config)
    # make normal query with litellm

    # make new request with lora

    # Lora does not exist, Load Lora and wait for it to load, 
    # if it fails, log error and default to previous working base? model
    # Use all the endpoints you created



    return
    

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())