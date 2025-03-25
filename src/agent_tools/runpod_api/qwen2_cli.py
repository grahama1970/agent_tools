import click
import os
import subprocess
import runpod
import requests
import json
import time
from loguru import logger

# Configure logger
logger.add("qwen2_cli.log", rotation="10 MB")

# Load .env manually since click runs in a new process
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    
    # Set RunPod API key
    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    if runpod_api_key:
        runpod.api_key = runpod_api_key
        return True
    return False

@click.group()
def cli():
    """CLI for managing Qwen2-72B inference on RunPod."""
    if not load_env():
        logger.error("Failed to load environment variables. Check .env file.")

@cli.command()
def build():
    """Build and push the Docker image."""
    script_path = os.path.join(os.path.dirname(__file__), "build_push_qwen2.sh")
    logger.info(f"Running build script: {script_path}")
    
    try:
        result = subprocess.run([script_path], check=True, text=True, capture_output=True)
        click.echo(result.stdout)
        logger.info("Build and push completed successfully.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Build and push failed with exit code {e.returncode}")
        click.echo(e.stdout)
        click.echo(e.stderr, err=True)
        logger.error(f"Build failed: {e.stderr}")
        exit(1)

@cli.command()
@click.option("--pod-id", help="Pod ID to takedown (default: from pod_id.txt)")
def takedown(pod_id):
    """Takedown the RunPod pod."""
    if not pod_id:
        pod_file = os.path.join(os.path.dirname(__file__), "qwen2-runpod/pod_id.txt")
        if os.path.exists(pod_file):
            with open(pod_file) as f:
                pod_id = f.read().strip()
        else:
            error_msg = "Error: No pod_id provided and pod_id.txt not found."
            click.echo(error_msg)
            logger.error(error_msg)
            exit(1)
    
    if not runpod.api_key:
        error_msg = "Error: RUNPOD_API_KEY not set."
        click.echo(error_msg)
        logger.error(error_msg)
        exit(1)
    
    try:
        click.echo(f"Stopping pod {pod_id}...")
        logger.info(f"Stopping pod {pod_id}")
        runpod.stop_pod(pod_id)
        
        # Wait for pod to stop
        click.echo("Waiting for pod to stop...")
        max_attempts = 10
        for i in range(max_attempts):
            pod_status = runpod.get_pod(pod_id)
            if pod_status["status"] == "STOPPED":
                break
            click.echo(f"Pod status: {pod_status['status']}. Waiting...")
            time.sleep(5)
        
        # Delete the pod
        click.echo(f"Deleting pod {pod_id}...")
        logger.info(f"Deleting pod {pod_id}")
        runpod.delete_pod(pod_id)
        click.echo(f"Pod {pod_id} deleted.")
    except Exception as e:
        error_msg = f"Error during takedown: {str(e)}"
        click.echo(error_msg)
        logger.error(error_msg)
        exit(1)

@cli.command()
