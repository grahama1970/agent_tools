"""
Example of using LiteLLM for LLM integration.

This demonstrates the recommended patterns for:
1. Making LLM API calls with proper error handling
2. Implementing caching for LLM responses
3. Handling streaming responses
4. Processing and validating LLM outputs

Documentation References:
- LiteLLM Documentation: https://docs.litellm.ai/docs/
- OpenAI API Reference: https://platform.openai.com/docs/api-reference
- Related Rule: See `.cursor/rules/003-package-usage.mdc` section on "LLM Integration"
- Async Error Handling: See `.cursor/rules/005-async-patterns.mdc` section on "Error Handling"
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from loguru import logger


class LiteLLMResponse:
    """Simulated LiteLLM response for the example."""
    
    def __init__(self, content: str, model: str = "gpt-3.5-turbo"):
        self.choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
        self.model = model
        self.usage = {
            "prompt_tokens": len(content) // 4,  # Simulated token count
            "completion_tokens": len(content) // 4,
            "total_tokens": len(content) // 2
        }
        self.id = f"chatcmpl-{int(time.time())}"
        self.created = int(time.time())
        self.object = "chat.completion"


class LiteLLMStreamResponse:
    """Simulated LiteLLM streaming response for the example."""
    
    def __init__(self, content: str, model: str = "gpt-3.5-turbo", chunk_size: int = 10):
        self.content = content
        self.model = model
        self.chunk_size = chunk_size
        self.id = f"chatcmpl-{int(time.time())}"
        self.created = int(time.time())
        self.object = "chat.completion.chunk"
        
        # Split content into chunks for streaming
        self.chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            self.chunks.append(chunk)
    
    async def __aiter__(self):
        for i, chunk in enumerate(self.chunks):
            # Simulate network delay
            await asyncio.sleep(0.1)
            
            # Create a response chunk
            response_chunk = {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant" if i == 0 else None,
                            "content": chunk
                        },
                        "finish_reason": "stop" if i == len(self.chunks) - 1 else None
                    }
                ],
                "model": self.model,
                "id": f"{self.id}-{i}",
                "created": self.created + i,
                "object": self.object
            }
            
            yield response_chunk


# Simulated cache for LLM responses
llm_cache = {}


async def get_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    stream: bool = False,
    max_retries: int = 3,
    timeout: float = 30.0
) -> Union[LiteLLMResponse, LiteLLMStreamResponse]:
    """
    Get a completion from an LLM using LiteLLM.
    
    This function demonstrates the proper pattern for LLM API calls:
    1. Use caching to avoid redundant API calls
    2. Implement retry logic with exponential backoff
    3. Handle errors properly
    4. Support streaming responses
    
    Args:
        model: The model to use (e.g., "gpt-3.5-turbo", "gpt-4")
        messages: The conversation messages
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        cache: Whether to use caching
        stream: Whether to stream the response
        max_retries: Maximum number of retries
        timeout: Timeout in seconds
        
    Returns:
        LLM response object
    """
    try:
        # Create a cache key from the request parameters
        cache_key = _create_cache_key(model, messages, temperature, max_tokens)
        
        # Check cache if enabled
        if cache and cache_key in llm_cache:
            logger.info(f"Cache hit for model: {model}")
            return llm_cache[cache_key]
        
        logger.info(f"Sending request to {model} with {len(messages)} messages")
        
        # Implement retry logic
        retries = 0
        while True:
            try:
                # Use asyncio.wait_for to implement timeout
                response = await asyncio.wait_for(
                    _simulate_llm_call(model, messages, temperature, max_tokens, stream),
                    timeout=timeout
                )
                
                # Cache the response if caching is enabled and not streaming
                if cache and not stream:
                    llm_cache[cache_key] = response
                
                return response
            
            except asyncio.TimeoutError:
                logger.warning(f"Request to {model} timed out after {timeout} seconds")
                retries += 1
                if retries >= max_retries:
                    raise TimeoutError(f"Request to {model} timed out after {max_retries} retries")
                
                # Exponential backoff with jitter
                backoff = (2 ** retries) * 1.0 + (0.1 * (asyncio.get_event_loop().time() % 1.0))
                logger.warning(f"Retrying in {backoff:.2f} seconds (attempt {retries}/{max_retries})")
                await asyncio.sleep(backoff)
            
            except Exception as e:
                logger.error(f"Error calling {model}: {e}")
                retries += 1
                if retries >= max_retries:
                    raise
                
                # Exponential backoff with jitter
                backoff = (2 ** retries) * 1.0 + (0.1 * (asyncio.get_event_loop().time() % 1.0))
                logger.warning(f"Retrying in {backoff:.2f} seconds (attempt {retries}/{max_retries})")
                await asyncio.sleep(backoff)
    
    except Exception as e:
        logger.error(f"Error getting completion from {model}: {e}")
        raise


def _create_cache_key(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int]
) -> str:
    """Create a cache key from the request parameters."""
    # Convert the parameters to a string for hashing
    key_data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    return json.dumps(key_data, sort_keys=True)


async def _simulate_llm_call(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stream: bool
) -> Union[LiteLLMResponse, LiteLLMStreamResponse]:
    """
    Simulate an LLM API call.
    
    In a real implementation, this would use the actual LiteLLM library.
    For this example, we're simulating the API call.
    """
    # In a real implementation, you would use:
    # from litellm import acompletion
    # response = await acompletion(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     stream=stream
    # )
    
    # Simulate API call latency
    await asyncio.sleep(0.5)
    
    # Randomly simulate an error (10% chance)
    import random
    if random.random() < 0.1:
        error_types = ["timeout", "rate_limit", "api_error"]
        error_type = random.choice(error_types)
        
        if error_type == "timeout":
            raise asyncio.TimeoutError("Request timed out")
        elif error_type == "rate_limit":
            raise Exception("Rate limit exceeded")
        else:
            raise Exception(f"API error: {error_type}")
    
    # Generate a simple response based on the last message
    last_message = messages[-1]["content"]
    response_text = f"This is a simulated response to: {last_message}"
    
    # Limit the response length if max_tokens is specified
    if max_tokens:
        # Simulate token limit (very rough approximation)
        token_limit = max_tokens * 4  # Assuming ~4 chars per token
        if len(response_text) > token_limit:
            response_text = response_text[:token_limit] + "..."
    
    # Return streaming or non-streaming response
    if stream:
        return LiteLLMStreamResponse(response_text, model)
    else:
        return LiteLLMResponse(response_text, model)


async def process_streaming_response(
    stream_response: LiteLLMStreamResponse,
    callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Process a streaming response from an LLM.
    
    Args:
        stream_response: The streaming response
        callback: Optional callback function to process each chunk
        
    Returns:
        The complete response text
    """
    full_response = ""
    
    try:
        async for chunk in stream_response:
            # Extract the content from the chunk
            content = chunk["choices"][0]["delta"].get("content", "")
            
            if content:
                # Append to the full response
                full_response += content
                
                # Call the callback if provided
                if callback:
                    callback(content)
        
        return full_response
    
    except Exception as e:
        logger.error(f"Error processing streaming response: {e}")
        raise


async def extract_json_from_llm_response(
    model: str,
    prompt: str,
    json_schema: Dict[str, Any],
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Extract structured JSON data from an LLM response.
    
    This function demonstrates how to:
    1. Prompt an LLM for structured data
    2. Parse and validate the response as JSON
    3. Retry with improved prompts if parsing fails
    
    Args:
        model: The model to use
        prompt: The prompt requesting JSON data
        json_schema: The expected JSON schema
        max_retries: Maximum number of retries
        
    Returns:
        Parsed JSON data
    """
    # Enhance the prompt to request valid JSON
    enhanced_prompt = f"""
    {prompt}
    
    IMPORTANT: Respond with valid JSON that follows this schema:
    {json.dumps(json_schema, indent=2)}
    
    Your response must be valid JSON only, with no additional text before or after.
    """
    
    messages = [{"role": "user", "content": enhanced_prompt}]
    
    for attempt in range(max_retries + 1):
        try:
            # Get completion from the model
            response = await get_completion(model, messages, temperature=0.2)
            response_text = response.choices[0]["message"]["content"]
            
            # Try to parse the response as JSON
            try:
                # Find JSON in the response (in case there's additional text)
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                
                # Clean up the string (remove any non-JSON parts)
                json_str = json_str.strip()
                if json_str.startswith('```') and json_str.endswith('```'):
                    json_str = json_str[3:-3].strip()
                
                # Parse the JSON
                parsed_data = json.loads(json_str)
                
                # Validate against the schema (simplified validation)
                for key in json_schema:
                    if key not in parsed_data:
                        raise ValueError(f"Missing required key: {key}")
                
                logger.info(f"Successfully extracted JSON data after {attempt + 1} attempts")
                return parsed_data
            
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    # Add a new message to clarify the issue
                    error_message = f"""
                    I couldn't parse your previous response as valid JSON. Error: {str(e)}
                    
                    Please provide a response in valid JSON format that matches the schema:
                    {json.dumps(json_schema, indent=2)}
                    
                    Make sure your response contains ONLY the JSON object, with no additional text.
                    """
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": error_message})
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: JSON parsing failed: {e}")
                else:
                    logger.error(f"Failed to extract valid JSON after {max_retries + 1} attempts")
                    raise ValueError(f"Failed to extract valid JSON: {e}")
        
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Retry {attempt + 1}/{max_retries}: LLM call failed: {e}")
            else:
                logger.error(f"Failed to get valid response after {max_retries + 1} attempts")
                raise


# Example usage
async def main():
    # Example 1: Basic completion
    try:
        response = await get_completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )
        
        print("Example 1: Basic Completion")
        print(f"Response: {response.choices[0]['message']['content']}")
        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage['total_tokens']}")
        print()
    
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    # Example 2: Streaming response
    try:
        print("Example 2: Streaming Response")
        print("Response: ", end="", flush=True)
        
        stream_response = await get_completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 10"}],
            stream=True
        )
        
        # Process the streaming response
        def print_chunk(chunk):
            print(chunk, end="", flush=True)
        
        full_response = await process_streaming_response(stream_response, print_chunk)
        print("\n")
    
    except Exception as e:
        print(f"\nError in Example 2: {e}")
    
    # Example 3: JSON extraction
    try:
        print("Example 3: JSON Extraction")
        
        schema = {
            "name": "string",
            "age": "number",
            "interests": "array",
            "contact": {
                "email": "string",
                "phone": "string"
            }
        }
        
        prompt = """
        Create a profile for a fictional person with the following information:
        - Name
        - Age
        - List of interests
        - Contact information (email and phone)
        """
        
        json_data = await extract_json_from_llm_response(
            model="gpt-3.5-turbo",
            prompt=prompt,
            json_schema=schema
        )
        
        print(f"Extracted JSON: {json.dumps(json_data, indent=2)}")
    
    except Exception as e:
        print(f"Error in Example 3: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 