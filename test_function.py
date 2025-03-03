"""
This module provides a function to generate responses using the litellm package.
"""

from typing import cast
import litellm
from loguru import logger

# Initialize logger
logger.add("debug.log", level="DEBUG")

def generate_response(prompt: str) -> str:
    """
    Generates a response using litellm.
    
    Args:
        prompt: The input prompt to generate a response for
        
    Returns:
        str: The generated response text or error message
    """
    logger.info(f"Received prompt: {prompt}")

    try:
        response = litellm.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = cast(str, response.choices[0].message.content)
        logger.info(f"Generated response: {response_text}")
        
        return response_text

    except (litellm.OpenAIError, ValueError, TypeError) as e:
        logger.error(f"An error occurred: {e}")
        return "Error generating response"

# Example usage
output = generate_response("Tell me a joke.")
print(output) 