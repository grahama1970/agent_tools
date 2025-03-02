from loguru import logger
import litellm
from typing import Optional
from litellm.types.utils import ModelResponse

# Configure logger - validated method
logger.add("llm_calls.log", rotation="1 day")

def chat_with_llm(prompt: str) -> Optional[str]:
    """Chat with LLM and log the interaction."""
    try:
        logger.info(f"Sending prompt: {prompt}")
        
        # Validated completion method
        response: ModelResponse = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        if response and response.choices and response.choices[0].message:
            result = response.choices[0].message.content
            if result:
                logger.success(f"Received response: {result}")
                return result
        
        logger.error("Received empty or invalid response")
        return None
        
    except Exception as e:
        logger.error(f"Error during LLM call: {str(e)}")
        raise

if __name__ == "__main__":
    chat_with_llm("Tell me a short joke about programming.") 