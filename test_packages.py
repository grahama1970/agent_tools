from loguru import logger
import litellm

# Configure logger
logger.add("test.log", rotation="1 MB")

def test_completion():
    logger.info("Testing LiteLLM completion...")
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )
        logger.success(f"Got response: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f"Error during completion: {e}")

if __name__ == "__main__":
    test_completion() 