import spacy
from loguru import logger
from functools import lru_cache
import subprocess
import sys
from pathlib import Path

@lru_cache(maxsize=1)
def get_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Get cached spaCy model with simplified installation."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model '{model_name}'...")
        try:
            # Use pip directly instead of uv to avoid dependency
            subprocess.run([
                sys.executable, 
                "-m", "pip", 
                "install", 
                f"{model_name}"
            ], check=True)
            return spacy.load(model_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


def count_tokens(text: str) -> int:
    """Count tokens in text using cached spaCy model."""
    nlp = get_spacy_model()
    return len(nlp(text))

def truncate_text_by_tokens(text: str, max_tokens: int = 50) -> str:
    """Truncate text to max_tokens while preserving meaning."""
    nlp = get_spacy_model()
    doc = nlp(text)
    
    if len(doc) <= max_tokens:
        return text
        
    # Get first and last n/2 tokens
    half_tokens = max_tokens // 2
    start_text = ''.join(token.text_with_ws for token in doc[:half_tokens])
    end_text = ''.join(token.text_with_ws for token in doc[-half_tokens:])
    
    return f"{start_text.strip()}... {end_text.strip()}"


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    print(nlp.pipe_names)
    # print('hello')
    # get_spacy_model()
    # print(count_tokens("Hello, world!"))
    # print(truncate_text_by_tokens("Hello, world! This is a test of the truncate function. It should truncate the text to 50 tokens while preserving the meaning of the text.", 50))
