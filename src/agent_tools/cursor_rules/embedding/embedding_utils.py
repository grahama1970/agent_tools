from functools import lru_cache
import os
import time
import asyncio
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore
from typing import Dict, List, Union, Tuple, Optional, Any, cast
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv
import warnings
from sentence_transformers import SentenceTransformer  # type: ignore

# Global variables for shared model and tokenizer
_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_sentence_transformer: Optional[SentenceTransformer] = None
_tqdm_lock: Optional[Any] = None

# Load environment variables if needed
load_dotenv()

# Default model configuration
DEFAULT_MODEL_NAME = 'nomic-ai/nomic-embed-text-v2-moe'
DEFAULT_MODEL_DIR = './models/huggingface/'
DEFAULT_EMBEDDING_DIM = 768

# Batching and Workers
def init_worker(model_name: str, model_dir: str, progress_lock: Optional[Any] = None) -> None:
    """Initialize the worker process."""
    global _model, _tokenizer, _tqdm_lock
    _tqdm_lock = progress_lock  # Store the lock in global worker state
    _model, _tokenizer = _load_model_and_tokenizer(model_dir, model_name)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply average pooling to model outputs."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Huggingface Embedding Model
@lru_cache(maxsize=1)
def _load_model_and_tokenizer(model_dir: str = DEFAULT_MODEL_DIR, 
                             model_name: str = DEFAULT_MODEL_NAME) -> Tuple[Any, Any]:
    """Load the model and tokenizer for local embedding."""
    warnings.filterwarnings('ignore', message='`resume_download` is deprecated and will be removed in version 1.0.0')
    
    try:
        # Ensure the model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f'Created model directory at {model_dir}')
        
        # Log the model directory being used
        logger.info(f'Using model directory: {model_dir}')
        
        # Load the tokenizer and model
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)
        
        # Log the model files in the cache directory
        model_files = os.listdir(model_dir)
        logger.info(f'Model files in cache directory: {model_files}')
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            logger.info('Using GPU for embeddings.')
        else:
            logger.info('Using CPU for embeddings.')
        
        # Log the load time
        load_time = time.time() - start_time
        logger.info(f'Model {model_name} loaded successfully in {load_time:.2f} seconds and stored in {model_dir}')
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        raise


def get_model_and_tokenizer(
        model_dir: str = DEFAULT_MODEL_DIR, 
        model_name: str = DEFAULT_MODEL_NAME
) -> Tuple[Any, Any]:
    """Get or lazily initialize model and tokenizer."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = _load_model_and_tokenizer(model_dir, model_name)
    return _model, _tokenizer


def ensure_text_has_prefix(text: str) -> str:
    """Ensure text has the required prefix for the Nomic Embed model."""
    # Check if text already has a prefix
    if text.startswith("search_query:") or text.startswith("search_document:"):
        return text
    
    # Add the document prefix by default
    return f"search_document: {text}"


async def create_embedding(text: str, embedder_config: Optional[Dict[str, Any]] = None) -> Dict[str, Union[List[float], Dict[str, Any]]]:
    """Generate an embedding using a local model."""
    if embedder_config is None:
        embedder_config = {
            'location': 'local', 
            'model_name': DEFAULT_MODEL_NAME, 
            'model_dir': DEFAULT_MODEL_DIR
        }
    
    model_name = embedder_config.get('model_name', DEFAULT_MODEL_NAME)
    model_dir = embedder_config.get('model_dir', DEFAULT_MODEL_DIR)
    
    # Ensure text has the required prefix
    text = ensure_text_has_prefix(text)
    
    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_dir, model_name)
    
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input = {k: v.to('cuda') for (k, v) in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        embedding = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1).cpu().tolist()[0]
    
    metadata = {
        'embedding_model': model_name, 
        'embedding_timestamp': datetime.now(timezone.utc).isoformat(), 
        'embedding_method': 'local',
        'embedding_dim': len(embedding)
    }
    return {'embedding': embedding, 'metadata': metadata}


def create_embedding_sync(text: str, embedder_config: Optional[Dict[str, Any]] = None) -> Dict[str, Union[List[float], Dict[str, Any]]]:
    """Generate an embedding using a shared model."""
    if embedder_config is None:
        embedder_config = {
            'model_name': DEFAULT_MODEL_NAME,
            'model_dir': DEFAULT_MODEL_DIR
        }
    
    # Ensure text has the required prefix
    text = ensure_text_has_prefix(text)
    
    # Get model and tokenizer using the cached loader
    model, tokenizer = get_model_and_tokenizer(
        model_dir=embedder_config.get('model_dir', DEFAULT_MODEL_DIR),
        model_name=embedder_config.get('model_name', DEFAULT_MODEL_NAME)
    )
    
    # Encode the input text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        encoded_input = {k: v.to('cuda') for (k, v) in encoded_input.items()}
    
    # Generate the embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
        embedding = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1).cpu().tolist()[0]
    
    # Prepare metadata
    metadata = {
        'embedding_model': embedder_config.get('model_name', DEFAULT_MODEL_NAME),
        'embedding_timestamp': datetime.now(timezone.utc).isoformat(),
        'embedding_method': 'local',
        'embedding_dim': len(embedding)
    }
    
    return {'embedding': embedding, 'metadata': metadata}


# Sentence Transformer
@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model with caching."""
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name, trust_remote_code=True)


def get_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Get or lazily initialize the SentenceTransformer model."""
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = _load_sentence_transformer(model_name)
    return _sentence_transformer


def create_embedding_with_sentence_transformer(
    text: str, 
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_name: str = "passage"
) -> Dict[str, Union[List[float], Dict[str, Any]]]:
    """Generate an embedding using SentenceTransformer."""
    # Get the model
    model = get_sentence_transformer(model_name)
    
    # Encode the text
    # SentenceTransformer will handle the prefix based on prompt_name
    embedding = model.encode(text, prompt_name=prompt_name).tolist()
    
    # Prepare metadata
    metadata = {
        'embedding_model': model_name,
        'embedding_timestamp': datetime.now(timezone.utc).isoformat(),
        'embedding_method': 'sentence_transformer',
        'embedding_dim': len(embedding),
        'prompt_name': prompt_name
    }
    
    return {'embedding': embedding, 'metadata': metadata}


async def main() -> None:
    """Test the embedding functionality."""
    print('Testing embedding functionality')
    
    # Test with default model
    embedding_result = await create_embedding('Hello, world!')
    result_dict = cast(Dict[str, Any], embedding_result)
    print(f"Embedding dimension: {len(result_dict['embedding'])}")
    print(f"Model used: {result_dict['metadata']['embedding_model']}")
    
    # Test with SentenceTransformer
    st_embedding = create_embedding_with_sentence_transformer('Hello, world!')
    st_result_dict = cast(Dict[str, Any], st_embedding)
    print(f"SentenceTransformer embedding dimension: {len(st_result_dict['embedding'])}")
    print(f"SentenceTransformer model used: {st_result_dict['metadata']['embedding_model']}")


if __name__ == '__main__':
    asyncio.run(main()) 