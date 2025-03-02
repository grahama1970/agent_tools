from typing import Dict, Optional
from pathlib import Path
import asyncio
import json
import os
from loguru import logger
import litellm
from src.utils.get_project_root import get_project_root, load_env_file


def initialize_cache(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None,
    ttl: int = 60 * 60 * 24 * 7  # 1 week default
):
    """Initialize LiteLLM Redis cache with given parameters"""
    litellm.cache = litellm.Cache(
        type="redis",
        host=host,
        port=port,
        password=password,
        ttl=ttl
    )
    litellm.enable_cache()


def load_config(config_path: Path) -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise


def validate_config(config: Dict) -> Dict:
    """Validate configuration and set defaults"""
    # Validate Arango config
    arango_config = config.get('arango_config', {})
    required_keys = ["hosts", "db_name", "username", "password"]
    missing_keys = [key for key in required_keys if key not in arango_config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
    # Validate and set defaults
    if "excluded_collections" in config and not isinstance(config["excluded_collections"], list):
        raise ValueError("excluded_collections must be a list")
    
    config.setdefault("excluded_collections", [])
    config.setdefault("include_queries", [])
    
    return config


async def test_cache():
    """Test function to verify Redis cache functionality"""
    try:
        unique_prompt = f"Test prompt {os.urandom(8).hex()}"
        messages = [{"role": "user", "content": unique_prompt}]
        
        # First call
        response1 = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=messages,
            caching=True
        )
        print(f"First call cache hit: {response1._hidden_params.get('cache_hit', False)}")
        
        await asyncio.sleep(1)  # Wait for cache to be set
        
        # Second call with same prompt
        response2 = await litellm.acompletion(
            model="gpt-4o-mini", 
            messages=messages,
            caching=True
        )
        print(f"Second call cache hit: {response2._hidden_params.get('cache_hit', False)}")
        print(f"Response IDs:", {
            "first": response1.id,
            "second": response2.id
        })
        
    except Exception as e:
        logger.error(f"Cache test failed: {str(e)}")
        raise


def initialize_environment():
    """Initialize environment and configurations"""
    os.environ['LITELLM_LOG'] = 'DEBUG'
    project_dir = get_project_root()
    
    # Load and validate the intitial config
    from src.settings.config import config
    validate_config(config)

    load_env_file(env_type="backend")
    initialize_cache()
    # we could validate the config here and load the ArangoDB connection?
    # return project_dir, db
    return project_dir


if __name__ == "__main__":
    initialize_environment()
    asyncio.run(test_cache())