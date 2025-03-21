"""LLM API call caching utilities.

This module provides caching functionality for LLM API calls,
reducing costs and improving performance by avoiding duplicate requests.
It implements both in-memory and disk-based caching with statistics tracking.

The module implements the following caching features:
1. Memory cache for fast access to recent results
2. Disk cache for persistence between program runs
3. Configurable cache size and eviction policies
4. Cache statistics for monitoring effectiveness
5. Deterministic cache key generation

Official documentation:
- diskcache: https://grantjenks.com/docs/diskcache/
- hashlib: https://docs.python.org/3/library/hashlib.html
- json: https://docs.python.org/3/library/json.html
- os: https://docs.python.org/3/library/os.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- tempfile: https://docs.python.org/3/library/tempfile.html
- typing: https://docs.python.org/3/library/typing.html

Expected input/output:
- initialize_cache: Takes optional cache directory, initializes and returns the cache instance
- get_from_cache: Takes LLM configuration, returns cached result or None
- add_to_cache: Takes LLM configuration and result, caches the result
- compute_cache_key: Takes LLM configuration, returns a unique cache key
- get_cache_stats: Returns statistics about cache usage
- cache_hit_rate: Returns the cache hit rate as a float
"""

import os
import json
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache

try:
    from diskcache import Cache
except ImportError:
    # If diskcache is not available, use a simple dict-based cache
    Cache = dict

logger = logging.getLogger(__name__)

# Global cache instance
_cache = None

# Cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_requests": 0
}

# Default cache configuration
DEFAULT_CACHE_SIZE = 1_000_000_000  # 1GB
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "dualipa_llm_cache")


def initialize_cache(cache_dir: Optional[str] = None, size_limit: int = DEFAULT_CACHE_SIZE) -> Any:
    """Initialize the LLM cache.
    
    This function initializes both the in-memory and disk-based cache
    for LLM requests. It ensures the cache directory exists and configures
    the cache with appropriate size limits and eviction policies.
    
    Args:
        cache_dir: Optional directory to store cache files, defaults to a temp directory
        size_limit: Maximum size of the cache in bytes
        
    Returns:
        Initialized cache instance
    """
    global _cache
    
    # Use default cache directory if none provided
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Initialize the disk cache
        if isinstance(Cache, type):  # Check if diskcache is available
            _cache = Cache(directory=cache_dir, size_limit=size_limit)
            logger.info(f"Initialized disk cache at {cache_dir} with size limit {size_limit} bytes")
        else:
            # Fallback to a simple dictionary cache
            _cache = {}
            logger.warning("diskcache not available, using in-memory dictionary cache")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        # Fallback to in-memory cache
        _cache = {}
    
    # Reset cache statistics
    clear_stats()
    
    return _cache


def clear_cache():
    """Clear the LLM cache and reset statistics.
    
    This function clears all cached results and resets the cache
    statistics counters.
    """
    global _cache
    
    if _cache is None:
        initialize_cache()
        return
    
    try:
        if isinstance(_cache, dict):
            _cache.clear()
        else:
            _cache.clear()
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
    
    # Reset statistics
    clear_stats()
    logger.info("Cache cleared and statistics reset")


def clear_stats():
    """Reset the cache statistics counters."""
    global _cache_stats
    
    _cache_stats = {
        "hits": 0,
        "misses": 0,
        "total_requests": 0
    }


def get_cache_stats() -> Dict[str, int]:
    """Get the current cache statistics.
    
    Returns:
        Dictionary with cache statistics including hits, misses, and total requests
    """
    return _cache_stats.copy()


def cache_hit_rate() -> float:
    """Calculate the cache hit rate.
    
    This function calculates the percentage of cache hits
    out of the total number of cache requests.
    
    Returns:
        Hit rate as a float between 0.0 and 1.0, or 0.0 if no requests made
    """
    if _cache_stats["total_requests"] == 0:
        return 0.0
    
    return _cache_stats["hits"] / _cache_stats["total_requests"]


def compute_cache_key(config: Dict[str, Any]) -> str:
    """Compute a unique cache key for an LLM request configuration.
    
    This function generates a deterministic hash-based key from the
    request configuration, ensuring identical requests have the same key.
    
    Args:
        config: LLM request configuration
        
    Returns:
        Unique cache key as a string
    """
    # Create a normalized and sorted representation of the config
    # This ensures consistent key generation regardless of dict order
    
    # Extract key components that affect the response
    key_components = {}
    
    # Model name is essential for the key
    if "model" in config:
        key_components["model"] = config["model"]
    
    # Temperature affects randomness
    if "temperature" in config:
        key_components["temperature"] = config["temperature"]
    
    # Messages are the core content
    if "messages" in config:
        key_components["messages"] = config["messages"]
    
    # Other parameters that affect generation
    for param in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
        if param in config:
            key_components[param] = config[param]
    
    # Convert to a consistent string representation
    try:
        # Sort keys for consistent ordering
        config_str = json.dumps(key_components, sort_keys=True)
        
        # Generate SHA-256 hash
        key = hashlib.sha256(config_str.encode()).hexdigest()
        return key
    except Exception as e:
        logger.error(f"Failed to compute cache key: {e}")
        # Fallback: use the hash of the string representation
        return str(hash(str(config)))


def get_from_cache(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get a result from the cache if available.
    
    This function checks if a result for the given configuration
    is available in the cache and updates the cache statistics.
    
    Args:
        config: LLM request configuration
        
    Returns:
        Cached result or None if not found
    """
    global _cache_stats
    
    if _cache is None:
        initialize_cache()
    
    # Compute the cache key
    key = compute_cache_key(config)
    
    # Update request counter
    _cache_stats["total_requests"] += 1
    
    try:
        # Check if key exists in cache
        if key in _cache:
            result = _cache[key]
            _cache_stats["hits"] += 1
            logger.debug(f"Cache hit for key {key[:8]}...")
            return result
    except Exception as e:
        logger.error(f"Error accessing cache: {e}")
    
    # Cache miss
    _cache_stats["misses"] += 1
    logger.debug(f"Cache miss for key {key[:8]}...")
    return None


def add_to_cache(config: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Add a result to the cache.
    
    This function adds an LLM API response to the cache using
    the request configuration as the key.
    
    Args:
        config: LLM request configuration
        result: LLM API response to cache
    """
    if _cache is None:
        initialize_cache()
    
    # Compute the cache key
    key = compute_cache_key(config)
    
    try:
        # Add to cache
        _cache[key] = result
        logger.debug(f"Added result to cache with key {key[:8]}...")
    except Exception as e:
        logger.error(f"Failed to add to cache: {e}")


@lru_cache(maxsize=1024)
def get_cache_dir() -> str:
    """Get the current cache directory.
    
    Returns:
        Path to the cache directory as a string
    """
    if _cache is None:
        return DEFAULT_CACHE_DIR
    
    if isinstance(_cache, dict):
        return "memory-cache"
    
    try:
        return _cache.directory
    except:
        return DEFAULT_CACHE_DIR