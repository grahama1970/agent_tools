"""Tests for LLM cache functionality.

This module tests the caching capabilities of LLM API calls,
ensuring that repeated identical requests use the cache instead of
making actual API calls. This improves performance and reduces costs
by avoiding duplicate requests.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- diskcache: https://grantjenks.com/docs/diskcache/
- tempfile: https://docs.python.org/3/library/tempfile.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- hashlib: https://docs.python.org/3/library/hashlib.html

Expected test coverage:
- Cache initialization and configuration
- Cache hit rate tracking
- Real-world cache effectiveness
- Cache key generation consistency
"""

import os
import json
import pytest
import asyncio
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

from agent_tools.dualipa.qa.llm.generation import generate_markdown_qa_pairs
from agent_tools.dualipa.qa.llm.retry_llm_call import retry_llm_call

# Import the module to be created
from agent_tools.dualipa.qa.utils.cache import (
    initialize_cache, 
    get_cache_stats,
    compute_cache_key, 
    clear_cache,
    cache_hit_rate,
    get_from_cache,
    add_to_cache
)


@pytest.fixture
def sample_markdown_content():
    """Provides sample markdown content for testing."""
    return """
    # Cache Testing
    
    This is a sample markdown document used for testing cache functionality.
    
    ## Key Features
    
    - Memory cache for fast access
    - Disk cache for persistence
    - Automatic key generation
    - Cache statistics tracking
    """


@pytest.fixture
def sample_llm_config():
    """Provides a sample LLM configuration for testing."""
    return {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a question about caching."}
        ],
        "temperature": 0.7
    }


@pytest.fixture
def cache_dir():
    """Creates a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.mark.asyncio
async def test_cache_hit_real(sample_markdown_content, cache_dir):
    """Test real-world cache effectiveness with actual content.
    
    This test verifies that:
    1. The cache is properly initialized
    2. The first request is a cache miss
    3. The second identical request is a cache hit
    4. The cached result matches the original result
    
    Input:
        - Sample markdown content
        - Temporary cache directory
    
    Expected output:
        - First request shows cache miss in stats
        - Second request shows cache hit in stats
        - Generated QA pairs from both requests match
    
    Dependencies:
        - initialize_cache from cache module
        - generate_markdown_qa_pairs from generation module
        - get_cache_stats from cache module
    """
    # Initialize cache with test directory
    initialize_cache(cache_dir=str(cache_dir))
    
    # Clear any existing stats
    clear_cache()
    
    # First request should be a cache miss
    first_result = await generate_markdown_qa_pairs(
        markdown_content=sample_markdown_content,
        temperature=0.5,
        max_pairs=2
    )
    
    # Get cache stats after first request
    stats_after_first = get_cache_stats()
    assert stats_after_first["hits"] == 0, "First request should be a cache miss"
    assert stats_after_first["misses"] >= 1, "First request should be recorded as a miss"
    
    # Second identical request should be a cache hit
    second_result = await generate_markdown_qa_pairs(
        markdown_content=sample_markdown_content,
        temperature=0.5,
        max_pairs=2
    )
    
    # Get cache stats after second request
    stats_after_second = get_cache_stats()
    assert stats_after_second["hits"] >= 1, "Second request should be a cache hit"
    
    # Results should match
    assert len(first_result) == len(second_result), "Both results should have same number of QA pairs"
    
    # Verify that both results have the same content
    # We can't directly compare the objects as they might have different timestamps
    # So we compare the questions and answers
    for i in range(len(first_result)):
        assert first_result[i].question == second_result[i].question, "Questions should match"
        assert first_result[i].answer == second_result[i].answer, "Answers should match"


@pytest.mark.asyncio
async def test_cache_hit_rate(sample_llm_config, cache_dir):
    """Test cache hit rate calculation and effectiveness.
    
    This test verifies that:
    1. The cache hit rate is correctly calculated
    2. Multiple identical requests result in a high hit rate
    3. Different requests have different cache keys
    
    Input:
        - Sample LLM configurations
        - Temporary cache directory
    
    Expected output:
        - Cache hit rate above threshold after repeated requests
        - Different requests have different cache keys
    
    Dependencies:
        - initialize_cache from cache module
        - retry_llm_call from retry_llm_call module
        - cache_hit_rate from cache module
    """
    # Initialize cache with test directory
    initialize_cache(cache_dir=str(cache_dir))
    
    # Clear any existing stats
    clear_cache()
    
    # Create a mock LLM call function
    async def mock_llm_call(config):
        # Simulate API call
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"question": "What is caching?", "answer": "A technique to store and reuse results.", "reasoning": "Caching improves performance. Oh wait?! It also reduces costs."}'
                    }
                }
            ]
        }
    
    # Test directly with cache functions instead of using retry_llm_call
    # This ensures we're directly testing the cache functionality
    config = sample_llm_config.copy()
    cache_key = compute_cache_key(config)
    
    # First call should be a miss
    result = get_from_cache(config)
    assert result is None, "First call should be a cache miss"
    
    # Add result to cache
    mock_result = await mock_llm_call(config)
    add_to_cache(config, mock_result)
    
    # Next calls should be hits
    for i in range(5):
        result = get_from_cache(config)
        assert result is not None, f"Call {i+1} should be a cache hit"
    
    # Check cache hit rate
    hit_rate = cache_hit_rate()
    assert hit_rate > 0.5, f"Cache hit rate should be above 50%, got {hit_rate*100:.2f}%"
    
    # Verify that different requests have different cache keys
    config1 = sample_llm_config.copy()
    config2 = sample_llm_config.copy()
    config2["temperature"] = 0.8  # Different temperature
    
    key1 = compute_cache_key(config1)
    key2 = compute_cache_key(config2)
    
    assert key1 != key2, "Different configurations should have different cache keys"


@pytest.mark.asyncio
async def test_cache_key_consistency(sample_llm_config):
    """Test that cache keys are consistently generated for identical configs.
    
    This test verifies that:
    1. The same config always generates the same cache key
    2. Order of keys in the config doesn't affect the cache key
    3. Cache keys are unique for different configs
    
    Input:
        - Sample LLM configuration
    
    Expected output:
        - Identical configs produce identical cache keys
        - Different configs produce different cache keys
    
    Dependencies:
        - compute_cache_key from cache module
    """
    # Base config
    config1 = sample_llm_config.copy()
    
    # Same config with keys in different order (should produce same key)
    config2 = {
        "messages": config1["messages"],
        "model": config1["model"],
        "temperature": config1["temperature"]
    }
    
    # Different config (should produce different key)
    config3 = sample_llm_config.copy()
    config3["temperature"] = 0.9
    
    # Generate keys
    key1 = compute_cache_key(config1)
    key2 = compute_cache_key(config2)
    key3 = compute_cache_key(config3)
    
    # Same config should produce same key
    assert key1 == key2, "Identical configs should produce identical cache keys"
    
    # Different config should produce different key
    assert key1 != key3, "Different configs should produce different cache keys"
    
    # Key should be deterministic
    assert compute_cache_key(config1) == key1, "Cache key generation should be deterministic"