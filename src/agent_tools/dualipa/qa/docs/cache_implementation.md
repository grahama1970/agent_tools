# Cache Implementation for LLM API Requests

## Overview

This document describes the implementation of a caching system for LLM API requests in the DuaLipa QA generation module. The caching system is designed to reduce costs and improve performance by avoiding duplicate API calls.

## Implementation Details

### Components

1. **Cache Storage**
   - Uses `diskcache` library for persistent on-disk caching
   - Falls back to in-memory dictionary if `diskcache` is unavailable
   - Configurable cache directory and size limits

2. **Cache Key Generation**
   - Deterministic hash-based key generation
   - Includes key request parameters: model, temperature, messages, etc.
   - Normalized JSON representation to ensure key consistency

3. **Statistics Tracking**
   - Tracks hits, misses, and total requests
   - Calculates hit rate for performance monitoring
   - Resets on demand for testing and benchmarking

4. **Integration with LLM Pipeline**
   - Seamless integration with `retry_llm_call` function
   - Optional toggle for cache usage (`use_cache` parameter)
   - Graceful error handling for cache failures

### File Structure

1. **`cache.py`**
   - Core cache functionality
   - Statistics tracking
   - Cache key generation

2. **`retry_llm_call.py`**
   - Cache integration with LLM API calls
   - Check cache before API calls
   - Add results to cache after successful calls

3. **`test_cache.py`**
   - Tests for cache functionality
   - Verifies hit rate calculation
   - Tests cache key consistency

## Performance Characteristics

- **Memory Usage**: Controlled by cache size limit (default: 1GB)
- **Disk Space**: Uses efficient serialization for storage
- **Cache Hit Rate**: Improves with repeated similar queries
- **LRU Eviction**: Automatically removes least recently used entries

## Testing Strategy

Tests focus on the core relationships between components, not just validating behavior. They serve as executable documentation of how the cache system interacts with the LLM pipeline.

1. **`test_cache_hit_real`**
   - Documents the relationship between cache and LLM generation
   - Shows how cached results maintain consistency across calls
   - Demonstrates the normal path users will experience
   - Focuses on the main use case: caching identical requests

2. **`test_cache_hit_rate`**
   - Illustrates how statistics are tracked and calculated
   - Demonstrates the direct cache API usage pattern
   - Shows core cache functionality without LLM integration
   - Focuses on understanding cache behavior, not just validating it works

3. **`test_cache_key_consistency`**
   - Documents a critical guaranty: deterministic key generation
   - Shows how configuration variations affect caching
   - Demonstrates the relationship between config structure and cache keys
   - Focuses on understanding the cache key logic, not edge cases

## Future Improvements

1. **TTL Support**
   - Add time-to-live for cache entries to handle model updates

2. **Cache Warmup**
   - Preload cache with common requests for faster startup

3. **Cache Sharing**
   - Enable cache sharing across multiple processes

4. **Cache Analytics**
   - More detailed statistics for optimization

## System Requirements

- **Python Version**: 3.8+
- **Disk Space**: At least 2GB free for cache storage
- **Dependencies**: `diskcache` (optional, falls back to in-memory if unavailable)

## Usage

```python
# Enable cache globally
from agent_tools.dualipa.qa.utils.cache import initialize_cache

# Initialize with custom settings
initialize_cache(cache_dir="/custom/path", size_limit=2_000_000_000)

# Use in LLM calls
from agent_tools.dualipa.qa.llm.retry_llm_call import retry_llm_call

result = await retry_llm_call(config, use_cache=True)  # Cache enabled

# Check cache statistics
from agent_tools.dualipa.qa.utils.cache import get_cache_stats, cache_hit_rate

stats = get_cache_stats()  # Get hit/miss counts
hit_rate = cache_hit_rate()  # Get hit rate (0.0-1.0)
```