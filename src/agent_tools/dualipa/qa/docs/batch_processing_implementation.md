# Batch Processing Implementation

This document outlines the design, implementation, and usage of the batch processing functionality implemented in Task 3.2 of the DuaLipa QA generation module.

## Official Documentation References

- asyncio: https://docs.python.org/3/library/asyncio.html
- logging: https://docs.python.org/3/library/logging.html
- time: https://docs.python.org/3/library/time.html
- typing: https://docs.python.org/3/library/typing.html
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/

## Overview

The batch processing system provides scalable concurrent processing of multiple sections with rate limiting to prevent API overload. It handles worker pool management, chunked processing for large datasets, and detailed performance tracking.

## Core Components

### 1. Semaphore-Based Rate Limiting

The `process_with_semaphore` function wraps any async function with a semaphore to control concurrency:

```python
async def process_with_semaphore(
    semaphore: asyncio.Semaphore,
    func: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """Process a function with semaphore-based rate limiting."""
    async with semaphore:
        # Execute function with controlled concurrency
        return await func(*args, **kwargs)
```

This ensures that only a limited number of operations run concurrently, preventing API rate limit issues and resource exhaustion.

### 2. Batch Processing for Sections

The `batch_process_sections` function handles processing multiple sections with worker pools:

```python
async def batch_process_sections(
    sections: List[Dict[str, Any]],
    config: QAGenerationConfig,
    process_function: Callable[..., Awaitable[T]],
    enable_bidirectional: bool = True,
    chunk_size: Optional[int] = None
) -> List[T]:
    """Process multiple sections in batches with worker pools."""
    # Create semaphore for rate limiting
    worker_count = getattr(config, 'worker_count', config.max_concurrent_requests)
    semaphore = asyncio.Semaphore(worker_count)
    
    # Create and execute tasks with controlled concurrency
    tasks = [
        process_with_semaphore(
            semaphore,
            process_function,
            section,
            config,
            enable_bidirectional
        )
        for section in sections
    ]
    
    # Execute all tasks with controlled concurrency
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions and return results
    return [[] if isinstance(r, Exception) else r for r in results]
```

### 3. Advanced Batch Processing with Statistics

For more detailed tracking, the `batch_process_with_stats` function provides comprehensive metrics:

```python
async def batch_process_with_stats(
    items: List[Any],
    process_func: Callable[..., Awaitable[T]],
    max_workers: int,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Process items in a batch with detailed performance statistics."""
    # Track stats including success/failure rates and processing times
    # Return both results and comprehensive stats
```

## Integration with Processor

The processor module has been updated to use batch processing instead of direct `asyncio.gather`:

```python
# Before:
tasks = [process_section(section, config, enable_bidirectional) 
         for section in sections]
section_results = await asyncio.gather(*tasks)

# After:
section_results = await batch_process_sections(
    sections=sections,
    config=config,
    process_function=process_section,
    enable_bidirectional=enable_bidirectional
)
```

This change maintains the same interface while adding scalability, rate limiting, and better error handling.

## Key Features

1. **Worker Pool Management**
   - Configurable worker count through QAGenerationConfig
   - Dynamic pool sizing based on available resources
   - Clean separation between worker management and business logic

2. **Rate Limiting**
   - Semaphore-based concurrency control
   - Consistent API rate limit adherence
   - Graceful handling of overloaded conditions

3. **Chunked Processing**
   - Support for very large datasets
   - Progress tracking for each chunk
   - Predictable memory usage for large inputs

4. **Performance Metrics**
   - Detailed timing for overall and per-item processing
   - Success/failure tracking
   - Throughput calculations for optimization

## Usage Examples

### Basic Usage

```python
from agent_tools.dualipa.qa.utils.batch_processing import batch_process_sections
from agent_tools.dualipa.qa.models.config import QAGenerationConfig

# Create configuration with worker pool size
config = QAGenerationConfig(worker_count=4)

# Process sections in batches
results = await batch_process_sections(
    sections=sections,
    config=config,
    process_function=process_function,
    enable_bidirectional=True
)
```

### With Performance Metrics

```python
from agent_tools.dualipa.qa.utils.batch_processing import batch_process_with_stats

# Process with detailed stats
results_with_stats = await batch_process_with_stats(
    items=items,
    process_func=process_function,
    max_workers=4
)

# Extract results and stats
results = [item["result"] for item in results_with_stats["results"]]
stats = results_with_stats["stats"]
print(f"Average processing time: {stats['avg_time']:.2f}s")
print(f"Throughput: {stats['throughput']:.2f} items/second")
```

## Testing Approach

1. **Functionality Testing**
   - `test_batch_process_sections_real`: Tests basic batch processing functionality
   - `test_batch_process_worker_pool`: Verifies worker pool concurrency limits
   - `test_semaphore_rate_limiting`: Tests semaphore-based concurrency control

2. **Performance Testing**
   - Measures timing across different worker pool sizes
   - Tracks concurrent executions by analyzing timing overlaps
   - Verifies resource usage with increasing load

## Architectural Benefits

1. **Scalability**
   - Handles increasing workloads efficiently
   - Allows for horizontal scaling by adjusting worker counts
   - Manages resources effectively with growing datasets

2. **Reliability**
   - Graceful error handling with per-section recovery
   - Proper rate limiting to prevent API bans
   - Detailed tracking to identify bottlenecks

3. **Maintainability**
   - Clean separation of concerns
   - Generic implementation usable across multiple contexts
   - Detailed logging for troubleshooting

## Next Steps

1. **Enhanced Monitoring**
   - Integrate with monitoring system for real-time tracking
   - Create dashboards for visualizing performance metrics
   - Set up alerts for performance degradation

2. **Dynamic Worker Adjustment**
   - Implement auto-scaling based on system load
   - Add backpressure handling for overloaded conditions
   - Create adaptive rate limiting based on API response times

3. **Optimizations**
   - Implement prioritization for critical sections
   - Add caching integration for improved throughput
   - Optimize worker allocation based on section complexity