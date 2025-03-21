"""Performance tests for QA generation pipeline.

This module tests the performance and scalability of the QA generation pipeline.
It measures processing time for various workloads and verifies the pipeline
meets performance requirements under load.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- time: https://docs.python.org/3/library/time.html
- json: https://docs.python.org/3/library/json.html
- asyncio: https://docs.python.org/3/library/asyncio.html
- dotenv: https://pypi.org/project/python-dotenv/
- litellm: https://litellm.ai/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html
- typing: https://docs.python.org/3/library/typing.html

Expected input/output:
- test_pipeline_performance_real: Takes sample_extraction_json fixture, returns performance metrics dictionary
  * Input: Sample extraction JSON with real content
  * Output: Processing completes successfully with real LLM calls, output file created
  * Verification: QA pairs are generated with expected structure and content

- test_batch_processing_small: Takes no parameters, returns performance comparison dictionary
  * Input: Small set of test items to process
  * Output: Processing results with single and multi-worker configurations
  * Verification: Multi-worker processing demonstrates performance improvement

- test_adaptive_resource_optimization: Takes no parameters, returns optimization metrics
  * Input: Tests different combinations of item counts and worker counts
  * Output: Optimal worker count and chunk sizes based on system resources
  * Verification: Values are within expected ranges and follow proper scaling

- test_cache_performance: Takes no parameters, returns cache performance metrics
  * Input: Test configuration processed with and without caching
  * Output: Performance comparison between cached and uncached calls
  * Verification: Cache provides significant performance improvement for repeated calls
"""

import json
import time
import asyncio
import pytest
import os
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Make sure OPENAI_API_KEY is available
if "OPENAI_API_KEY" not in os.environ:
    pytest.skip("OPENAI_API_KEY environment variable not set", allow_module_level=True)

# Mark all tests as async
pytestmark = pytest.mark.asyncio


async def test_pipeline_performance_real(sample_extraction_json, tmp_path):
    """Test the performance of the full pipeline with real data.
    
    This establishes baseline performance metrics for optimization.
    It uses real API calls to test actual integration and performance.
    
    Success criteria:
    - Processing completes successfully with real LLM calls
    - Output file is created with expected structure
    - QA pairs are generated from real content
    """
    # Import the necessary functions and classes
    from agent_tools.dualipa.qa.processor import process_extraction_json
    from agent_tools.dualipa.qa.models.config import QAGenerationConfig
    
    # Performance test configuration - optimize for speed
    config = QAGenerationConfig(
        worker_count=2,  # Small worker count for test
        max_concurrent_requests=2,
        temperature_range=[0.7],  # Single temperature for speed
        model="gpt-4o-mini",  # Use smaller/faster model for testing
        max_qa_pairs_per_section=1,  # Minimum pairs for testing
        bidirectional_ratio=0.0  # Disable bidirectional for test speed
    )
    
    # Set up output path
    output_file = tmp_path / "qa_output.json"
    
    # Take a small subset of sections to test (for speed)
    # Add a limit to avoid long test times
    if len(sample_extraction_json["sections"]) > 3:
        # Create a limited copy to avoid modifying the fixture
        limited_data = sample_extraction_json.copy()
        limited_data["sections"] = sample_extraction_json["sections"][:3]
        test_data = limited_data
    else:
        test_data = sample_extraction_json
    
    # Measure execution time
    start_time = time.time()
    
    # Process the extraction JSON with real LLM calls
    result = await process_extraction_json(
        input_data=test_data,
        output_file=output_file,
        config=config,
        enable_monitoring=True
    )
    
    elapsed_time = time.time() - start_time
    
    # Verify the output structure
    assert result is not None
    assert output_file.exists()
    assert len(result.qa_pairs) > 0, "Should generate at least one QA pair"
    
    # Verify QA pairs have expected structure 
    if result.qa_pairs:
        qa_pair = result.qa_pairs[0]
        assert qa_pair.question, "Question should not be empty"
        assert qa_pair.answer, "Answer should not be empty"
        assert len(qa_pair.reasoning) >= 15, "Reasoning should meet minimum length"
    
    # Log performance metrics
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(test_data['sections'])} sections")
    print(f"Generated {len(result.qa_pairs)} QA pairs")
    
    # Return performance metrics for reference
    return {
        "elapsed_time": elapsed_time,
        "sections_processed": len(test_data["sections"]),
        "pairs_generated": len(result.qa_pairs)
    }


async def test_batch_processing_small():
    """Test batch processing with a small set of items.
    
    This test focuses specifically on the batch processing functionality
    with a smaller test to keep execution time reasonable.
    
    Success criteria:
    - Batch processing completes successfully
    - Results are properly processed for all items
    """
    from agent_tools.dualipa.qa.utils.batch_processing import batch_process_with_stats
    
    # Create a small sample of test items
    test_items = [
        {"id": i, "data": f"Test content for item {i}"} 
        for i in range(5)
    ]
    
    # Simple processing function that simulates work
    async def process_item(item):
        # Simulate some work
        await asyncio.sleep(0.1)
        return {"processed_id": item["id"], "result": f"Processed {item['data']}"}
    
    # Measure execution time with different worker counts
    results_single = await batch_process_with_stats(
        items=test_items,
        process_func=process_item,
        max_workers=1
    )
    
    results_multi = await batch_process_with_stats(
        items=test_items,
        process_func=process_item,
        max_workers=4
    )
    
    # Verify correct processing
    assert len(results_single["results"]) == len(test_items)
    assert len(results_multi["results"]) == len(test_items)
    
    # Verify performance statistics are tracked
    assert "stats" in results_single
    assert "stats" in results_multi
    assert "total_time" in results_single["stats"]
    assert "total_time" in results_multi["stats"]
    
    # Log performance comparison
    single_time = results_single["stats"]["total_time"]
    multi_time = results_multi["stats"]["total_time"]
    print(f"\nSingle worker processing time: {single_time:.2f}s")
    print(f"Multi worker processing time: {multi_time:.2f}s")
    print(f"Speed improvement: {single_time/multi_time:.2f}x")
    
    # Basic verification of parallel efficiency
    # Multi-worker should be faster than single worker for this workload
    assert multi_time < single_time, "Multi-worker processing should be faster"
    
    return {
        "single_worker_time": single_time,
        "multi_worker_time": multi_time,
        "items_processed": len(test_items),
        "speedup_factor": single_time/multi_time if multi_time > 0 else 0
    }


async def test_adaptive_resource_optimization():
    """Test the adaptive resource optimization functionality.
    
    This test verifies that our optimization utilities can correctly
    determine appropriate worker counts and chunk sizes based on
    system resources.
    
    Success criteria:
    - Resource detection functions correctly
    - Adaptive worker count is reasonable
    - Adaptive chunk sizing provides appropriate values
    """
    # Import the performance utilities
    from agent_tools.dualipa.qa.utils.performance import (
        get_optimal_worker_count,
        adaptive_chunk_size,
        profile_performance
    )
    
    # Test optimal worker count calculation
    worker_count = get_optimal_worker_count(
        min_workers=2,
        max_workers=16,
        cpu_factor=0.75,
        memory_factor=0.5
    )
    
    # Should return a reasonable number within bounds
    assert worker_count >= 2, "Worker count should respect minimum bound"
    assert worker_count <= 16, "Worker count should respect maximum bound"
    
    # Test adaptive chunk size calculation
    # Test various combinations of items and workers
    test_cases = [
        (100, 4),   # 100 items, 4 workers
        (10, 2),    # 10 items, 2 workers
        (1000, 8)   # 1000 items, 8 workers
    ]
    
    for total_items, worker_count in test_cases:
        chunk_size = adaptive_chunk_size(
            total_items=total_items,
            worker_count=worker_count,
            min_chunk_size=5,
            max_chunk_size=50
        )
        
        # Verify chunk size is within bounds
        assert chunk_size >= 5, "Chunk size should respect minimum bound"
        assert chunk_size <= 50, "Chunk size should respect maximum bound"
        
        # Verify reasonable distribution
        expected_chunks = total_items / chunk_size
        assert expected_chunks >= worker_count, f"Should have at least as many chunks ({expected_chunks}) as workers ({worker_count})"
    
    # Test performance profiling with a simple async function
    async def test_function():
        await asyncio.sleep(0.1)
        return "test result"
    
    profile_result = await profile_performance(test_function)
    
    # Verify profiling result structure
    assert "elapsed_time" in profile_result
    assert profile_result["elapsed_time"] >= 0.1
    assert "function" in profile_result
    assert profile_result["function"] == "test_function"
    
    # Log results
    print(f"\nOptimal worker count: {worker_count}")
    print(f"Sample chunk size (100 items, 4 workers): {adaptive_chunk_size(100, 4)}")
    print(f"Performance profiling elapsed time: {profile_result['elapsed_time']:.2f}s")
    
    # Return results for reference
    return {
        "optimal_worker_count": worker_count,
        "chunk_sizes": {f"{items}_{workers}": adaptive_chunk_size(items, workers) 
                        for items, workers in test_cases},
        "profiling_time": profile_result["elapsed_time"]
    }


async def test_cache_performance():
    """Test cache performance with real LLM configurations.
    
    This test verifies that the caching system correctly improves
    performance by reusing previously generated results.
    
    Success criteria:
    - Cache initialization succeeds
    - Cache provides performance benefits for repeated requests
    - Cache hit/miss tracking works correctly
    """
    from agent_tools.dualipa.qa.utils.cache import (
        initialize_cache, add_to_cache, get_from_cache, clear_cache, get_cache_stats
    )
    import litellm
    
    # Initialize cache with clean state
    clear_cache()
    cache = initialize_cache()
    
    # Create a test LLM configuration (similar to actual usage)
    test_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
    
    # Function to simulate an LLM call with timing
    async def call_llm_with_cache(config, use_cache=True):
        start_time = time.time()
        
        # Try to get from cache first if cache is enabled
        if use_cache:
            result = get_from_cache(config)
            if result is not None:
                return result, time.time() - start_time
        
        # Simulate actual LLM call
        try:
            # For testing, we'll simulate a response without actually calling the API
            # In a real scenario, this would be: result = litellm.completion(config)
            await asyncio.sleep(0.5)  # Simulate API latency
            result = {
                "choices": [{
                    "message": {
                        "content": "Paris is the capital of France."
                    }
                }]
            }
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise
        
        # Add to cache if cache is enabled
        if use_cache:
            add_to_cache(config, result)
        
        return result, time.time() - start_time
    
    # First call (cold, should be a cache miss)
    result1, time1 = await call_llm_with_cache(test_config)
    
    # Second call (warm, should be a cache hit)
    result2, time2 = await call_llm_with_cache(test_config)
    
    # Third call with cache disabled (should not use cache)
    result3, time3 = await call_llm_with_cache(test_config, use_cache=False)
    
    # Verify cache is working
    assert result1 == result2, "Cached result should match original"
    assert result1 == result3, "All results should be consistent"
    
    # Verify performance improvement
    assert time2 < time1, "Cached call should be faster than original call"
    assert time2 < time3, "Cached call should be faster than non-cached call"
    
    # Check cache statistics
    stats = get_cache_stats()
    
    # Log performance information
    print(f"\nFirst call (cache miss): {time1:.3f}s")
    print(f"Second call (cache hit): {time2:.3f}s")
    print(f"Third call (no cache): {time3:.3f}s")
    print(f"Cache speedup factor: {time1/time2:.1f}x")
    print(f"Cache stats: {stats}")
    
    # Basic verification of stats
    assert "hits" in stats
    assert "misses" in stats
    assert stats["hits"] >= 1, "Cache hits should be recorded"
    assert stats["misses"] >= 1, "Cache misses should be recorded"
    
    # Return performance metrics
    return {
        "uncached_time": time1,
        "cached_time": time2,
        "no_cache_time": time3,
        "speedup_factor": time1/time2 if time2 > 0 else 0,
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"]
    }