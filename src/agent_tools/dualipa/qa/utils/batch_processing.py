"""Batch processing utilities for QA generation.

This module implements batch processing with scalable worker pools and
rate limiting using asyncio semaphores. It efficiently processes
multiple sections in parallel while respecting API rate limits.

Official documentation:
- asyncio: https://docs.python.org/3/library/asyncio.html
- logging: https://docs.python.org/3/library/logging.html
- time: https://docs.python.org/3/library/time.html
- typing: https://docs.python.org/3/library/typing.html

Expected input/output:
- process_with_semaphore: Takes semaphore, function, and args, returns function result with rate limiting
- batch_process_sections: Takes sections, config, and process_function, returns processed results for all sections
- batch_process_with_stats: Takes items, process_func, and max_workers, returns results with detailed performance statistics
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Awaitable, cast

from agent_tools.dualipa.qa.models.config import QAGenerationConfig

# Import performance optimization utilities (use try/except to handle potential import error)
try:
    from agent_tools.dualipa.qa.utils.performance import (
        get_optimal_worker_count,
        adaptive_chunk_size,
        performance_tracker
    )
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Generic type for return values
T = TypeVar('T')


async def process_with_semaphore(
    semaphore: asyncio.Semaphore,
    func: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """Process a function with semaphore-based rate limiting.
    
    This utility function wraps any async function with a semaphore to
    ensure rate limiting across multiple concurrent executions.
    
    Args:
        semaphore: The semaphore to use for rate limiting
        func: The async function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    async with semaphore:
        start_time = time.time()
        logger.debug(f"Starting process with semaphore (args: {args})")
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed process in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in semaphore-protected process after {elapsed:.2f}s: {str(e)}")
            # Re-raise to allow caller to handle
            raise


async def batch_process_sections(
    sections: List[Dict[str, Any]],
    config: QAGenerationConfig,
    process_function: Callable[..., Awaitable[T]],
    enable_bidirectional: bool = True,
    chunk_size: Optional[int] = None
) -> List[T]:
    """Process multiple sections in batches with worker pools.
    
    This function handles batch processing of sections with:
    1. Configurable worker pool size
    2. API rate limiting using semaphores
    3. Proper concurrency control
    4. Chunked processing for very large datasets
    5. Progress tracking and logging
    6. Adaptive performance optimization
    
    Args:
        sections: List of sections to process
        config: QA generation configuration
        process_function: Function to process each section
        enable_bidirectional: Whether to enable bidirectional processing
        chunk_size: Optional size of chunks for very large datasets
        
    Returns:
        List of processing results, one per section
    """
    if not sections:
        logger.warning("No sections provided for batch processing")
        return []
    
    # Start performance tracking if available
    if PERFORMANCE_UTILS_AVAILABLE:
        performance_tracker.start("batch_process_sections")
    
    # Determine optimal worker count
    if PERFORMANCE_UTILS_AVAILABLE:
        # If worker_count is explicitly set in config, use that
        if hasattr(config, 'worker_count'):
            worker_count = config.worker_count
        else:
            # Otherwise calculate optimal worker count based on system resources
            worker_count = get_optimal_worker_count(
                min_workers=2,
                max_workers=config.max_concurrent_requests,
                cpu_factor=0.75,
                memory_factor=0.5
            )
    else:
        # Use the original fallback logic
        worker_count = getattr(config, 'worker_count', config.max_concurrent_requests)
    
    # Create semaphore for rate limiting
    # This ensures we don't exceed API rate limits
    semaphore = asyncio.Semaphore(worker_count)
    
    # Determine optimal chunk size if needed
    if chunk_size is None and PERFORMANCE_UTILS_AVAILABLE and len(sections) > worker_count * 3:
        # Calculate adaptive chunk size for large datasets
        chunk_size = adaptive_chunk_size(
            total_items=len(sections),
            worker_count=worker_count,
            min_chunk_size=5,
            max_chunk_size=50,
            target_chunks_per_worker=2.0
        )
        logger.info(f"Using adaptive chunk size: {chunk_size}")
    elif chunk_size is None and len(sections) > 50:
        # Default chunking for large datasets without performance utils
        chunk_size = 25
    
    logger.info(f"Starting batch processing of {len(sections)} sections "
                f"with {worker_count} workers")
    
    start_time = time.time()
    
    # Track section types for performance analysis
    section_type_counts = {}
    for section in sections:
        section_type = section.get("type", "unknown")
        section_type_counts[section_type] = section_type_counts.get(section_type, 0) + 1
    
    logger.debug(f"Section type distribution: {section_type_counts}")
    
    # Process sections in chunks if appropriate
    if chunk_size and chunk_size > 0 and len(sections) > chunk_size:
        all_results = []
        chunks = [sections[i:i+chunk_size] for i in range(0, len(sections), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            # Start chunk performance tracking
            if PERFORMANCE_UTILS_AVAILABLE:
                performance_tracker.start(f"process_chunk_{i}")
            
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} sections")
            
            # Sort chunk by section type for better locality and cache utilization
            chunk.sort(key=lambda s: s.get("type", "unknown"))
            
            # Process each chunk with worker pool
            tasks = [
                process_with_semaphore(
                    semaphore,
                    process_function,
                    section,
                    config,
                    enable_bidirectional
                )
                for section in chunk
            ]
            
            # Execute all tasks with controlled concurrency
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions by returning empty results
            processed_results = []
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing section: {str(result)}")
                    processed_results.append([])  # Empty result on error
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Log progress
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1) / len(chunk)
            logger.info(f"Completed chunk {i+1}/{len(chunks)} in {elapsed:.2f}s "
                        f"(avg {avg_time:.2f}s per section)")
            
            # End chunk performance tracking
            if PERFORMANCE_UTILS_AVAILABLE:
                chunk_time = performance_tracker.end(f"process_chunk_{i}")
                logger.debug(f"Chunk {i+1} processing time: {chunk_time:.2f}s")
        
        results = all_results
    else:
        # For smaller datasets, process all sections in a single batch
        # First sort by section type for better cache locality
        if len(sections) > 1:
            sorted_sections = sorted(sections, key=lambda s: s.get("type", "unknown"))
        else:
            sorted_sections = sections
        
        tasks = [
            process_with_semaphore(
                semaphore,
                process_function,
                section,
                config,
                enable_bidirectional
            )
            for section in sorted_sections
        ]
        
        # Execute all tasks with controlled concurrency
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions by returning empty results
        results = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error(f"Error processing section: {str(result)}")
                results.append([])  # Empty result on error
            else:
                results.append(result)
    
    # Calculate final performance metrics
    elapsed = time.time() - start_time
    avg_time = elapsed / len(sections)
    sections_per_second = len(sections) / elapsed if elapsed > 0 else 0
    
    logger.info(f"Completed batch processing of {len(sections)} sections "
                f"in {elapsed:.2f}s (avg {avg_time:.2f}s per section, "
                f"{sections_per_second:.2f} sections/second)")
    
    # End performance tracking
    if PERFORMANCE_UTILS_AVAILABLE:
        total_time = performance_tracker.end("batch_process_sections")
        logger.debug(f"Total batch processing time: {total_time:.2f}s")
    
    return results


async def batch_process_with_stats(
    items: List[Any],
    process_func: Callable[..., Awaitable[T]],
    max_workers: int,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Process items in a batch with detailed performance statistics.
    
    This function provides advanced batch processing with performance tracking:
    1. Measures individual and total processing times
    2. Tracks success/failure rates
    3. Calculates throughput and avg/min/max processing times
    4. Handles worker pool management
    
    Args:
        items: List of items to process
        process_func: Function to process each item 
        max_workers: Maximum number of concurrent workers
        *args, **kwargs: Additional arguments for the process function
        
    Returns:
        Dictionary with results and performance statistics
    """
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_workers)
    stats = {
        "total_items": len(items),
        "successful": 0,
        "failed": 0,
        "times": [],
        "exceptions": []
    }
    
    # Track timing and status for each item
    async def process_with_stats(item, index):
        item_start = time.time()
        try:
            async with semaphore:
                result = await process_func(item, *args, **kwargs)
                stats["successful"] += 1
                return {
                    "result": result,
                    "success": True,
                    "index": index,
                    "time": time.time() - item_start
                }
        except Exception as e:
            logger.error(f"Error processing item {index}: {str(e)}")
            stats["failed"] += 1
            stats["exceptions"].append(str(e))
            return {
                "result": None,
                "success": False,
                "index": index,
                "time": time.time() - item_start,
                "error": str(e)
            }
    
    # Create and execute tasks
    tasks = [process_with_stats(item, i) for i, item in enumerate(items)]
    results = await asyncio.gather(*tasks)
    
    # Collect timing information
    total_time = time.time() - start_time
    process_times = [r["time"] for r in results]
    
    # Calculate statistics
    stats.update({
        "total_time": total_time,
        "avg_time": sum(process_times) / len(process_times) if process_times else 0,
        "min_time": min(process_times) if process_times else 0,
        "max_time": max(process_times) if process_times else 0,
        "throughput": len(items) / total_time if total_time > 0 else 0
    })
    
    return {
        "results": results,
        "stats": stats
    }