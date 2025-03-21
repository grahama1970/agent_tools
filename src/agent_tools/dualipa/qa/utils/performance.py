"""Performance optimization utilities for QA generation.

This module provides tools and utilities for optimizing the performance
of the QA generation pipeline, including adaptive worker pools, chunking
strategies, and resource management.

Official documentation:
- multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- asyncio: https://docs.python.org/3/library/asyncio.html
- psutil: https://pypi.org/project/psutil/
- time: https://docs.python.org/3/library/time.html
- typing: https://docs.python.org/3/library/typing.html
- logging: https://docs.python.org/3/library/logging.html
- os: https://docs.python.org/3/library/os.html

Expected input/output:
- get_optimal_worker_count: Calculates optimal worker count based on system resources, returns int
- adaptive_chunk_size: Calculates optimal chunk size based on data characteristics, returns int
- profile_performance: Profiles function performance, returns ProfilerResult with metrics
- PerformanceTracker: Class for tracking performance metrics across operations
"""

import os
import time
import asyncio
import logging
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, TypeVar, Awaitable, cast

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
ProfilerResult = Dict[str, Any]


def get_optimal_worker_count(
    min_workers: int = 2,
    max_workers: int = 32,
    cpu_factor: float = 0.75,
    memory_factor: float = 0.5
) -> int:
    """Calculate the optimal worker count based on system resources.
    
    This function determines the optimal number of concurrent workers
    based on available CPU cores and memory, with configurable usage factors.
    
    Args:
        min_workers: Minimum number of workers (default: 2)
        max_workers: Maximum number of workers (default: 32)
        cpu_factor: Fraction of CPU cores to use (default: 0.75)
        memory_factor: Fraction of available memory to consider (default: 0.5)
        
    Returns:
        Optimal worker count based on system resources
    """
    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()
    cpu_based_workers = max(1, int(cpu_count * cpu_factor))
    
    # Calculate memory-based workers if psutil available
    if PSUTIL_AVAILABLE:
        try:
            # Get available memory in GB
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Assume each worker needs ~0.25GB (conservative estimate)
            worker_memory_gb = 0.25
            memory_based_workers = max(1, int((available_memory_gb * memory_factor) / worker_memory_gb))
            
            # Use the more conservative estimate
            optimal_workers = min(cpu_based_workers, memory_based_workers)
            
            logger.debug(f"System resources: {cpu_count} CPUs, {available_memory_gb:.2f}GB available memory")
            logger.debug(f"Resource-based worker calculation: CPU={cpu_based_workers}, Memory={memory_based_workers}")
        except Exception as e:
            logger.warning(f"Error calculating memory-based workers: {e}")
            optimal_workers = cpu_based_workers
    else:
        # Without psutil, just use CPU-based calculation
        optimal_workers = cpu_based_workers
    
    # Apply min/max bounds
    optimal_workers = max(min_workers, min(max_workers, optimal_workers))
    
    logger.info(f"Calculated optimal worker count: {optimal_workers}")
    return optimal_workers


def adaptive_chunk_size(
    total_items: int, 
    worker_count: int,
    min_chunk_size: int = 5,
    max_chunk_size: int = 50,
    target_chunks_per_worker: float = 2.0
) -> int:
    """Calculate adaptive chunk size based on workload and workers.
    
    This function determines the optimal chunk size for batch processing
    based on the total number of items, available workers, and target
    chunks per worker ratio.
    
    Args:
        total_items: Total number of items to process
        worker_count: Number of workers available
        min_chunk_size: Minimum chunk size (default: 5)
        max_chunk_size: Maximum chunk size (default: 50)
        target_chunks_per_worker: Target chunks per worker (default: 2.0)
        
    Returns:
        Optimal chunk size for the given workload
    """
    if total_items <= 0 or worker_count <= 0:
        return min_chunk_size
    
    # Calculate how many chunks we want in total
    target_chunk_count = worker_count * target_chunks_per_worker
    
    # Calculate chunk size to achieve the target chunk count
    calculated_chunk_size = int(total_items / target_chunk_count)
    
    # Apply min/max bounds
    optimal_chunk_size = max(min_chunk_size, min(max_chunk_size, calculated_chunk_size))
    
    logger.debug(f"Adaptive chunk size for {total_items} items with {worker_count} workers: {optimal_chunk_size}")
    return optimal_chunk_size


async def profile_performance(
    func: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> ProfilerResult:
    """Profile the performance of an async function.
    
    This utility measures execution time, memory usage, and provides
    performance metrics for async functions.
    
    Args:
        func: Async function to profile
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Dictionary with performance metrics
    """
    start_time = time.time()
    start_memory = None
    
    # Get initial memory usage if psutil available
    if PSUTIL_AVAILABLE:
        try:
            start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        except Exception as e:
            logger.warning(f"Error getting initial memory usage: {e}")
    
    # Execute the function
    try:
        result = await func(*args, **kwargs)
        success = True
    except Exception as e:
        result = e
        success = False
    
    # Calculate performance metrics
    elapsed_time = time.time() - start_time
    
    # Get final memory usage if psutil available
    end_memory = None
    memory_used = None
    
    if PSUTIL_AVAILABLE and start_memory is not None:
        try:
            end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            memory_used = end_memory - start_memory
        except Exception as e:
            logger.warning(f"Error getting final memory usage: {e}")
    
    # Prepare profiler result
    profiler_result = {
        "elapsed_time": elapsed_time,
        "success": success,
        "function": func.__name__,
        "timestamp": time.time()
    }
    
    # Add memory metrics if available
    if start_memory is not None and end_memory is not None:
        profiler_result.update({
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "memory_used_mb": memory_used
        })
    
    # Provide optimization recommendations
    if elapsed_time > 10.0:
        profiler_result["recommendations"] = ["Consider parallelization or chunking"]
    
    if memory_used and memory_used > 100:  # More than 100MB used
        profiler_result["recommendations"] = profiler_result.get("recommendations", [])
        profiler_result["recommendations"].append("Consider memory optimization or streaming")
    
    return profiler_result


class PerformanceTracker:
    """Track and analyze performance metrics over time.
    
    This class collects performance metrics for functions and provides
    analysis to identify performance trends and bottlenecks.
    
    Example:
        tracker = PerformanceTracker()
        tracker.start("batch_processing")
        # ... do batch processing ...
        tracker.end("batch_processing")
        metrics = tracker.get_metrics()
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics = {}
        self.start_times = {}
        self.call_counts = {}
    
    def start(self, operation_name: str) -> None:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
        """
        self.start_times[operation_name] = time.time()
    
    def end(self, operation_name: str) -> float:
        """End timing an operation and record the metric.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Elapsed time in seconds
        """
        if operation_name not in self.start_times:
            logger.warning(f"No start time recorded for {operation_name}")
            return 0.0
        
        elapsed = time.time() - self.start_times[operation_name]
        
        # Update metrics
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "total_time": 0.0,
                "min_time": elapsed,
                "max_time": elapsed,
                "times": []
            }
            self.call_counts[operation_name] = 0
        
        metrics = self.metrics[operation_name]
        metrics["total_time"] += elapsed
        metrics["min_time"] = min(metrics["min_time"], elapsed)
        metrics["max_time"] = max(metrics["max_time"], elapsed)
        metrics["times"].append(elapsed)
        
        self.call_counts[operation_name] += 1
        
        return elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected performance metrics.
        
        Returns:
            Dictionary with performance metrics for all tracked operations
        """
        results = {}
        
        for operation, metrics in self.metrics.items():
            call_count = self.call_counts.get(operation, 0)
            
            if call_count > 0:
                avg_time = metrics["total_time"] / call_count
            else:
                avg_time = 0.0
            
            results[operation] = {
                "total_time": metrics["total_time"],
                "avg_time": avg_time,
                "min_time": metrics["min_time"],
                "max_time": metrics["max_time"],
                "call_count": call_count
            }
            
            # Calculate percentiles if we have enough data
            times = metrics.get("times", [])
            if len(times) >= 3:
                times.sort()
                results[operation]["percentile_50"] = times[len(times) // 2]
                results[operation]["percentile_90"] = times[int(len(times) * 0.9)]
                results[operation]["percentile_99"] = times[int(len(times) * 0.99)]
        
        return results
    
    def reset(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {}
        self.start_times = {}
        self.call_counts = {}


# Global performance tracker instance
performance_tracker = PerformanceTracker()