#!/usr/bin/env python3
"""Performance verification for CI.

This script verifies performance against established baselines, checking for
regressions and performance gates to ensure the QA generation module
maintains expected performance characteristics.

Official documentation:
- argparse: https://docs.python.org/3/library/argparse.html
- json: https://docs.python.org/3/library/json.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- sys: https://docs.python.org/3/library/sys.html
- typing: https://docs.python.org/3/library/typing.html

Expected input/output:
- verify_performance_metrics: Takes metrics and baselines, returns success_flag and error_message
  * Input: Current metrics and baseline metrics
  * Output: Boolean success status and error message if applicable
  * Verification: Metrics must be within acceptable threshold of baselines

- load_baseline_metrics: Takes baseline file path, returns baseline metrics dict
  * Input: Path to baseline metrics JSON file
  * Output: Dictionary of baseline metrics
  * Verification: Baseline file must exist and contain valid data

- generate_baseline_metrics: Takes no parameters, returns metrics dictionary
  * Input: None
  * Output: Dictionary of performance metrics from test execution
  * Verification: Tests run successfully and generate valid metrics
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define performance thresholds (as percentages)
PERFORMANCE_THRESHOLDS = {
    "batch_processing_time": 20,  # 20% slower is acceptable
    "worker_scaling_efficiency": 15,  # 15% degradation in worker scaling is acceptable
    "adaptive_chunk_calculation": 5,  # 5% deviation in chunk size calculation
    "cache_hit_rate": 10,  # 10% degradation in cache hit rate
    "memory_usage": 25,  # 25% increase in memory usage is acceptable
}

# Path to baseline metrics file (relative to module directory)
BASELINE_FILE = Path(__file__).parent / "baseline_metrics.json"


def verify_performance_metrics(
    metrics: Dict[str, Any], 
    baselines: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Verify performance metrics against baselines.
    
    This function compares current performance metrics with established baselines,
    ensuring they remain within acceptable thresholds to catch performance regressions.
    
    Args:
        metrics: Current performance metrics
        baselines: Baseline performance metrics
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    if not metrics or not baselines:
        return False, "Missing metrics or baselines"
    
    failures = []
    
    # Check batch processing metrics
    if "batch_processing" in metrics and "batch_processing" in baselines:
        current = metrics["batch_processing"].get("multi_worker_time", 0)
        baseline = baselines["batch_processing"].get("multi_worker_time", 0)
        
        if current > 0 and baseline > 0:
            percent_change = (current - baseline) / baseline * 100
            threshold = PERFORMANCE_THRESHOLDS["batch_processing_time"]
            
            if percent_change > threshold:
                failures.append(
                    f"Batch processing regression: {percent_change:.1f}% slower than baseline "
                    f"(threshold: {threshold}%)"
                )
    
    # Check worker scaling efficiency
    if "worker_scaling" in metrics and "worker_scaling" in baselines:
        current = metrics["worker_scaling"].get("speedup_factor", 0)
        baseline = baselines["worker_scaling"].get("speedup_factor", 0)
        
        if current > 0 and baseline > 0:
            percent_change = (baseline - current) / baseline * 100
            threshold = PERFORMANCE_THRESHOLDS["worker_scaling_efficiency"]
            
            if percent_change > threshold:
                failures.append(
                    f"Worker scaling regression: {percent_change:.1f}% less efficient than baseline "
                    f"(threshold: {threshold}%)"
                )
    
    # Check adaptive chunk calculation
    if "adaptive_chunk" in metrics and "adaptive_chunk" in baselines:
        for key in baselines["adaptive_chunk"]:
            if key in metrics["adaptive_chunk"]:
                current = metrics["adaptive_chunk"][key]
                baseline = baselines["adaptive_chunk"][key]
                
                if current > 0 and baseline > 0:
                    percent_change = abs(current - baseline) / baseline * 100
                    threshold = PERFORMANCE_THRESHOLDS["adaptive_chunk_calculation"]
                    
                    if percent_change > threshold:
                        failures.append(
                            f"Adaptive chunk size regression for {key}: {percent_change:.1f}% deviation "
                            f"(threshold: {threshold}%)"
                        )
    
    # Check cache performance
    if "cache" in metrics and "cache" in baselines:
        current = metrics["cache"].get("speedup_factor", 0)
        baseline = baselines["cache"].get("speedup_factor", 0)
        
        if current > 0 and baseline > 0:
            percent_change = (baseline - current) / baseline * 100
            threshold = PERFORMANCE_THRESHOLDS["cache_hit_rate"]
            
            if percent_change > threshold:
                failures.append(
                    f"Cache performance regression: {percent_change:.1f}% degradation "
                    f"(threshold: {threshold}%)"
                )
    
    # Check memory usage
    if "memory" in metrics and "memory" in baselines:
        current = metrics["memory"].get("peak_usage_mb", 0)
        baseline = baselines["memory"].get("peak_usage_mb", 0)
        
        if current > 0 and baseline > 0:
            percent_change = (current - baseline) / baseline * 100
            threshold = PERFORMANCE_THRESHOLDS["memory_usage"]
            
            if percent_change > threshold:
                failures.append(
                    f"Memory usage regression: {percent_change:.1f}% increase "
                    f"(threshold: {threshold}%)"
                )
    
    # Return success if no failures, otherwise return error message
    if not failures:
        return True, None
    else:
        return False, "\n".join(failures)


def load_baseline_metrics(baseline_file: Union[str, Path] = BASELINE_FILE) -> Dict[str, Any]:
    """Load baseline performance metrics from file.
    
    Args:
        baseline_file: Path to baseline metrics JSON file
        
    Returns:
        Dictionary of baseline metrics
    """
    try:
        with open(baseline_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Baseline file not found: {baseline_file}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in baseline file: {baseline_file}")
        return {}


def generate_baseline_metrics() -> Dict[str, Any]:
    """Generate baseline performance metrics.
    
    This function runs the performance tests and collects metrics
    to establish a baseline for future verification.
    
    Returns:
        Dictionary of performance metrics
    """
    import pytest
    import tempfile
    import time
    
    # Use pytest to run the performance tests
    test_results = {}
    
    # Run batch processing test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run batch processing test and capture output
        from agent_tools.dualipa.qa.utils.batch_processing import batch_process_with_stats
        import asyncio
        
        async def test_batch_processing():
            test_items = [{"id": i, "data": f"Test content {i}"} for i in range(10)]
            
            async def process_item(item):
                await asyncio.sleep(0.05)
                return {"processed": item["id"]}
            
            single_result = await batch_process_with_stats(
                items=test_items, 
                process_func=process_item, 
                max_workers=1
            )
            
            multi_result = await batch_process_with_stats(
                items=test_items, 
                process_func=process_item, 
                max_workers=4
            )
            
            return {
                "single_worker_time": single_result["stats"]["total_time"],
                "multi_worker_time": multi_result["stats"]["total_time"],
                "speedup_factor": single_result["stats"]["total_time"] / multi_result["stats"]["total_time"] 
                                  if multi_result["stats"]["total_time"] > 0 else 0
            }
        
        # Run the test
        results = asyncio.run(test_batch_processing())
        test_results["batch_processing"] = results
        test_results["worker_scaling"] = {
            "speedup_factor": results["speedup_factor"]
        }
    
    # Run adaptive chunk size test
    try:
        from agent_tools.dualipa.qa.utils.performance import adaptive_chunk_size
        
        test_cases = [
            (100, 4),   # 100 items, 4 workers
            (10, 2),    # 10 items, 2 workers
            (1000, 8)   # 1000 items, 8 workers
        ]
        
        chunk_sizes = {}
        for items, workers in test_cases:
            key = f"{items}_{workers}"
            chunk_sizes[key] = adaptive_chunk_size(items, workers)
        
        test_results["adaptive_chunk"] = chunk_sizes
    except ImportError:
        logger.warning("Could not import adaptive_chunk_size")
    
    # Run cache performance test
    try:
        from agent_tools.dualipa.qa.utils.cache import initialize_cache, add_to_cache, get_from_cache, clear_cache
        
        # Clear existing cache
        clear_cache()
        cache = initialize_cache()
        
        # Test basic caching
        test_key = {"model": "test-model", "messages": [{"role": "user", "content": "test"}]}
        test_value = {"choices": [{"message": {"content": "test response"}}]}
        
        # Measure time without cache
        start_time = time.time()
        time.sleep(0.1)  # Simulate API call
        no_cache_time = time.time() - start_time
        
        # Add to cache
        add_to_cache(test_key, test_value)
        
        # Measure time with cache
        start_time = time.time()
        cached_result = get_from_cache(test_key)
        cache_time = time.time() - start_time
        
        test_results["cache"] = {
            "uncached_time": no_cache_time,
            "cached_time": cache_time,
            "speedup_factor": no_cache_time / cache_time if cache_time > 0 else 0
        }
    except ImportError:
        logger.warning("Could not import cache utilities")
    
    # Add memory usage metrics
    try:
        import psutil
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        test_results["memory"] = {
            "peak_usage_mb": memory_info.rss / (1024 * 1024)
        }
    except ImportError:
        logger.warning("Could not import psutil for memory metrics")
    
    return test_results


def save_baseline_metrics(metrics: Dict[str, Any], output_file: Union[str, Path] = BASELINE_FILE):
    """Save performance metrics as baseline.
    
    Args:
        metrics: Performance metrics to save
        output_file: Path to save baseline metrics
    """
    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to file
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved baseline metrics to {output_path}")


def main():
    """Main entry point for performance verification."""
    parser = argparse.ArgumentParser(
        description="Verify performance against baselines"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate baseline metrics instead of verifying"
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=str(BASELINE_FILE),
        help=f"Path to baseline metrics file (default: {BASELINE_FILE})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.generate_baseline:
        # Generate and save baseline metrics
        logger.info("Generating baseline performance metrics...")
        baseline_metrics = generate_baseline_metrics()
        save_baseline_metrics(baseline_metrics, args.baseline_file)
        logger.info("Baseline metrics generated successfully")
        return 0
    else:
        # Load baseline metrics
        logger.info(f"Loading baseline metrics from {args.baseline_file}")
        baseline_metrics = load_baseline_metrics(args.baseline_file)
        
        if not baseline_metrics:
            logger.error("No baseline metrics available. Run with --generate-baseline first.")
            return 1
        
        # Generate current metrics
        logger.info("Generating current performance metrics...")
        current_metrics = generate_baseline_metrics()
        
        # Verify metrics against baseline
        logger.info("Verifying performance metrics against baseline...")
        success, error_message = verify_performance_metrics(current_metrics, baseline_metrics)
        
        if success:
            logger.info("✅ Performance verification passed: All metrics within acceptable thresholds")
            return 0
        else:
            logger.error(f"❌ Performance verification failed:\n{error_message}")
            return 1


if __name__ == "__main__":
    sys.exit(main())