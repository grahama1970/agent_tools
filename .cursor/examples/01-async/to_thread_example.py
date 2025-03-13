"""
Example of using asyncio.to_thread for CPU-bound operations.

This pattern should be used whenever you have a CPU-intensive task that would
block the event loop if run directly in an async function.

Documentation References:
- asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
- Concurrency and Parallelism: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-parallelism
- Related Rule: See `.cursor/rules/005-async-patterns.mdc` for comprehensive async patterns
"""

import asyncio
import time
from typing import Dict, Any, List
from loguru import logger


def cpu_intensive_calculation(data: List[int]) -> int:
    """
    A CPU-intensive synchronous function that would block the event loop.
    
    This is a simple example - in real code, this could be:
    - Complex data processing
    - Embedding generation
    - Image manipulation
    - Any computation that doesn't involve I/O
    """
    logger.debug(f"Starting CPU-intensive calculation on {len(data)} items")
    # Simulate CPU-intensive work
    result = 0
    for i in range(1000000):  # Arbitrary computation to simulate CPU work
        result += sum(data)
    logger.debug(f"Completed CPU-intensive calculation, result: {result}")
    return result


async def process_data_async(data: List[int]) -> Dict[str, Any]:
    """
    Process data asynchronously using to_thread for CPU-bound operations.
    
    This pattern allows the event loop to handle other tasks while
    the CPU-intensive work is being done in a separate thread.
    """
    try:
        logger.info(f"Processing data asynchronously: {data[:5]}...")
        
        # Use to_thread to run CPU-bound operation in a separate thread
        start_time = time.time()
        result = await asyncio.to_thread(cpu_intensive_calculation, data)
        elapsed = time.time() - start_time
        
        logger.info(f"Processed data in {elapsed:.2f} seconds")
        return {
            "result": result,
            "processing_time": elapsed,
            "items_processed": len(data)
        }
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


async def process_multiple_datasets(datasets: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Process multiple datasets concurrently.
    
    This demonstrates how to use to_thread with multiple concurrent tasks.
    """
    logger.info(f"Processing {len(datasets)} datasets concurrently")
    
    # Create tasks for each dataset
    tasks = [
        asyncio.create_task(process_data_async(dataset))
        for dataset in datasets
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Completed processing {len(datasets)} datasets")
    return results


# Example usage
async def main():
    # Create some sample datasets
    datasets = [
        [i for i in range(100)] for _ in range(5)
    ]
    
    # Process all datasets concurrently
    results = await process_multiple_datasets(datasets)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Dataset {i}: Processed {result['items_processed']} items "
              f"in {result['processing_time']:.2f} seconds")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 