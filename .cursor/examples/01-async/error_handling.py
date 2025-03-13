"""
Example of proper error handling in async functions.

This demonstrates best practices for handling errors in async code, including:
- Using try/except blocks
- Implementing timeouts
- Handling cancellation
- Proper cleanup in finally blocks

Documentation References:
- Error Handling: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.set_exception
- Timeouts: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
- Cancellation: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel
- Related Rule: See `.cursor/rules/005-async-patterns.mdc` section on "Error Handling"
"""

import asyncio
import random
from typing import Dict, Any, Optional
from loguru import logger


class ServiceUnavailableError(Exception):
    """Custom exception for when a service is unavailable."""
    pass


async def fetch_data_from_service(service_id: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Fetch data from a service with proper error handling and timeout.
    
    Args:
        service_id: Identifier for the service
        timeout: Maximum time to wait for response in seconds
        
    Returns:
        Dictionary containing the service response
        
    Raises:
        asyncio.TimeoutError: If the request times out
        ServiceUnavailableError: If the service is unavailable
        Exception: For other unexpected errors
    """
    # Simulate a resource that needs cleanup
    resource = {"connection": f"connection_to_{service_id}", "open": True}
    
    try:
        logger.info(f"Fetching data from service: {service_id}")
        
        # Implement timeout using asyncio.wait_for
        return await asyncio.wait_for(
            _fetch_with_retry(service_id),
            timeout=timeout
        )
    
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching data from service: {service_id}")
        raise asyncio.TimeoutError(f"Request to {service_id} timed out after {timeout} seconds")
    
    except ServiceUnavailableError:
        logger.error(f"Service unavailable: {service_id}")
        raise
    
    except asyncio.CancelledError:
        logger.warning(f"Task for service {service_id} was cancelled")
        # Always re-raise CancelledError to properly propagate cancellation
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error fetching data from {service_id}: {e}")
        raise
    
    finally:
        # Always clean up resources, even if an error occurred
        if resource["open"]:
            logger.debug(f"Closing connection to {service_id}")
            resource["open"] = False


async def _fetch_with_retry(service_id: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Internal function to fetch data with retry logic.
    
    Implements exponential backoff for retries.
    """
    retries = 0
    
    while True:
        try:
            # Simulate a service call that might fail
            return await _simulate_service_call(service_id)
        
        except ServiceUnavailableError:
            retries += 1
            if retries > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded for {service_id}")
                raise
            
            # Exponential backoff with jitter
            backoff = (2 ** retries) * 0.1 + (random.random() * 0.1)
            logger.warning(f"Retrying {service_id} in {backoff:.2f} seconds (attempt {retries}/{max_retries})")
            await asyncio.sleep(backoff)


async def _simulate_service_call(service_id: str) -> Dict[str, Any]:
    """Simulate a call to a service that might fail."""
    # Simulate some async work
    await asyncio.sleep(0.1)
    
    # Randomly simulate different outcomes
    rand = random.random()
    
    if rand < 0.2:  # 20% chance of service being unavailable
        raise ServiceUnavailableError(f"Service {service_id} is unavailable")
    
    if rand < 0.3:  # 10% chance of other error
        raise Exception(f"Unexpected error from service {service_id}")
    
    # Success case
    return {
        "service_id": service_id,
        "status": "success",
        "data": {"value": random.randint(1, 100)}
    }


async def fetch_from_multiple_services(service_ids: list[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch data from multiple services concurrently with error isolation.
    
    This function demonstrates how to handle errors in individual tasks
    without failing the entire operation.
    """
    logger.info(f"Fetching data from {len(service_ids)} services")
    
    # Use gather with return_exceptions=True to prevent one failure from cancelling all tasks
    tasks = [
        asyncio.create_task(fetch_data_from_service(service_id))
        for service_id in service_ids
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results, handling any exceptions
    processed_results = {}
    for service_id, result in zip(service_ids, results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to fetch from {service_id}: {result}")
            processed_results[service_id] = None
        else:
            processed_results[service_id] = result
    
    return processed_results


# Example usage
async def main():
    service_ids = ["service1", "service2", "service3", "service4", "service5"]
    
    try:
        results = await fetch_from_multiple_services(service_ids)
        
        # Print results
        for service_id, result in results.items():
            if result is None:
                print(f"{service_id}: Failed to fetch data")
            else:
                print(f"{service_id}: {result['status']} - {result['data']}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 