"""LLM API call retry utilities.

This module provides utilities for retrying LLM API calls with exponential backoff,
circuit breaker pattern, and fallback models. It implements robust error handling
with a configurable circuit breaker and persistent dead letter queue.

The module implements the following resilience patterns:
1. Retry with exponential backoff: Automatically retry failed API calls
2. Circuit breaker: Prevent cascading failures by failing fast when service is degraded
3. Dead letter queue: Store failed requests for later analysis or replay
4. Fallback models: Use alternative models when primary model fails consistently

Official documentation:
- tenacity: https://github.com/jd/tenacity
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
- logging: https://docs.python.org/3/library/logging.html
- uuid: https://docs.python.org/3/library/uuid.html
- datetime: https://docs.python.org/3/library/datetime.html
- pathlib: https://docs.python.org/3/library/pathlib.html

Expected input/output:
- retry_llm_call: Takes LLM configuration, returns API response or raises RetryError
- add_to_dead_letter_queue: Takes failed request configuration, persists to queue
- process_dead_letter_queue: Processes queued requests, returns counts of processed/succeeded
- reset_circuit_if_needed: Checks and resets circuit breaker if timeout has passed
"""

import os
import json
import time
import logging
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

# Import cache functionality
from agent_tools.dualipa.qa.utils.cache import (
    initialize_cache,
    get_from_cache,
    add_to_cache
)

logger = logging.getLogger(__name__)

# For MVP, we'll mock the LLM client since this is just a demo
# In a real implementation, this would be replaced with litellm or another client
class APIError(Exception):
    """API error exception."""
    pass


# Circuit breaker state
circuit_state = {
    "open": False,
    "failures": 0,
    "last_failure_time": 0,
    "reset_timeout": 300,  # 5 minutes
    "failure_threshold": 5,  # Number of failures before opening circuit
    "half_open_success": 0,  # Successes in half-open state
    "success_threshold": 3,  # Successes needed to close circuit
    "state": "closed"  # closed, open, half-open
}

# Dead letter queue for failed requests
dead_letter_queue = []
DEAD_LETTER_FILE = "dead_letter.json"  # Module level constant used by tests
MAX_QUEUE_SIZE = 1000  # Prevent unbounded growth


async def reset_circuit_if_needed():
    """Reset circuit breaker if timeout has passed.
    
    This implements the circuit breaker pattern which prevents cascading failures
    by stopping requests after multiple failures. After a timeout period, the
    circuit enters a 'half-open' state where a limited number of requests are
    allowed through to test if the system has recovered.
    
    Returns:
        bool: True if circuit was reset to half-open, False otherwise
    """
    if not circuit_state["open"]:
        return False
        
    current_time = time.time()
    if current_time - circuit_state["last_failure_time"] > circuit_state["reset_timeout"]:
        logger.info("Transitioning circuit breaker from open to half-open state")
        
        # Move to half-open state
        circuit_state["open"] = False
        circuit_state["state"] = "half-open"
        circuit_state["half_open_success"] = 0
        
        return True
    
    return False


async def mock_litellm_call(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mock LLM API call for MVP.
    
    In a real implementation, this would be replaced with litellm.
    
    Args:
        config: Model configuration
        
    Returns:
        Mock API response
    """
    model = config.get("model", "gpt-4-turbo")
    temperature = config.get("temperature", 0.5)
    messages = config.get("messages", [])
    
    # Simulate response - fails for both original and fallback uses of error-model
    if "error-model" in model:
        raise APIError("Simulated API error")
    
    # Determine the type of request based on messages
    is_reversal = False
    for message in messages:
        if message.get("role") == "user" and "Given the following answer" in message.get("content", ""):
            is_reversal = True
            break
    
    if is_reversal:
        # This is a reversal request (generate question from answer)
        mock_reversed_pair = {
            "question": f"Why is this technology useful for {temperature:.1f} type applications?",
            "reasoning": f"Based on the provided answer, I can see it's about a technology. Oh wait?! It appears to be specifically about its applications at {temperature:.1f} temperature settings."
        }
        response_content = json.dumps(mock_reversed_pair)
    else:
        # Standard QA generation
        mock_qa_pair = {
            "question": "What is the purpose of this module?",
            "answer": "This module provides a retry mechanism for LLM API calls.",
            "reasoning": "Based on the imports and function names, this module is designed to handle retries for LLM API calls with exponential backoff. Oh wait?! It also implements a circuit breaker pattern for error handling."
        }
        
        # Simulate different outputs based on temperature
        if temperature > 0.6:
            mock_qa_pair["question"] = "How does the circuit breaker pattern work in this module?"
            
        response_content = json.dumps([mock_qa_pair])
    
    # Mock response format
    return {
        "choices": [
            {
                "message": {
                    "content": response_content
                }
            }
        ]
    }


async def add_to_dead_letter_queue(config: Dict[str, Any], error: Optional[str] = None) -> None:
    """Add a failed request to the dead letter queue and persist it.
    
    The dead letter queue stores failed requests for later analysis and potential
    replay. It is persisted to disk to survive process restarts.
    
    Args:
        config: The request configuration
        error: Optional error message
    """
    entry = {
        "config": config, 
        "timestamp": datetime.now().isoformat(),
        "error": error or "Unknown error"
    }
    
    if "request_id" not in config:
        entry["config"]["request_id"] = str(uuid.uuid4())
    
    dead_letter_queue.append(entry)
    if len(dead_letter_queue) > MAX_QUEUE_SIZE:
        dead_letter_queue.pop(0)
        
    try:
        with open(DEAD_LETTER_FILE, "w") as f:
            json.dump(dead_letter_queue, f, indent=2)
        logger.info(f"Added request to dead letter queue, current size: {len(dead_letter_queue)}")
    except Exception as save_error:
        logger.error(f"Failed to save dead letter queue: {save_error}")


async def load_dead_letter_queue() -> List[Dict[str, Any]]:
    """Load the dead letter queue from disk.
    
    Returns:
        The dead letter queue
    """
    try:
        if os.path.exists(DEAD_LETTER_FILE):
            with open(DEAD_LETTER_FILE, "r") as f:
                return json.load(f)
    except Exception as load_error:
        logger.error(f"Failed to load dead letter queue: {load_error}")
    
    return []


async def process_dead_letter_queue(
    max_items: int = 10, 
    retry_interval_hours: int = 24
) -> Tuple[int, int]:
    """Process items from the dead letter queue.
    
    This function attempts to replay failed requests that have aged
    for a specified interval, helping to recover from transient failures.
    
    Args:
        max_items: Maximum number of items to process
        retry_interval_hours: Minimum age in hours before retrying
        
    Returns:
        Tuple of (processed_count, success_count)
    """
    if not dead_letter_queue:
        dead_letter_queue.extend(await load_dead_letter_queue())
        
    if not dead_letter_queue:
        return (0, 0)
    
    now = datetime.now()
    retry_delta = retry_interval_hours * 3600
    
    eligible_indices = []
    for i, item in enumerate(dead_letter_queue):
        try:
            item_time = datetime.fromisoformat(item["timestamp"])
            age_seconds = (now - item_time).total_seconds()
            if age_seconds >= retry_delta:
                eligible_indices.append(i)
                if len(eligible_indices) >= max_items:
                    break
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid item in dead letter queue: {e}")
    
    processed_count = 0
    success_count = 0
    
    for i in sorted(eligible_indices, reverse=True):
        try:
            item = dead_letter_queue[i]
            config = item["config"]
            if "model" in config and config["model"] != "gpt-3.5-turbo":
                config["model"] = "gpt-3.5-turbo"
            # Disable retry decorator for dead letter processing
            response = await mock_litellm_call(config)
            dead_letter_queue.pop(i)
            success_count += 1
            logger.info(f"Successfully processed dead letter item: {config.get('request_id')}")
        except Exception as e:
            logger.warning(f"Failed to process dead letter item: {e}")
        processed_count += 1
    
    try:
        with open(DEAD_LETTER_FILE, "w") as f:
            json.dump(dead_letter_queue, f, indent=2)
    except Exception as save_error:
        logger.error(f"Failed to save dead letter queue: {save_error}")
    
    return (processed_count, success_count)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(APIError)
)
async def retry_llm_call(
    config: Dict[str, Any], 
    cost_aware: bool = True,
    max_retries: int = 3,
    circuit_breaker_enabled: bool = True,
    use_cache: bool = True,
    llm_call_func=None  # Dependency injection for easier testing
) -> Dict[str, Any]:
    """Retry LLM API call with exponential backoff, circuit breaker, and caching.
    
    Args:
        config: Model configuration
        cost_aware: Whether to use cost-aware routing
        max_retries: Maximum number of retries
        circuit_breaker_enabled: Whether to use circuit breaker
        use_cache: Whether to use cache for requests
        llm_call_func: Optional function to call the LLM API (for testing/injection)
        
    Returns:
        API response
        
    Raises:
        APIError: If the API call fails after retries
    """
    if llm_call_func is None:
        llm_call_func = mock_litellm_call

    if "request_id" not in config:
        config["request_id"] = str(uuid.uuid4())
    
    # Check cache first if enabled
    if use_cache:
        # Initialize cache if this is the first call
        try:
            cached_result = get_from_cache(config)
            if cached_result is not None:
                logger.debug(f"Cache hit for request {config['request_id']}")
                return cached_result
        except Exception as cache_error:
            logger.warning(f"Cache access error: {cache_error}")
    
    if circuit_breaker_enabled:
        await reset_circuit_if_needed()
        if circuit_state["open"]:
            logger.warning(f"Circuit breaker open, sending to dead letter queue: {config['request_id']}")
            await add_to_dead_letter_queue(config, "Circuit breaker open")
            raise APIError("Circuit breaker open")
    
    if cost_aware:
        content_size = 0
        if "messages" in config:
            for msg in config["messages"]:
                content_size += len(msg.get("content", ""))
        if config.get("model") == "gpt-4-turbo" and content_size < 500:
            config["model"] = "gpt-3.5-turbo"
            logger.info(f"Cost-aware routing: Using cheaper model for request {config['request_id']}")
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            response = await llm_call_func(config)
            
            # Update circuit breaker state if needed
            if circuit_breaker_enabled and circuit_state["state"] == "half-open":
                circuit_state["half_open_success"] += 1
                if circuit_state["half_open_success"] >= circuit_state["success_threshold"]:
                    logger.info("Circuit breaker closing after successful requests")
                    circuit_state["state"] = "closed"
                    circuit_state["failures"] = 0
            
            # Add successful response to cache if enabled
            if use_cache:
                try:
                    add_to_cache(config, response)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache response: {cache_error}")
            
            return response
        except APIError as e:
            last_error = e
            retry_count += 1
            
            if circuit_breaker_enabled:
                circuit_state["failures"] += 1
                circuit_state["last_failure_time"] = time.time()
                if circuit_state["failures"] >= circuit_state["failure_threshold"]:
                    logger.error(f"Circuit breaker opened after {circuit_state['failures']} failures")
                    circuit_state["open"] = True
                    circuit_state["state"] = "open"
                    await add_to_dead_letter_queue(config, str(e))
                    raise APIError(f"Circuit breaker opened: {str(e)}")
                if circuit_state["state"] == "half-open":
                    logger.warning("Failure in half-open state, reopening circuit")
                    circuit_state["open"] = True
                    circuit_state["state"] = "open"
                    await add_to_dead_letter_queue(config, str(e))
                    raise APIError(f"Circuit reopened in half-open state: {str(e)}")
            
            if retry_count == max_retries and config.get("model") != "gpt-3.5-turbo":
                logger.warning(f"Trying fallback model after {retry_count} failures: {config['request_id']}")
                fallback_config = config.copy()
                fallback_config["model"] = "gpt-3.5-turbo"
                try:
                    fallback_response = await llm_call_func(fallback_config)
                    
                    # Add fallback response to cache if enabled
                    if use_cache:
                        try:
                            add_to_cache(fallback_config, fallback_response)
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache fallback response: {cache_error}")
                    
                    return fallback_response
                except APIError as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    await add_to_dead_letter_queue(config, str(fallback_error))
                    raise
            
            backoff = min(30, 2 ** retry_count)
            logger.warning(f"Retry {retry_count}/{max_retries} after {backoff}s: {str(e)}")
            await asyncio.sleep(backoff)
    
    if last_error:
        await add_to_dead_letter_queue(config, str(last_error))
    raise APIError(f"Failed after {max_retries} retries: {str(last_error)}")
