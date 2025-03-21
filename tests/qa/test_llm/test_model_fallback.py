"""Tests for model fallback strategy and cost-aware routing.

This module tests the model fallback strategy and cost-aware routing capabilities
of the LLM API call mechanism. These tests verify that the system properly
falls back to alternative models when the primary model fails and routes requests
to the most cost-effective model based on input complexity.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- tenacity: https://github.com/jd/tenacity
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html

Expected test coverage:
- Model fallback on persistent failures
- Cost-aware routing based on content size
- Prioritization based on model cost vs. quality
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from agent_tools.dualipa.qa.llm.retry_llm_call import (
    retry_llm_call,
    APIError,
    circuit_state,
    mock_litellm_call
)

# Define an error model that initially fails but succeeds on fallback
async def fail_once_then_succeed_on_fallback(config):
    model = config.get("model", "")
    if model == "gpt-4-turbo":
        raise APIError("Simulated API error for testing fallback")
    else:
        # Return success for fallback model
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"question": "What is this?", "answer": "A test", "reasoning": "Testing fallback. Oh wait?! This is for model fallback."}'
                    }
                }
            ]
        }

# Define an error model for cost routing testing
async def cost_aware_test_llm_call(config):
    model = config.get("model", "")
    messages = config.get("messages", [])
    message_content = ""
    
    if messages and len(messages) > 0:
        message_content = messages[0].get("content", "")
    
    # Track which model was chosen for testing by adding to the response content
    return {
        "choices": [
            {
                "message": {
                    "content": f'{{"question": "What is this?", "answer": "A test with {model}", "reasoning": "Testing model selection. Oh wait?! This is for {model}."}}'
                }
            }
        ],
        "model_used": model  # Add this field to track which model was used
    }


@pytest.mark.asyncio
async def test_fallback_model():
    """Test that retry mechanism falls back to alternative model on failures.
    
    This test verifies that:
    1. When the primary model fails, a fallback model is attempted
    2. The fallback model is used automatically after retries
    3. The response from the fallback model is properly processed
    
    Input:
        - Test configuration with primary model
        - A mock LLM call function that fails for primary but succeeds for fallback
    
    Expected output:
        - Response from the fallback model
        - Log evidence that fallback occurred
    
    Dependencies:
        - retry_llm_call function from agent_tools.dualipa.qa.llm.retry_llm_call
        - pytest-asyncio for async test support
    """
    test_config = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": "Test content for fallback"}]
    }
    
    # Call retry_llm_call with our test function
    response = await retry_llm_call(
        config=test_config,
        max_retries=1,  # Only retry once to make the test faster
        llm_call_func=fail_once_then_succeed_on_fallback  # Mock function that fails then succeeds
    )
    
    # Check that we got a valid response
    assert response is not None
    assert "choices" in response
    assert len(response["choices"]) > 0
    
    # Check the content to verify it's from the fallback model
    content = response["choices"][0]["message"]["content"]
    assert "testing fallback" in content.lower()


@pytest.mark.asyncio
async def test_cost_aware_routing():
    """Test that the system routes requests to appropriate models based on content size.
    
    This test verifies that:
    1. Small requests use a cheaper model (gpt-3.5-turbo)
    2. Large requests use a more capable model (gpt-4-turbo)
    3. The cost-aware routing can be toggled
    
    Input:
        - Small content test configuration
        - Large content test configuration
        - Cost-aware routing setting
    
    Expected output:
        - Small content uses cheaper model
        - Large content uses more capable model
        - When cost-aware is disabled, requested model is used regardless
    
    Dependencies:
        - retry_llm_call function from agent_tools.dualipa.qa.llm.retry_llm_call
        - pytest-asyncio for async test support
    """
    # Test with small content
    small_content = "This is a short test prompt that should use the cheaper model."
    small_config = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": small_content}]
    }
    
    # Test with large content
    large_content = "This is a much longer test prompt " + ("with repeated text " * 50) + "that should use the more capable model due to its size exceeding the threshold for cost-aware routing."
    large_config = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": large_content}]
    }
    
    # Test small content with cost-aware routing enabled
    small_response = await retry_llm_call(
        config=small_config.copy(),
        cost_aware=True,
        llm_call_func=cost_aware_test_llm_call
    )
    
    # Test large content with cost-aware routing enabled
    large_response = await retry_llm_call(
        config=large_config.copy(),
        cost_aware=True,
        llm_call_func=cost_aware_test_llm_call
    )
    
    # Test small content with cost-aware routing disabled
    small_no_cost_aware = await retry_llm_call(
        config=small_config.copy(),
        cost_aware=False,
        llm_call_func=cost_aware_test_llm_call
    )
    
    # Verify small content used cheaper model with cost-aware routing
    assert small_response["model_used"] == "gpt-3.5-turbo", "Small content should use cheaper model"
    
    # Verify large content used more capable model with cost-aware routing
    assert large_response["model_used"] == "gpt-4-turbo", "Large content should use more capable model"
    
    # Verify requested model is used when cost-aware routing is disabled
    assert small_no_cost_aware["model_used"] == "gpt-4-turbo", "With cost-aware off, requested model should be used"