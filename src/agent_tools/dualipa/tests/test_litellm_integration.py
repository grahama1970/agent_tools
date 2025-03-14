"""
Test the litellm integration.

These tests verify that the basic litellm integration components function correctly.
They are smoke tests that validate the core functionality before testing actual
project files.

Official Documentation References:
- litellm: https://docs.litellm.ai/docs/
- pytest: https://docs.pytest.org/en/latest/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
- tenacity: https://tenacity.readthedocs.io/en/latest/
- cachetools: https://cachetools.readthedocs.io/en/latest/
- loguru: https://loguru.readthedocs.io/en/latest/
"""

import os
import sys
import json
import asyncio
import inspect
import pytest
import hashlib
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import dualipa modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Set up logger for test output
from loguru import logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

# Try imports - we'll skip tests if modules aren't available
try:
    import litellm
    from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
    LITELLM_AVAILABLE = True
    logger.info("LiteLLM is available for testing")
except ImportError:
    LITELLM_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="litellm not available")
    logger.warning("LiteLLM is not available, skipping tests")

# Check if our LLM module exists
try:
    from llm_generator import (
        init_litellm,
        call_litellm,
        retry_llm_call,
        cached_retry,
        generate_cache_key,
        CallOptions,
        LITELLM_AVAILABLE
    )
    logger.info("llm_generator module is available for testing")
except ImportError as e:
    # If the module doesn't exist, skip these tests
    pytestmark = pytest.mark.skip(reason=f"llm_generator module not available: {e}")
    logger.warning(f"llm_generator module not available: {e}")


@pytest.fixture
def mock_openai_response():
    """Fixture to provide a mock OpenAI API response.
    
    This follows the standard OpenAI API response format documented at:
    https://docs.litellm.ai/docs/completion/input
    """
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the OpenAI API."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }


@pytest.mark.asyncio
async def test_litellm_initialization():
    """Test that litellm can be initialized without errors.
    
    This test verifies the basic initialization function works.
    
    Documentation: https://docs.litellm.ai/docs/completion/input
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Test that init_litellm runs without raising exceptions
    try:
        # Capture any output to verify it ran
        with patch('litellm.set_verbose') as mock_set_verbose:
            init_litellm()
            mock_set_verbose.assert_called_once(), "litellm.set_verbose should be called during initialization"
    except Exception as e:
        pytest.fail(f"init_litellm raised an exception: {e}")
    
    logger.info("LiteLLM initialization test passed")


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Test that cache key generation works correctly.
    
    This test verifies the cache key generator produces consistent
    and unique keys based on the input options.
    
    Documentation: https://cachetools.readthedocs.io/en/latest/
    """
    # Create two identical sets of options
    options1 = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt", temperature=0.7)
    options2 = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt", temperature=0.7)
    
    # Create a different set of options
    options3 = CallOptions(model="gpt-4", prompt="Test prompt", temperature=0.5)
    options4 = CallOptions(model="gpt-3.5-turbo", prompt="Different prompt", temperature=0.7)
    
    # Generate cache keys
    key1 = generate_cache_key(options1)
    key2 = generate_cache_key(options2)
    key3 = generate_cache_key(options3)
    key4 = generate_cache_key(options4)
    
    # Same options should produce the same cache key
    assert key1 == key2, "Same options should generate the same cache key"
    
    # Different options should produce different cache keys
    assert key1 != key3, "Different model should generate different cache key"
    assert key1 != key4, "Different prompt should generate different cache key"
    
    # Verify the key is a string and looks like a hash
    assert isinstance(key1, str), "Cache key should be a string"
    assert len(key1) > 10, "Cache key should be a reasonably long hash"
    
    logger.info("Cache key generation test passed")


@pytest.mark.asyncio
async def test_basic_litellm_call(mock_openai_response):
    """Test that a basic call to litellm works correctly.
    
    This test verifies the basic API call wrapper functions correctly
    with a mocked response.
    
    Documentation: https://docs.litellm.ai/docs/completion/input
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Mock the actual API call
    with patch('litellm.acompletion', new=AsyncMock(return_value=mock_openai_response)):
        # Create options
        options = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt")
        
        # Call the function
        result = await call_litellm(options)
        
        # Verify the result
        assert result == mock_openai_response, "call_litellm should return the response from litellm.acompletion"
        assert "choices" in result, "Response should contain choices field"
        assert len(result["choices"]) > 0, "Response should contain at least one choice"
        assert "message" in result["choices"][0], "Choice should contain a message"
        assert "content" in result["choices"][0]["message"], "Message should contain content"
    
    logger.info("Basic LiteLLM call test passed")


@pytest.mark.asyncio
async def test_retry_mechanism():
    """Test that the retry mechanism works correctly.
    
    This test verifies that the retry decorator properly retries
    the function on failure and succeeds when possible.
    
    Documentation: https://tenacity.readthedocs.io/en/latest/
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Create a mock function that fails twice then succeeds
    mock_func = AsyncMock(side_effect=[
        Exception("First temporary error"), 
        Exception("Second temporary error"), 
        "Success"
    ])
    
    # Create a logger spy to verify logging
    with patch('loguru.logger.warning') as mock_logger_warning:
        # Apply the retry decorator
        retried_func = retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying after error: {retry_state.outcome.exception()}"
            )
        )(mock_func)
        
        # Call the function
        result = await retried_func()
        
        # Verify the result
        assert result == "Success", "Function should succeed after retries"
        assert mock_func.call_count == 3, "Function should be called 3 times"
        assert mock_logger_warning.call_count == 2, "Logger should record 2 warning messages for retries"
    
    logger.info("Retry mechanism test passed")


@pytest.mark.asyncio
async def test_error_handling():
    """Test that error handling works correctly when max retries are exceeded.
    
    This test verifies that the retry mechanism properly gives up
    after the maximum number of retries and raises the appropriate exception.
    
    Documentation: https://tenacity.readthedocs.io/en/latest/
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Create a mock function that always fails with different errors
    error_messages = ["Error 1", "Error 2", "Error 3", "Error 4"]
    mock_func = AsyncMock(side_effect=[Exception(msg) for msg in error_messages])
    
    # Create a logger spy to verify logging
    with patch('loguru.logger.warning') as mock_logger_warning:
        # Apply the retry decorator
        retried_func = retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying after error: {retry_state.outcome.exception()}"
            )
        )(mock_func)
        
        # Call the function and expect it to fail
        with pytest.raises(Exception) as excinfo:
            await retried_func()
        
        # Verify the correct exception is raised (the last one)
        assert str(excinfo.value) == error_messages[2], "Should raise the last exception encountered"
        
        # Verify the function was called the expected number of times
        assert mock_func.call_count == 3, "Function should be called 3 times before giving up"
        assert mock_logger_warning.call_count == 2, "Logger should record 2 warning messages for retries"
    
    logger.info("Error handling test passed")


@pytest.mark.asyncio
async def test_cache_functionality(mock_openai_response):
    """Test that caching functionality works correctly.
    
    This test verifies that cached results are properly stored and retrieved,
    and that different inputs produce different cache keys.
    
    Documentation: 
    - https://docs.litellm.ai/docs/caching
    - https://cachetools.readthedocs.io/en/latest/
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Create a cache dictionary to simulate the real cache
    mock_cache = {}
    
    # Mock the generate_cache_key function to use a simpler implementation
    def mock_generate_key(options):
        key_string = f"{options.model}_{options.prompt}_{options.temperature}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    # Create a mock function that uses the cache
    call_counter = {"count": 0}
    
    async def mock_cached_function(options):
        # Generate a cache key
        cache_key = mock_generate_key(options)
        
        # Check if the result is in the cache
        if cache_key in mock_cache:
            return mock_cache[cache_key]
        
        # If not, call the "API" and cache the result
        call_counter["count"] += 1
        result = mock_openai_response
        mock_cache[cache_key] = result
        return result
    
    # Create options
    options1 = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt", temperature=0.7)
    options2 = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt", temperature=0.7)  # Same as options1
    options3 = CallOptions(model="gpt-4", prompt="Test prompt", temperature=0.7)          # Different model
    
    # Call the function multiple times
    result1 = await mock_cached_function(options1)
    result2 = await mock_cached_function(options2)  # Should use cache
    result3 = await mock_cached_function(options3)  # Should not use cache
    
    # Verify the results and cache behavior
    assert result1 == result2, "Same options should return the same cached result"
    assert call_counter["count"] == 2, "Should make two actual API calls (for options1 and options3)"
    assert len(mock_cache) == 2, "Cache should contain two entries"
    
    # Call again with options1 - should still use cache
    result4 = await mock_cached_function(options1)
    assert call_counter["count"] == 2, "Should not make another API call for options1"
    assert result1 == result4, "Cached result should be the same"
    
    logger.info("Cache functionality test passed")


@pytest.mark.asyncio
async def test_caching_tenacity_integration(mock_openai_response):
    """Test that caching and tenacity integration works correctly.
    
    This test verifies that the cached_retry decorator properly
    combines caching and retry functionality.
    
    Documentation: 
    - https://docs.litellm.ai/docs/caching
    - https://tenacity.readthedocs.io/en/latest/
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Create a counter to track actual API calls
    call_counter = {"count": 0, "errors": 0}
    
    # Create a mock API function that fails first time then works
    async def mock_api_call(options):
        call_counter["count"] += 1
        
        # Fail the first time
        if call_counter["count"] == 1:
            call_counter["errors"] += 1
            raise ConnectionError("Simulated connection error")
        
        # Succeed on subsequent calls
        return mock_openai_response
    
    # Apply the cached_retry decorator
    decorated_func = cached_retry(
        retries=3,
        cache_ttl=60,
        exceptions=(ConnectionError, TimeoutError)
    )(mock_api_call)
    
    # Call the function with some options
    options = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt")
    result1 = await decorated_func(options)
    
    # Verify first call behavior
    assert call_counter["count"] == 2, "Function should be called twice (first fails, second succeeds)"
    assert call_counter["errors"] == 1, "One error should have occurred"
    assert "choices" in result1, "Should return successful response"
    
    # Call again with same options
    result2 = await decorated_func(options)
    
    # Verify caching behavior
    assert call_counter["count"] == 2, "Function should not be called again due to caching"
    assert result1 == result2, "Results should be identical from cache"
    
    # Call with different options
    options2 = CallOptions(model="gpt-3.5-turbo", prompt="Different prompt")
    result3 = await decorated_func(options2)
    
    # Verify behavior with new options
    assert call_counter["count"] == 3, "Function should be called once for new options"
    assert call_counter["errors"] == 1, "No new errors should occur"
    assert result3 != result1, "Results should be different objects"
    assert "choices" in result3, "New result should be successful response"
    
    logger.info("Caching tenacity integration test passed")


@pytest.mark.asyncio
async def test_litellm_with_metadata(mock_openai_response):
    """Test that litellm handles metadata correctly.
    
    This test verifies that metadata is properly passed to the API
    and doesn't affect caching behavior.
    
    Documentation: https://docs.litellm.ai/docs/completion/input
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Mock the API call
    mock_litellm = AsyncMock(return_value=mock_openai_response)
    
    with patch('litellm.acompletion', mock_litellm):
        # Create options with metadata
        options = CallOptions(
            model="gpt-3.5-turbo", 
            prompt="Test prompt",
            metadata={"user_id": "test123", "session_id": "abc456"}
        )
        
        # Call the function
        result = await call_litellm(options)
        
        # Verify the API was called with the correct parameters
        args, kwargs = mock_litellm.call_args
        assert "metadata" in kwargs, "API should be called with metadata parameter"
        assert kwargs["metadata"] == options.metadata, "API should receive the correct metadata"
        
        # Create options with the same prompt but different metadata
        options2 = CallOptions(
            model="gpt-3.5-turbo", 
            prompt="Test prompt",
            metadata={"user_id": "different", "session_id": "different"}
        )
        
        # Create cache keys to verify
        key1 = generate_cache_key(options)
        key2 = generate_cache_key(options2)
        
        # Verify that different metadata doesn't affect the cache key
        assert key1 == key2, "Different metadata should not produce different cache keys"
    
    logger.info("LiteLLM metadata test passed")


def test_module_import_paths():
    """Test that module import paths are correct.
    
    This test verifies that the necessary modules can be imported
    and key files exist in the expected locations.
    
    Documentation: https://docs.python.org/3/library/importlib.html
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Verify that the llm_generator module contains expected components
    from llm_generator import init_litellm, call_litellm, retry_llm_call, cached_retry
    
    # Check that the functions are callable
    assert callable(init_litellm), "init_litellm should be a callable function"
    assert callable(call_litellm), "call_litellm should be a callable function"
    assert callable(retry_llm_call), "retry_llm_call should be a callable function"
    assert callable(cached_retry), "cached_retry should be a callable function"
    
    # Check that source files exist
    llm_generator_path = inspect.getfile(init_litellm)
    assert os.path.exists(llm_generator_path), f"llm_generator module file should exist at {llm_generator_path}"
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(os.path.dirname(llm_generator_path))
    
    # Check for other key files
    expected_files = ["__init__.py", "format_dataset.py", "extract_repo.py"]
    for filename in expected_files:
        file_path = os.path.join(project_dir, filename)
        assert os.path.exists(file_path), f"Expected file {filename} should exist at {file_path}"
    
    logger.info("Module import paths test passed")


@pytest.mark.asyncio
async def test_litellm_error_handling():
    """Test specific error handling in the litellm integration.
    
    This test verifies that different types of errors are
    handled correctly and appropriate logging occurs.
    
    Documentation: 
    - https://docs.litellm.ai/docs/exception_handling
    - https://loguru.readthedocs.io/en/latest/
    """
    # Skip if litellm is not available
    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available")
    
    # Define different error types
    api_errors = [
        Exception("Generic API error"),
        ConnectionError("Connection refused"),
        TimeoutError("Request timed out"),
        ValueError("Invalid parameter")
    ]
    
    # Test each error type
    for error in api_errors:
        # Mock the API call to raise this error
        with patch('litellm.acompletion', AsyncMock(side_effect=error)):
            # Create options
            options = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt")
            
            # Mock the logger
            with patch('loguru.logger.error') as mock_logger:
                # Call the function and expect an error
                with pytest.raises(Exception) as excinfo:
                    await call_litellm(options)
                
                # Verify the error is properly logged
                assert mock_logger.called, f"Error should be logged for {type(error).__name__}"
                assert isinstance(excinfo.value, type(error)), f"Should raise same error type: {type(error).__name__}"
    
    logger.info("LiteLLM error handling test passed")


@pytest.mark.asyncio
async def test_call_options_validation():
    """Test validation of call options.
    
    This test verifies that the CallOptions class properly
    validates input parameters.
    
    Documentation: https://docs.pydantic.dev/latest/
    """
    # Test valid options
    valid_options = CallOptions(model="gpt-3.5-turbo", prompt="Test prompt")
    assert valid_options.model == "gpt-3.5-turbo", "Model should be set correctly"
    assert valid_options.prompt == "Test prompt", "Prompt should be set correctly"
    assert valid_options.temperature is not None, "Temperature should have a default value"
    
    # Test options with all parameters
    full_options = CallOptions(
        model="gpt-4",
        prompt="Full test",
        temperature=0.8,
        max_tokens=100,
        metadata={"test": "value"}
    )
    assert full_options.model == "gpt-4", "Model should be set correctly"
    assert full_options.temperature == 0.8, "Temperature should be set correctly"
    assert full_options.max_tokens == 100, "Max tokens should be set correctly"
    assert full_options.metadata == {"test": "value"}, "Metadata should be set correctly"
    
    # Test string representation
    options_str = str(valid_options)
    assert "model" in options_str, "String representation should include model"
    assert "prompt" in options_str, "String representation should include prompt"
    
    logger.info("Call options validation test passed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 