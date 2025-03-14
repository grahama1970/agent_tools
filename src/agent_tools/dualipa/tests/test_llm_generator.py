"""
Test the llm_generator module integration with litellm.

These tests verify that the llm_generator module properly integrates with
the litellm components for generating question-answer pairs.

Official Documentation References:
- litellm: https://docs.litellm.ai/docs/
- pytest: https://docs.pytest.org/en/latest/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
"""

import os
import sys
import json
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path

# Add the parent directory to the path to import dualipa modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try to import the llm_generator module
try:
    from llm_generator import (
        generate_code_qa_pairs,
        generate_markdown_qa_pairs,
        generate_reverse_qa_pairs,
        QuestionAnswerPair,
        LITELLM_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Skipping llm_generator tests: {e}", allow_module_level=True)

# Skip if litellm is not available
pytestmark = pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not available")


@pytest.fixture
def mock_llm_response():
    """Fixture to provide a mock LLM response with QA pairs."""
    return {
        "choices": [
            {
                "message": {
                    "content": """[
                        {
                            "question": "What does the calculate_average function do?",
                            "answer": "The calculate_average function calculates the average of a list of numbers."
                        },
                        {
                            "question": "What happens if an empty list is passed to calculate_average?",
                            "answer": "If an empty list is passed, the function raises a ValueError with the message 'Cannot calculate average of empty list'."
                        }
                    ]"""
                }
            }
        ]
    }


@pytest.mark.asyncio
async def test_generate_code_qa_pairs(mock_llm_response):
    """Test that generate_code_qa_pairs works correctly."""
    test_code = """
def calculate_average(numbers: list) -> float:
    \"\"\"Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers
        
    Returns:
        The average of the numbers
        
    Raises:
        ValueError: If the list is empty
    \"\"\"
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
"""
    
    # Mock the retry_llm_call function to return our mock response
    with patch('llm_generator.retry_llm_call', new=AsyncMock(return_value=mock_llm_response)):
        # Call the function
        result = await generate_code_qa_pairs(test_code, max_pairs=2)
        
        # Verify the result
        assert len(result) == 2, "Should return 2 QA pairs"
        assert result[0]["question"] == "What does the calculate_average function do?"
        assert "average" in result[0]["answer"]
        assert "ValueError" in result[1]["answer"]


@pytest.mark.asyncio
async def test_generate_markdown_qa_pairs(mock_llm_response):
    """Test that generate_markdown_qa_pairs works correctly."""
    test_markdown = """# Test Documentation
    
This is a test document for Q&A generation.

## Installation

To install the package, run:

```
pip install mypackage
```
"""
    
    # Mock the retry_llm_call function to return our mock response
    with patch('llm_generator.retry_llm_call', new=AsyncMock(return_value=mock_llm_response)):
        # Call the function
        result = await generate_markdown_qa_pairs(test_markdown, max_pairs=2)
        
        # Verify the result
        assert len(result) == 2, "Should return 2 QA pairs"
        assert result[0]["question"] == "What does the calculate_average function do?"
        assert "average" in result[0]["answer"]


@pytest.mark.asyncio
async def test_generate_reverse_qa_pairs(mock_llm_response):
    """Test that generate_reverse_qa_pairs works correctly."""
    original_pairs = [
        {
            "question": "What does the calculate_average function do?",
            "answer": "The calculate_average function calculates the average of a list of numbers."
        },
        {
            "question": "What happens if an empty list is passed?",
            "answer": "If an empty list is passed, the function raises a ValueError."
        }
    ]
    
    # Mock the retry_llm_call function to return our mock response
    with patch('llm_generator.retry_llm_call', new=AsyncMock(return_value=mock_llm_response)):
        # Call the function
        result = await generate_reverse_qa_pairs(original_pairs, max_reverse_pairs=2)
        
        # Verify the result
        assert len(result) == 2, "Should return 2 reversed QA pairs"
        assert result[0]["question"] == "What does the calculate_average function do?"
        assert "average" in result[0]["answer"]


@pytest.mark.asyncio
async def test_pydantic_validation():
    """Test that the Pydantic validation works correctly."""
    # Create a valid QA pair
    valid_pair = {
        "question": "What does this function do?",
        "answer": "This function calculates the average of a list of numbers."
    }
    
    # Validate it
    validated_pair = QuestionAnswerPair(**valid_pair)
    assert validated_pair.question == valid_pair["question"]
    assert validated_pair.answer == valid_pair["answer"]
    
    # Test with an invalid question (too short)
    invalid_pair = {
        "question": "What?",  # Too short
        "answer": "This function calculates the average of a list of numbers."
    }
    
    # Should raise a validation error
    with pytest.raises(ValueError):
        QuestionAnswerPair(**invalid_pair)


@pytest.mark.asyncio
async def test_fallback_to_basic_generation():
    """Test that fallback to basic generation works when LLM fails."""
    test_code = """
def calculate_average(numbers: list) -> float:
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
"""
    
    # Mock the retry_llm_call function to return None (simulating failure)
    with patch('llm_generator.retry_llm_call', new=AsyncMock(return_value=None)):
        # Call the function
        result = await generate_code_qa_pairs(test_code, max_pairs=2)
        
        # Verify that we got a result from the basic generation fallback
        assert len(result) > 0, "Should return at least one QA pair from basic generation"
        assert "calculate_average" in result[0]["question"]


@pytest.mark.asyncio
async def test_cached_retry_decorator():
    """Test that the cached_retry decorator works correctly."""
    # Try to import cached_retry (this is just to verify it's accessible)
    try:
        from llm_generator import cached_retry
    except ImportError:
        pytest.skip("cached_retry not available")
    
    # Verify that the decorator is applied to our functions
    assert hasattr(generate_code_qa_pairs, "__wrapped__"), "generate_code_qa_pairs should be decorated with cached_retry"
    assert hasattr(generate_markdown_qa_pairs, "__wrapped__"), "generate_markdown_qa_pairs should be decorated with cached_retry"
    assert hasattr(generate_reverse_qa_pairs, "__wrapped__"), "generate_reverse_qa_pairs should be decorated with cached_retry"


def test_imports_and_initialization():
    """Test that all necessary imports and initializations are present."""
    # Verify that the key components are imported
    from llm_generator import (
        call_litellm,
        CallOptions,
        init_litellm,
        retry_llm_call,
    )
    
    # These should all be callable
    assert callable(call_litellm), "call_litellm should be callable"
    assert callable(retry_llm_call), "retry_llm_call should be callable"
    assert callable(init_litellm), "init_litellm should be callable" 