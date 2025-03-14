"""Test the enhanced QA generation with temperature variation and RapidFuzz validation.

This module tests the enhanced QA generation functionality to ensure:
1. Temperature variation is applied correctly
2. RapidFuzz validation enhances function-related questions
3. Duplicate QA pairs are properly filtered

Official documentation:
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
- pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

import os
import sys
import json
import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from llm_generator import (
    generate_code_qa_pairs,
    generate_markdown_qa_pairs,
    generate_reverse_qa_pairs
)
from qa_validator import (
    validate_and_enhance_qa_pairs,
    detect_duplicate_pairs
)


# Sample code for testing
SAMPLE_PYTHON_CODE = '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average
        
    Returns:
        The average as a float
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
'''

# Sample markdown for testing
SAMPLE_MARKDOWN = '''
# Sample Documentation

This is a sample markdown document for testing QA generation.

## Functions

The `calculate_average` function computes the mean of a list of numbers.

## Usage

```python
result = calculate_average([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```
'''

# Sample QA pairs for testing
SAMPLE_QA_PAIRS = [
    {
        "question": "What does the calculate_average function do?",
        "answer": "It calculates the average of a list of numbers."
    },
    {
        "question": "How do I use the calculate_average function?",
        "answer": "You can call it with a list of numbers: calculate_average([1, 2, 3, 4, 5])"
    }
]


@pytest.mark.asyncio
async def test_temperature_variation_in_code_qa_generation():
    """Test that temperature variation is applied in code QA generation."""
    # Mock the LiteLLM call to return a fixed response
    with patch('llm_generator.retry_llm_call') as mock_llm_call:
        # Set up the mock to return a valid response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "question": "What does the calculate_average function do?",
                                "answer": "It calculates the average of a list of numbers."
                            }
                        ])
                    }
                }
            ]
        }
        mock_llm_call.return_value = mock_response
        
        # Call the function with temperature=None to use random variation
        result = await generate_code_qa_pairs(
            code_content=SAMPLE_PYTHON_CODE,
            function_name="calculate_average",
            temperature=None,
            max_pairs=3
        )
        
        # Verify that the LLM was called with a temperature value
        args, kwargs = mock_llm_call.call_args
        options = args[0]
        
        # The temperature should be a float between 0.1 and 0.5 for code QA
        assert isinstance(options.temperature, float), "Temperature should be a float"
        assert 0.1 <= options.temperature <= 0.5, "Temperature should be between 0.1 and 0.5"


@pytest.mark.asyncio
async def test_temperature_variation_in_markdown_qa_generation():
    """Test that temperature variation is applied in markdown QA generation."""
    # Mock the LiteLLM call to return a fixed response
    with patch('llm_generator.retry_llm_call') as mock_llm_call:
        # Set up the mock to return a valid response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "question": "What is described in the documentation?",
                                "answer": "The documentation describes the calculate_average function."
                            }
                        ])
                    }
                }
            ]
        }
        mock_llm_call.return_value = mock_response
        
        # Call the function with temperature=None to use random variation
        result = await generate_markdown_qa_pairs(
            markdown_content=SAMPLE_MARKDOWN,
            temperature=None,
            max_pairs=3
        )
        
        # Verify that the LLM was called with a temperature value
        args, kwargs = mock_llm_call.call_args
        options = args[0]
        
        # The temperature should be a float between 0.3 and 0.7 for markdown QA
        assert isinstance(options.temperature, float), "Temperature should be a float"
        assert 0.3 <= options.temperature <= 0.7, "Temperature should be between 0.3 and 0.7"


@pytest.mark.asyncio
async def test_function_qa_enhancement():
    """Test that function-related QA pairs are enhanced with implementation details."""
    # Create a QA pair about a function
    qa_pair = {
        "question": "What does the calculate_average function do?",
        "answer": "It calculates the average of a list of numbers."
    }
    
    # Validate and enhance the QA pair
    validated_pairs = await validate_and_enhance_qa_pairs(
        qa_pairs=[qa_pair],
        original_content=SAMPLE_PYTHON_CODE,
        function_name="calculate_average"
    )
    
    # Verify that the answer was enhanced with implementation details
    assert len(validated_pairs) == 1, "Should have one validated pair"
    enhanced_answer = validated_pairs[0]["answer"]
    assert "def calculate_average" in enhanced_answer, "Answer should include function definition"
    assert "return sum(numbers) / len(numbers)" in enhanced_answer, "Answer should include function implementation"


@pytest.mark.asyncio
async def test_duplicate_qa_removal():
    """Test that duplicate QA pairs are properly filtered."""
    # Create some duplicate QA pairs
    qa_pairs = [
        {
            "question": "What does the calculate_average function do?",
            "answer": "It calculates the average of a list of numbers."
        },
        {
            "question": "What is the purpose of calculate_average?",
            "answer": "It computes the mean of a list of numbers."
        },
        {
            "question": "How do I use the calculate_average function?",
            "answer": "You can call it with a list of numbers: calculate_average([1, 2, 3, 4, 5])"
        }
    ]
    
    # Detect and remove duplicates
    unique_pairs = detect_duplicate_pairs(qa_pairs)
    
    # Verify that duplicates were removed
    assert len(unique_pairs) < len(qa_pairs), "Should have removed some duplicates"
    # The first two questions are similar, so one should be removed
    assert len(unique_pairs) == 2, "Should have 2 unique pairs"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 