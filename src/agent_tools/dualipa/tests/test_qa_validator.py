"""Test the QA validator functionality.

This module tests the QA validation utilities to ensure they:
1. Correctly identify and filter duplicate QA pairs 
2. Enhance function-related questions with complete implementations
3. Validate questions and answers for quality and completeness

Official documentation:
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
- pytest: https://docs.pytest.org/
"""

import os
import sys
import pytest
import asyncio
from typing import Dict, List, Optional

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_validator import (
    detect_duplicate_pairs,
    validate_function_qa_pair,
    validate_and_enhance_qa_pairs,
    QAValidationResult,
    MIN_SIMILARITY_THRESHOLD
)

# Import pytest-asyncio for async tests
import pytest_asyncio

# Sample code for testing function validation
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

def process_data(data, threshold=0.5):
    """Process data with a threshold filter.
    
    Args:
        data: Data to process
        threshold: Threshold value for filtering
        
    Returns:
        Processed data
    """
    result = []
    for item in data:
        if item > threshold:
            result.append(item * 2)
    return result
'''

# Sample QA pairs for tests
SAMPLE_QA_PAIRS = [
    {
        "question": "What does the calculate_average function do?",
        "answer": "It calculates the average of a list of numbers."
    },
    {
        "question": "What are the parameters of calculate_average?",
        "answer": "The function takes a single parameter 'numbers', which is a list of numbers to average."
    },
    {
        "question": "What is the purpose of the calculate_average function?",
        "answer": "The calculate_average function computes the mean of a list of numbers."
    },
    {
        "question": "How does the process_data function work?",
        "answer": "It filters data based on a threshold and doubles values that exceed the threshold."
    },
    {
        "question": "Can you explain how process_data works?",
        "answer": "process_data takes a data list and threshold, filters items above the threshold, and doubles them."
    }
]


def test_detect_duplicate_pairs():
    """Test that duplicate QA pairs are correctly identified and filtered."""
    # Arrange
    qa_pairs = SAMPLE_QA_PAIRS.copy()
    
    # Act
    unique_pairs = detect_duplicate_pairs(qa_pairs, similarity_threshold=MIN_SIMILARITY_THRESHOLD)
    
    # Assert
    assert len(unique_pairs) < len(qa_pairs), "Should remove some duplicate pairs"
    # The first and third questions are similar, and the fourth and fifth
    assert len(unique_pairs) == 3, f"Expected 3 unique pairs, got {len(unique_pairs)}"


def test_validate_function_qa_pair():
    """Test validation and enhancement of function-related QA pairs."""
    # Test with function question
    qa_pair = {
        "question": "What does the calculate_average function do?",
        "answer": "It calculates the average of a list of numbers."
    }
    
    result = validate_function_qa_pair(qa_pair, SAMPLE_PYTHON_CODE, "calculate_average")
    
    assert result.valid, "Should be valid"
    assert result.enhanced_answer is not None, "Should enhance the answer with implementation"
    assert "def calculate_average" in result.enhanced_answer, "Should include function definition"
    assert "return sum(numbers) / len(numbers)" in result.enhanced_answer, "Should include function body"
    
    # Test with implementation question
    qa_pair = {
        "question": "How is the calculate_average function implemented?",
        "answer": "It sums all numbers and divides by the count."
    }
    
    result = validate_function_qa_pair(qa_pair, SAMPLE_PYTHON_CODE)
    
    assert result.valid, "Should be valid"
    assert result.enhanced_answer is not None, "Should enhance the answer with implementation"
    
    # Test with non-function question
    qa_pair = {
        "question": "What is Python?",
        "answer": "Python is a programming language."
    }
    
    result = validate_function_qa_pair(qa_pair, SAMPLE_PYTHON_CODE)
    
    assert result.valid, "Should be valid even for non-function questions"
    assert result.enhanced_answer is None, "Should not enhance non-function questions"


@pytest.mark.asyncio
async def test_validate_and_enhance_qa_pairs():
    """Test batch validation and enhancement of QA pairs."""
    # Arrange
    qa_pairs = [
        {
            "question": "How is the calculate_average function implemented?",
            "answer": "It sums all numbers and divides by the count."
        },
        {
            "question": "What are the parameters of process_data?",
            "answer": "It takes data and an optional threshold parameter."
        },
        {
            "question": "What is the purpose of the process_data function?",
            "answer": "It processes data with a threshold filter."
        },
        # Invalid pair (too short)
        {
            "question": "What?",
            "answer": "Yes."
        }
    ]
    
    # Act
    validated_pairs = await validate_and_enhance_qa_pairs(
        qa_pairs=qa_pairs,
        original_content=SAMPLE_PYTHON_CODE,
        function_name="calculate_average"
    )
    
    # Assert
    assert len(validated_pairs) >= 2, "Should have at least 2 valid pairs"
    assert any("calculate_average" in pair["question"] for pair in validated_pairs), "Should include the calculate_average question"
    
    # Test with empty list
    empty_result = await validate_and_enhance_qa_pairs([], "")
    assert empty_result == [], "Should handle empty list"


def test_empty_and_edge_cases():
    """Test handling of empty and edge cases."""
    # Empty list
    assert detect_duplicate_pairs([]) == [], "Should handle empty list"
    assert validate_and_enhance_qa_pairs([], "") == [], "Should handle empty list"
    
    # Empty question or answer
    empty_qa = {
        "question": "",
        "answer": "Some answer"
    }
    result = validate_function_qa_pair(empty_qa, SAMPLE_PYTHON_CODE)
    assert not result.valid, "Empty question should be invalid"
    
    # Invalid function name
    qa_pair = {
        "question": "What does the non_existent_function do?",
        "answer": "It does something."
    }
    result = validate_function_qa_pair(qa_pair, SAMPLE_PYTHON_CODE, "non_existent_function")
    assert not result.valid, "Question about non-existent function should be invalid"
    assert any("function" in issue.lower() for issue in result.issues), "Should mention function issue"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 