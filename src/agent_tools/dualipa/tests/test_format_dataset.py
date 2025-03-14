"""
Tests for the format_dataset module.

Official Documentation References:
- pytest: https://docs.pytest.org/
- pytest-mock: https://pytest-mock.readthedocs.io/
- json: https://docs.python.org/3/library/json.html
- tempfile: https://docs.python.org/3/library/tempfile.html
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the function to be tested
from agent_tools.dualipa.format_dataset import format_for_lora


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    data = {
        "files": [
            {
                "path": "test_file.py",
                "content": "def test_function():\n    pass\n\nclass TestClass:\n    def method(self):\n        pass"
            },
            {
                "path": "another_file.py",
                "content": "def another_function():\n    return True"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        json.dump(data, temp_file)
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


@pytest.fixture
def output_file():
    """Create a temporary output file."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def test_format_for_lora_creates_output(sample_input_data, output_file):
    """Test that format_for_lora creates an output file with the correct structure."""
    # Call the function
    format_for_lora(sample_input_data, output_file)
    
    # Check that the output file exists
    assert os.path.exists(output_file)
    
    # Check the content of the output file
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    # Verify structure
    assert "qa_pairs" in output_data
    assert isinstance(output_data["qa_pairs"], list)


def test_format_for_lora_qa_pairs_content(sample_input_data, output_file):
    """Test that format_for_lora generates appropriate Q&A pairs."""
    # Call the function
    format_for_lora(sample_input_data, output_file)
    
    # Read the output
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    qa_pairs = output_data["qa_pairs"]
    
    # Check we have at least some pairs (exact number depends on whether method_validator is available)
    assert len(qa_pairs) > 0
    
    # Check each pair has question and answer fields
    for pair in qa_pairs:
        assert "question" in pair
        assert "answer" in pair
        
        # Questions should contain function or class name
        assert any(keyword in pair["question"] for keyword in ["test_function", "TestClass", "method", "another_function"])


def test_format_for_lora_identifies_definitions(sample_input_data, output_file):
    """Test that format_for_lora correctly identifies function and class definitions."""
    # Call the function
    format_for_lora(sample_input_data, output_file)
    
    # Read the output
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    # Get all questions
    questions = [pair["question"] for pair in output_data["qa_pairs"]]
    
    # Check that questions about our test functions/classes exist
    assert any("test_function" in q for q in questions)
    # Either the class or its method should be in the questions
    assert any("TestClass" in q for q in questions) or any("method" in q for q in questions)
    assert any("another_function" in q for q in questions)


@patch("agent_tools.dualipa.format_dataset.METHOD_VALIDATOR_AVAILABLE", True)
@patch("agent_tools.dualipa.format_dataset.generate_enhanced_qa_pairs")
def test_format_for_lora_with_method_validator(mock_generate_enhanced_qa_pairs, sample_input_data, output_file):
    """Test that format_for_lora uses method_validator when available."""
    # Setup mock
    mock_qa_pairs = [
        {"question": "What does test_function do?", "answer": "It does testing."},
        {"question": "What are the parameters of test_function?", "answer": "None"}
    ]
    mock_generate_enhanced_qa_pairs.return_value = mock_qa_pairs
    
    # Call the function
    format_for_lora(sample_input_data, output_file)
    
    # Verify enhanced QA generation was called
    mock_generate_enhanced_qa_pairs.assert_called_once()
    
    # Read the output
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    # Check the enhanced QA pairs were used
    assert output_data["qa_pairs"] == mock_qa_pairs


@patch("agent_tools.dualipa.format_dataset.METHOD_VALIDATOR_AVAILABLE", False)
def test_format_for_lora_without_method_validator(sample_input_data, output_file):
    """Test that format_for_lora falls back to basic processing when method_validator is not available."""
    # Call the function
    format_for_lora(sample_input_data, output_file)
    
    # Read the output
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    # Check we have the expected number of pairs (3 functions/classes in our test data)
    assert len(output_data["qa_pairs"]) == 3
    
    # Verify the format of basic QA pairs
    for pair in output_data["qa_pairs"]:
        assert pair["question"].startswith("What does `")
        assert pair["question"].endswith("` do?")
        assert isinstance(pair["answer"], str)
        assert len(pair["answer"]) > 0


def test_format_for_lora_with_missing_input_file():
    """Test that format_for_lora raises FileNotFoundError for missing input file."""
    with pytest.raises(FileNotFoundError):
        format_for_lora("nonexistent_file.json", "output.json")


def test_format_for_lora_with_invalid_json(output_file):
    """Test that format_for_lora handles invalid JSON input."""
    # Create an invalid JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_file.write(b"{invalid json")
        invalid_json_path = temp_file.name
    
    try:
        with pytest.raises(ValueError):
            format_for_lora(invalid_json_path, output_file)
    finally:
        # Cleanup
        if os.path.exists(invalid_json_path):
            os.remove(invalid_json_path) 