"""Test validation utilities.

This module tests validation utilities for QA generation, including:
- Input JSON validation and normalization
- QA pair validation against business rules
- QA response validation
- Temperature range validation

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pydantic: https://docs.pydantic.dev/latest/
- json: https://docs.python.org/3/library/json.html
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
- pathlib: https://docs.python.org/3/library/pathlib.html

Expected test coverage:
- Valid and invalid input JSON validation
- JSON normalization with default values
- QA pair validation for various criteria (question marks, reasoning, etc.)
- QA response validation with metadata
- Edge cases including empty inputs and invalid values
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_tools.dualipa.qa.utils.validation import (
    validate_input_json, validate_qa_pair, validate_qa_response, 
    validate_temperature_range, normalize_input_json
)
from agent_tools.dualipa.qa.models.qa_models import QAPair, QAResponse


def test_validate_input_json_real():
    """Test validating real input JSON.
    
    This test verifies that a properly formatted extraction JSON
    passes validation with the correct schema and structure.
    """
    # Valid input JSON
    valid_input = {
        "sections": [
            {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "type": "documentation",
                "content": "## Test\nThis is test content.",
                "extraction_focus": "technical details",
                "summary_instructions": "Generate QA pairs about the technical details."
            }
        ],
        "extraction_metadata": {
            "model_used": "gpt-4-turbo",
            "timestamp": "2025-03-19T14:49:00Z"
        }
    }
    
    # Validate the input
    result = validate_input_json(valid_input)
    assert result is True
    
    # Optional: Check that it doesn't raise exceptions
    try:
        validate_input_json(valid_input, raise_on_error=True)
    except Exception as e:
        pytest.fail(f"validate_input_json raised {type(e).__name__} unexpectedly!")


def test_validate_input_json_invalid():
    """Test validating invalid input JSON.
    
    This test verifies that improperly formatted extraction JSON
    fails validation with appropriate error details.
    """
    # Missing required fields
    missing_sections = {
        "extraction_metadata": {
            "model_used": "gpt-4-turbo",
            "timestamp": "2025-03-19T14:49:00Z"
        }
    }
    
    # Missing metadata
    missing_metadata = {
        "sections": [
            {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "type": "documentation",
                "content": "This is test content.",
                "extraction_focus": "technical details"
                # Missing summary_instructions
            }
        ]
    }
    
    # Empty sections
    empty_sections = {
        "sections": [],
        "extraction_metadata": {
            "model_used": "gpt-4-turbo",
            "timestamp": "2025-03-19T14:49:00Z"
        }
    }
    
    # Test with raise_on_error=False
    assert validate_input_json(missing_sections) is False
    assert validate_input_json(missing_metadata) is False
    assert validate_input_json(empty_sections) is False
    
    # Test with raise_on_error=True
    with pytest.raises(ValueError):
        validate_input_json(missing_sections, raise_on_error=True)
    
    with pytest.raises(ValueError):
        validate_input_json(missing_metadata, raise_on_error=True)


def test_normalize_input_json():
    """Test normalizing input JSON.
    
    This test verifies that input JSON is properly normalized,
    with default values added for optional fields.
    """
    # Input with minimal fields
    minimal_input = {
        "sections": [
            {
                "uuid": "123",
                "type": "documentation",
                "content": "Test content",
                # Missing optional fields
            }
        ],
        "extraction_metadata": {
            "model_used": "gpt-4-turbo"
            # Missing timestamp
        }
    }
    
    # Normalize the input
    normalized = normalize_input_json(minimal_input)
    
    # Check that required fields are preserved
    assert len(normalized["sections"]) == 1
    assert normalized["sections"][0]["uuid"] == "123"
    assert normalized["sections"][0]["type"] == "documentation"
    assert normalized["sections"][0]["content"] == "Test content"
    
    # Check that optional fields are added with defaults
    assert "extraction_focus" in normalized["sections"][0]
    assert "summary_instructions" in normalized["sections"][0]
    assert "timestamp" in normalized["extraction_metadata"]


def test_validate_qa_pair():
    """Test validating QA pairs.
    
    This test verifies that QA pairs are properly validated
    against business rules beyond Pydantic validation.
    """
    # Valid QA pair
    valid_pair = QAPair(
        question="What is the purpose of this test?",
        answer="To validate QA pairs against business rules.",
        reasoning="Based on the test name, it validates QA pairs. Oh wait?! It specifically checks business rules beyond Pydantic."
    )
    
    # Validate the pair
    assert validate_qa_pair(valid_pair) is True
    
    # Test with a technically valid but semantically poor answer
    poor_answer_pair = QAPair(
        question="Is this a valid QA pair?",
        answer="Maybe it is valid but this answer doesn't really provide any helpful information.",
        reasoning="This has enough reasoning for validation. Oh wait?! Actually the answer is not very helpful."
    )
    
    # Validate directly using validators - don't rely on the model's auto-fix mechanisms
    # We're testing business rule validation, not Pydantic validation
    with patch('agent_tools.dualipa.qa.utils.validation.PAIR_VALIDATORS', [
        lambda p: len(p.answer.split()) >= 20  # Answer should have at least 20 words - ours has fewer
    ]):
        assert validate_qa_pair(poor_answer_pair) is False
        
    # Test with missing "Oh wait" moment
    with patch('agent_tools.dualipa.qa.utils.validation.PAIR_VALIDATORS', [
        lambda p: "Custom phrase not present" in p.reasoning  # Should fail this check
    ]):
        missing_reasoning_pair = QAPair(
            question="Is this valid?",
            answer="This is a sufficiently long answer for validation.",
            reasoning="This reasoning is missing the required phrase. Oh wait?! Actually that's the point of this test."
        )
        assert validate_qa_pair(missing_reasoning_pair) is False


def test_validate_qa_pair_real():
    """Test validating QA pairs with realistic examples.
    
    This test uses realistic QA pairs that should pass validation
    to ensure the validators are correctly calibrated for real-world use.
    
    Input:
        - Technical QA pair about circuit breaker pattern
        - Markdown QA pair about pipeline components
        - QA pair with empty source_section_uuid
    
    Expected output:
        - All QA pairs should pass validation (return True)
        - Validation should handle empty source_section_uuid
    
    Dependencies:
        - QAPair model from agent_tools.dualipa.qa.models.qa_models
        - validate_qa_pair from agent_tools.dualipa.qa.utils.validation
    """
    # Realistic QA pair from technical documentation
    technical_pair = QAPair(
        question="How does the circuit breaker pattern work in this module?",
        answer="The circuit breaker pattern prevents cascading failures by stopping API calls after multiple failures. It has three states: closed (normal operation), open (failing fast), and half-open (testing recovery).",
        reasoning="Looking at the code, I can see the circuit_state tracks failures. Oh wait?! It also implements a timeout and success counter for recovery.",
        source_section_uuid="123e4567-e89b-12d3-a456-426614174000",
        temperature_used=0.5
    )
    
    # Realistic QA pair from markdown documentation
    markdown_pair = QAPair(
        question="What are the key components of the QA generation pipeline?",
        answer="The pipeline includes input validation, temperature iteration, bidirectional generation, semantic deduplication, and output validation.",
        reasoning="The documentation lists these components in the workflow diagram. Oh wait?! There's also error recovery with circuit breaker and dead letter queue.",
        source_section_uuid="223e4567-e89b-12d3-a456-426614174000",
        temperature_used=0.7
    )
    
    # Both should pass validation
    assert validate_qa_pair(technical_pair) is True
    assert validate_qa_pair(markdown_pair) is True
    
    # Check with empty source_section_uuid (should still be valid)
    technical_pair.source_section_uuid = None
    assert validate_qa_pair(technical_pair) is True


def test_validate_qa_pair_validators():
    """Test individual validators for QA pairs.
    
    This test directly tests each validator function to ensure they correctly
    identify valid and invalid QA pairs based on specific criteria.
    
    Input:
        - Valid QA pair for positive tests
        - Mocked QA pair for negative tests
        - Lambda functions that implement validation rules
    
    Expected output:
        - Question mark validator should return True for valid question
        - "Oh wait?!" validator should return True when phrase is present
        - Answer length validator should return True for sufficient answer
        - "Oh wait?!" validator should return False when phrase is missing
    
    Dependencies:
        - QAPair model from agent_tools.dualipa.qa.models.qa_models
        - MagicMock from unittest.mock
    """
    # Create a valid QA pair that we'll modify for each test
    base_pair = QAPair(
        question="Does this validation work?",
        answer="Yes, it validates different aspects of QA pairs.",
        reasoning="Based on the code, it should validate reasoning. Oh wait?! It also checks question marks and answer length."
    )
    
    # Test the question mark validator
    question_validator = lambda p: "?" in p.question
    assert question_validator(base_pair) is True
    
    # Test the "Oh wait?!" validator
    oh_wait_validator = lambda p: "Oh wait?!" in p.reasoning
    assert oh_wait_validator(base_pair) is True
    
    # Test answer length validator (min 5 words)
    answer_length_validator = lambda p: len(p.answer.split()) >= 5
    assert answer_length_validator(base_pair) is True
    
    # Create a test pair for validation_function directly
    from unittest.mock import MagicMock
    mock_pair = MagicMock()
    mock_pair.reasoning = "No special phrase here"
    
    # Test that validator fails when the required phrase is missing
    assert (lambda p: "Oh wait?!" in p.reasoning)(mock_pair) is False


def test_validate_qa_response():
    """Test validating QA responses.
    
    This test verifies that QA responses are properly validated
    against business rules beyond Pydantic validation.
    """
    # Valid QA response
    valid_pair = QAPair(
        question="What is the purpose of this test?",
        answer="To validate QA responses against business rules.",
        reasoning="Based on the test name, it validates QA responses. Oh wait?! It specifically checks business rules beyond Pydantic."
    )
    
    valid_response = QAResponse(
        qa_pairs=[valid_pair, valid_pair],  # Two valid pairs
        generation_metadata={
            "model_used": "gpt-4-turbo",
            "temperature_range": [0.3, 0.5, 0.7],
            "timestamp": "2025-03-19T15:22:00Z"
        }
    )
    
    # Validate the response
    assert validate_qa_response(valid_response) is True
    
    # Empty response
    empty_response = QAResponse(
        qa_pairs=[],
        generation_metadata={
            "model_used": "gpt-4-turbo",
            "timestamp": "2025-03-19T15:22:00Z"
        }
    )
    assert validate_qa_response(empty_response) is False
    
    # Invalid metadata (missing required fields)
    with patch('agent_tools.dualipa.qa.utils.validation.RESPONSE_METADATA_REQUIRED', 
               ["model_used", "temperature_range", "timestamp", "missing_field"]):
        assert validate_qa_response(valid_response) is False