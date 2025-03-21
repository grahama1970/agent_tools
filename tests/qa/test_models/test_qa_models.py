"""Test QA models.

This module tests the QA model validation and constraints, ensuring that
the Pydantic models correctly enforce data validation rules. Tests cover
both valid data that should be accepted and invalid data that should be
rejected with appropriate validation errors.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pydantic: https://docs.pydantic.dev/latest/
- uuid: https://docs.python.org/3/library/uuid.html

Expected test coverage:
- QAPair creation with valid data
- QAPair validation failures for invalid data
- QAResponse creation with valid data
- QAResponse validation failures for invalid data
- Enum validation for direction field
- Validation for numeric fields (temperature, confidence)
"""

import pytest
from pydantic import ValidationError

from agent_tools.dualipa.qa.models.qa_models import QAPair, QAResponse


def test_qapair_model_validation():
    """Test that a valid QA pair passes validation."""
    # Create a valid QA pair
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
    )
    
    # Assert field values
    assert pair.question == "What does this module do?"
    assert pair.answer == "This module provides QA generation capabilities."
    assert "Oh wait?!" in pair.reasoning
    assert pair.direction == "forward"  # Default value
    assert 0.0 <= pair.temperature_used <= 1.0
    assert pair.uuid is not None


def test_qapair_model_validation_question_mark():
    """Test that question ends with a question mark."""
    # Create a QA pair with a question that doesn't end with ?
    pair = QAPair(
        question="What does this module do",  # Missing ?
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
    )
    
    # Validator should add question mark
    assert pair.question.endswith("?")


def test_qapair_model_validation_oh_wait():
    """Test that reasoning contains 'Oh wait?!'."""
    # Create a QA pair with reasoning that doesn't contain 'Oh wait?!'
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs."
    )
    
    # Validator should add 'Oh wait?!'
    assert "Oh wait?!" in pair.reasoning


def test_qapair_model_failure():
    """Test that invalid QA pairs fail validation."""
    # Question too short
    with pytest.raises(ValidationError):
        QAPair(
            question="What?",
            answer="This module provides QA generation capabilities.",
            reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
        )
    
    # Answer too short
    with pytest.raises(ValidationError):
        QAPair(
            question="What does this module do?",
            answer="QA.",
            reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
        )
    
    # Reasoning too short
    with pytest.raises(ValidationError):
        QAPair(
            question="What does this module do?",
            answer="This module provides QA generation capabilities.",
            reasoning="Oh wait?!"
        )
    
    # Invalid direction
    with pytest.raises(ValidationError):
        QAPair(
            question="What does this module do?",
            answer="This module provides QA generation capabilities.",
            reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
            direction="sideways"  # Invalid direction
        )


def test_qaresponse_model_validation():
    """Test that a valid QA response passes validation."""
    # Create a QA pair
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
    )
    
    # Create a QA response
    response = QAResponse(
        qa_pairs=[pair],
        generation_metadata={
            "model_used": "gpt-4-turbo",
            "temperature_range": [0.3, 0.5, 0.7],
            "timestamp": "2025-03-19T15:22:00Z"
        }
    )
    
    # Assert field values
    assert len(response.qa_pairs) == 1
    assert response.generation_metadata["model_used"] == "gpt-4-turbo"
    assert response.generation_metadata["temperature_range"] == [0.3, 0.5, 0.7]