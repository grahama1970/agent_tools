"""Test constraint tracking for QA generation.

This module tests constraint tracking for the QA generation module across
different phases of implementation.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pydantic: https://docs.pydantic.dev/latest/
"""

import pytest
from pydantic import ValidationError

from agent_tools.dualipa.qa.models.qa_models import QAPair, QAResponse, Direction
from agent_tools.dualipa.qa.models.config import QAGenerationConfig
from agent_tools.dualipa.qa.utils.validation import validate_temperature_range


def test_constraint_tracking_phase1():
    """Test constraint tracking for Phase 1.
    
    This test verifies that:
    1. Temperature in QAPair is constrained within 0.0-1.0
    2. Question always ends with ?
    3. Reasoning always contains 'Oh wait?!'
    4. Direction is one of the allowed values
    """
    # Valid temperatures
    valid_temps = [0.0, 0.3, 0.5, 0.7, 1.0]
    for temp in valid_temps:
        pair = QAPair(
            question="What does this module do?",
            answer="This module provides QA generation capabilities.",
            reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
            temperature_used=temp
        )
        assert pair.temperature_used == temp
    
    # Invalid temperatures
    invalid_temps = [-0.1, 1.1, 2.0]
    for temp in invalid_temps:
        with pytest.raises(ValidationError):
            QAPair(
                question="What does this module do?",
                answer="This module provides QA generation capabilities.",
                reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
                temperature_used=temp
            )
    
    # Test question mark constraint
    pair = QAPair(
        question="What does this module do",  # No question mark
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation."
    )
    assert pair.question.endswith("?")
    
    # Test reasoning constraint
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs."  # No Oh wait?!
    )
    assert "Oh wait?!" in pair.reasoning
    
    # Test direction constraint
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
        direction=Direction.FORWARD
    )
    assert pair.direction == Direction.FORWARD
    
    pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities.",
        reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
        direction=Direction.REVERSE
    )
    assert pair.direction == Direction.REVERSE
    
    # Invalid direction
    with pytest.raises(ValidationError):
        QAPair(
            question="What does this module do?",
            answer="This module provides QA generation capabilities.",
            reasoning="Based on the documentation, it seems to generate QA pairs. Oh wait?! It also handles validation.",
            direction="invalid"  # Invalid direction
        )


def test_constraint_tracking_phase2():
    """Test temperature constraints in Phase 2.
    
    Ensures temperature constraints are enforced throughout the pipeline,
    including configuration validation and response tracking.
    """
    # Valid temperature range in config
    valid_config = QAGenerationConfig(temperature_range=[0.1, 0.5, 0.9])
    assert valid_config.temperature_range == [0.1, 0.5, 0.9]
    
    # Invalid temperature range (out of bounds)
    with pytest.raises(ValueError):
        QAGenerationConfig(temperature_range=[-0.1, 0.5, 1.1])
    
    # Invalid temperature range (not ascending)
    with pytest.raises(ValueError):
        QAGenerationConfig(temperature_range=[0.9, 0.5, 0.1])
    
    # Test temperature validation utility
    assert validate_temperature_range([0.1, 0.3, 0.5]) == [0.1, 0.3, 0.5]
    assert validate_temperature_range([]) == [0.5]  # Default
    assert validate_temperature_range([2.0, -1.0]) == [0.5]  # Default
    
    # Test temperature range in response metadata
    response = QAResponse(
        qa_pairs=[
            QAPair(
                question="What is the temperature range?",
                answer="The list of temperatures used for generation.",
                reasoning="Temperature controls output randomness. Oh wait?! "
                         "Multiple temperatures provide output diversity.",
                temperature_used=0.3
            ),
            QAPair(
                question="Why track temperatures?",
                answer="To ensure proper constraint enforcement.",
                reasoning="Tracking ensures we can verify constraints. Oh wait?! "
                         "It also helps with debugging generation issues.",
                temperature_used=0.7
            )
        ],
        generation_metadata={
            "model_used": "test-model",
            "temperature_range": [0.3, 0.7],
            "timestamp": "2025-03-20T12:00:00Z"
        }
    )
    
    # Verify metadata contains temperature range
    assert "temperature_range" in response.generation_metadata
    assert response.generation_metadata["temperature_range"] == [0.3, 0.7]
    
    # Verify each QA pair respects the temperature range
    temps_used = {pair.temperature_used for pair in response.qa_pairs}
    for temp in temps_used:
        assert 0.0 <= temp <= 1.0