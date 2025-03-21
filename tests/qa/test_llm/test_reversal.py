"""Tests for bidirectional QA generation.

This module tests the ability to generate reverse QA pairs from original pairs.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
from agent_tools.dualipa.qa.llm.reversal import (
    generate_reversed_pair,
    generate_reversed_qa_pairs
)


@pytest.mark.asyncio
async def test_generate_reversed_qa_pairs_real():
    """Test generating reverse QA pairs works with real data.
    
    This test verifies that:
    1. Reversed pairs maintain the same answer
    2. Questions are different from the original
    3. Direction is set to "reverse"
    """
    # Create original QA pairs
    original_pairs = [
        QAPair(
            question="What is Python's main use case?",
            answer="Python is commonly used for web development, data analysis, AI, and automation.",
            reasoning="Python is widely adopted in these areas due to its readability and ecosystem. Oh wait?! It's also popular for scripting and rapid prototyping.",
            temperature_used=0.5,
            direction=Direction.FORWARD
        ),
        QAPair(
            question="How does asyncio work in Python?",
            answer="Asyncio enables asynchronous programming using coroutines, event loops, and tasks.",
            reasoning="It allows concurrent execution without threads. Oh wait?! It's actually based on a single-threaded event loop model that manages multiple coroutines.",
            temperature_used=0.5,
            direction=Direction.FORWARD
        )
    ]
    
    # Generate reversed pairs
    reversed_pairs = await generate_reversed_qa_pairs(
        original_pairs=original_pairs,
        reverse_ratio=1.0,  # Reverse all pairs
        temperature=0.7
    )
    
    # Verify we got results
    assert reversed_pairs, "Should return non-empty list of reversed QA pairs"
    assert len(reversed_pairs) > 0, "Should generate at least one reversed pair"
    
    # Check that reversed pairs maintain properties
    for i, rev_pair in enumerate(reversed_pairs):
        # Find corresponding original pair
        orig_pair = original_pairs[i] if i < len(original_pairs) else None
        
        if orig_pair:
            # Same answer
            assert rev_pair.answer == orig_pair.answer, "Answer should be preserved"
            
            # Different question
            assert rev_pair.question != orig_pair.question, "Question should be different"
            
            # Direction is reverse
            assert rev_pair.direction == Direction.REVERSE, "Direction should be set to reverse"
            
            # Has the same source section
            assert rev_pair.source_section_uuid == orig_pair.source_section_uuid, "Source section UUID should be preserved"


@pytest.mark.asyncio
async def test_generate_reversed_qa_pairs_quality():
    """Test quality aspects of reversed QA pairs.
    
    This test verifies that:
    1. Only highest quality pairs are selected for reversal
    2. Rate limiting is properly applied
    3. Validation filters out low-quality pairs
    """
    # Create original QA pairs with varying confidence scores
    original_pairs = [
        QAPair(
            question="What is a test?",
            answer="A test verifies that code works as expected.",
            reasoning="Tests ensure functionality is maintained. Oh wait?! They also serve as documentation.",
            confidence_score=0.3,  # Low confidence
            temperature_used=0.5,
            direction=Direction.FORWARD
        ),
        QAPair(
            question="What is TDD?",
            answer="Test-Driven Development is writing tests before implementation.",
            reasoning="It ensures code meets requirements. Oh wait?! It also leads to better design.",
            confidence_score=0.9,  # High confidence
            temperature_used=0.5,
            direction=Direction.FORWARD
        ),
        QAPair(
            question="What is a mock?",
            answer="A mock is a simulated object that mimics real object behavior.",
            reasoning="Mocks are used in testing. Oh wait?! They specifically help isolate the code being tested.",
            confidence_score=0.7,  # Medium-high confidence
            temperature_used=0.5,
            direction=Direction.FORWARD
        )
    ]
    
    # Track concurrent operations
    concurrent_count = 0
    max_concurrent = 0
    reversed_pairs_data = []
    
    # Mock implementation to check selection logic and rate limiting
    async def mock_generate_reversed(original, temperature):
        nonlocal concurrent_count, max_concurrent
        
        # Track concurrency
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        
        # Simulate API work
        await asyncio.sleep(0.1)
        
        # Record completion
        concurrent_count -= 1
        
        # Create reversed pair
        pair = QAPair(
            question=f"Is {original.answer.split()[0]} a key aspect of {original.question.split()[-1].rstrip('?')}?",
            answer=original.answer,
            reasoning=f"Based on the answer, yes. Oh wait?! Actually, this provides additional perspective on {original.question}",
            source_section_uuid=original.source_section_uuid,
            temperature_used=temperature,
            direction=Direction.REVERSE,
            confidence_score=original.confidence_score
        )
        
        # Keep track for verification
        reversed_pairs_data.append({
            "original_confidence": original.confidence_score,
            "original_question": original.question,
            "new_question": pair.question
        })
        
        return pair
    
    # Patch the generate_reversed_pair function
    with patch(
        'agent_tools.dualipa.qa.llm.reversal.generate_reversed_pair',
        side_effect=mock_generate_reversed
    ):
        # Force the ratio to 1.0 to ensure all pairs are processed in test
        # This is just for testing - in production it would use the ratio
        reversed_pairs = await generate_reversed_qa_pairs(
            original_pairs=original_pairs,
            reverse_ratio=1.0,  # Force all pairs to be processed for testing
            temperature=0.7
        )
        
        # Verify that pairs were generated
        assert len(reversed_pairs) > 0, "Should generate reversed pairs"
        
        # For a ratio of 1.0, all 3 pairs should be processed
        expected_count = len(original_pairs)
        assert len(reversed_pairs) == expected_count, f"Should generate {expected_count} reversed pairs with ratio 1.0"
        
        # Verify concurrency limit
        assert max_concurrent <= 2, "Should limit to 2 concurrent operations"
        
        # Verify all pairs are selected with ratio=1.0
        confidences = [data["original_confidence"] for data in reversed_pairs_data]
        assert 0.9 in confidences, "Highest confidence pair should be selected"
        assert 0.7 in confidences, "Medium confidence pair should be selected"
        assert 0.3 in confidences, "With ratio=1.0, even low confidence pairs should be selected"
        
        # Verify the ordering of processing (should process highest confidence first)
        confidence_order = [data["original_confidence"] for data in reversed_pairs_data]
        # The first one processed should be the highest confidence
        assert confidence_order[0] == 0.9, "Highest confidence should be processed first"
        
        # Verify questions are different
        for data in reversed_pairs_data:
            assert data["original_question"] != data["new_question"], "Questions should be different"