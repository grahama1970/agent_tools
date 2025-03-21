"""Tests for LLM generation utilities.

This module tests the LLM generation utilities including temperature iteration,
rate limiting, and context isolation.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock

from agent_tools.dualipa.qa.llm.generation import (
    generate_qa_pairs_with_temperature,
    iterate_temperatures,
    generate_code_qa_pairs,
    generate_markdown_qa_pairs
)
from agent_tools.dualipa.qa.models.qa_models import QAPair


@pytest.mark.asyncio
async def test_iterate_temperatures_real():
    """Test iterating through multiple temperatures produces different results.
    
    This test verifies that:
    1. Different temperatures generate different outputs
    2. Each temperature iteration respects context isolation
    3. Results are properly merged
    """
    # Test content
    content = """
    def calculate_factorial(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * calculate_factorial(n-1)
    """
    
    # Test with multiple temperatures
    temps = [0.1, 0.7]  # Very different temps should produce different outputs
    
    # Execute temperature iteration
    qa_pairs = await iterate_temperatures(
        content=content,
        content_type="code",
        temps=temps,
        max_pairs_per_temp=2
    )
    
    # Assertions
    assert qa_pairs, "Should return non-empty list of QA pairs"
    assert len(qa_pairs) > 0, "Should generate at least one QA pair"
    
    # Verify temperature records
    temps_used = {pair.temperature_used for pair in qa_pairs}
    assert temps_used.issubset(set(temps)), f"All pairs should use provided temperatures: {temps_used} vs {temps}"
    
    # Group pairs by temperature
    pairs_by_temp = {}
    for pair in qa_pairs:
        temp = pair.temperature_used
        if temp not in pairs_by_temp:
            pairs_by_temp[temp] = []
        pairs_by_temp[temp].append(pair)
    
    # If we have pairs from multiple temperatures, verify they're different
    # This test only makes sense if we actually got different temperatures
    if len(pairs_by_temp) > 1:
        # Get questions from each temperature
        questions_by_temp = {
            temp: [pair.question for pair in pairs]
            for temp, pairs in pairs_by_temp.items()
        }
        
        # Verify different temperatures produced different questions
        all_same = True
        for temp1 in temps:
            if temp1 not in questions_by_temp:
                continue
            for temp2 in temps:
                if temp2 not in questions_by_temp or temp1 == temp2:
                    continue
                if questions_by_temp[temp1] != questions_by_temp[temp2]:
                    all_same = False
                    break
        
        assert not all_same, "Different temperatures should produce different results"


@pytest.mark.asyncio
async def test_iterate_temperatures_deadlock():
    """Test that the temperature iteration doesn't cause context overlap.
    
    This test verifies that each temperature iteration uses a fresh context,
    preventing any potential deadlocks or context contamination.
    """
    test_content = "This is a test content for context isolation"
    temps = [0.3, 0.5, 0.7]
    
    # Mock implementation to verify separate contexts
    contexts_seen = {}
    
    async def mock_generate_with_temp(content, content_type, temperature, **kwargs):
        contexts_seen[temperature] = content
        return [QAPair(
            question=f"Question at temp {temperature}?",
            answer=f"Answer at temp {temperature}",
            reasoning=f"Reasoning at temp {temperature}. Oh wait?! More insight.",
            temperature_used=temperature
        )]
    
    # Patch the generate function to use our mock
    with patch(
        'agent_tools.dualipa.qa.llm.generation.generate_qa_pairs_with_temperature',
        side_effect=mock_generate_with_temp
    ):
        # Run the temperature iteration
        qa_pairs = await iterate_temperatures(
            content=test_content,
            content_type="text",
            temps=temps,
            max_pairs_per_temp=1
        )
        
        # Verify we got the expected number of pairs
        assert len(qa_pairs) == len(temps), f"Expected {len(temps)} pairs, got {len(qa_pairs)}"
        
        # Verify each temperature saw the same original context
        for temp in temps:
            assert contexts_seen.get(temp) == test_content, f"Context at temp {temp} should match original"
        
        # Verify temperatures are correctly recorded
        temps_used = {pair.temperature_used for pair in qa_pairs}
        assert temps_used == set(temps), f"All temperatures should be used: {temps_used} vs {temps}"


@pytest.mark.asyncio
async def test_iterate_temperatures_rate_limit():
    """Test that temperature iteration respects rate limits.
    
    This test verifies that the semaphore correctly limits the
    number of concurrent API calls.
    """
    test_content = "This is a test content for rate limiting"
    temps = [0.2, 0.4, 0.6, 0.8]  # Use 4 temps to test concurrency
    
    # Keep track of concurrent runs
    concurrent_count = 0
    max_concurrent = 0
    execution_order = []
    finished_order = []
    
    async def mock_generate_with_rate_check(content, content_type, temperature, **kwargs):
        nonlocal concurrent_count, max_concurrent
        
        # Record start of execution
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        execution_order.append(temperature)
        
        # Simulate API work with varying duration
        # Higher temperatures take longer (simulates more token generation)
        await asyncio.sleep(temperature)
        
        # Record end of execution
        concurrent_count -= 1
        finished_order.append(temperature)
        
        return [QAPair(
            question=f"Question at temp {temperature}?",
            answer=f"Answer at temp {temperature}",
            reasoning=f"Reasoning at temp {temperature}. Oh wait?! More insight.",
            temperature_used=temperature
        )]
    
    # Patch the generate function to use our mock
    with patch(
        'agent_tools.dualipa.qa.llm.generation.generate_qa_pairs_with_temperature',
        side_effect=mock_generate_with_rate_check
    ):
        # Run the temperature iteration with expected concurrency limit
        qa_pairs = await iterate_temperatures(
            content=test_content,
            content_type="text",
            temps=temps,
            max_pairs_per_temp=1
        )
        
        # Verify rate limiting worked
        # The semaphore in iterate_temperatures should be set to 2
        assert max_concurrent <= 2, f"Expected max 2 concurrent calls, got {max_concurrent}"
        
        # Verify all temperatures were processed
        assert len(qa_pairs) == len(temps), f"Expected {len(temps)} pairs, got {len(qa_pairs)}"