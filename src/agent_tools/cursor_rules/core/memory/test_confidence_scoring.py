#!/usr/bin/env python3
"""Test module for confidence scoring functionality"""

import pytest
from loguru import logger
from typing import List, Dict, Any

from agent_tools.cursor_rules.core.memory.confidence_scoring import (
    evaluate_confidence,
    analyze_evidence,
    initialize_cache,
    DEFAULT_MODEL,
    ConfidenceEvaluation,
    EvidenceAnalysis
)

# Test data
TEST_FACT = "The Earth orbits the Sun."
TEST_EVIDENCE = [
    "Astronomical observations confirm the Earth's elliptical orbit around the Sun.",
    "NASA and other space agencies base their calculations on heliocentric orbits.",
    "Satellite data consistently shows Earth's orbital path around the Sun."
]

@pytest.mark.asyncio
async def test_confidence_evaluation():
    """Test the confidence evaluation functionality with caching"""
    initialize_cache()
    logger.info("Testing confidence evaluation with caching...")

    # First call should miss cache
    response1 = await evaluate_confidence(
        fact_content=TEST_FACT,
        evidence=TEST_EVIDENCE,
        model=DEFAULT_MODEL
    )
    logger.info(f"First evaluation confidence: {response1.confidence}")

    # Second call should hit cache
    response2 = await evaluate_confidence(
        fact_content=TEST_FACT,
        evidence=TEST_EVIDENCE,
        model=DEFAULT_MODEL
    )
    logger.info(f"Second evaluation confidence: {response2.confidence}")

    # Verify responses
    assert isinstance(response1, ConfidenceEvaluation)
    assert isinstance(response2, ConfidenceEvaluation)
    assert 0.0 <= response1.confidence <= 1.0
    assert response1.confidence == response2.confidence  # Should be identical due to caching

@pytest.mark.asyncio
async def test_evidence_analysis():
    """Test the evidence analysis functionality with caching"""
    initialize_cache()
    logger.info("Testing evidence analysis with caching...")

    # First call should miss cache
    response1 = await analyze_evidence(
        fact_content=TEST_FACT,
        evidence=TEST_EVIDENCE,
        model=DEFAULT_MODEL
    )
    logger.info(f"First analysis support score: {response1.support_score}")

    # Second call should hit cache
    response2 = await analyze_evidence(
        fact_content=TEST_FACT,
        evidence=TEST_EVIDENCE,
        model=DEFAULT_MODEL
    )
    logger.info(f"Second analysis support score: {response2.support_score}")

    # Verify responses
    assert isinstance(response1, EvidenceAnalysis)
    assert isinstance(response2, EvidenceAnalysis)
    assert 0.0 <= response1.support_score <= 1.0
    assert 0.0 <= response1.contradiction_score <= 1.0
    assert response1.support_score == response2.support_score  # Should be identical due to caching

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_confidence_evaluation())
    asyncio.run(test_evidence_analysis()) 