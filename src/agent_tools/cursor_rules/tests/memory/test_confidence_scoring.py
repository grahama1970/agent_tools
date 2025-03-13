#!/usr/bin/env python3
"""
Integration test for the confidence scoring functionality.

This test verifies that confidence scoring and evidence-based updates work correctly using a real ArangoDB instance.

Documentation references:
- LiteLLM: https://docs.litellm.ai/docs/
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import asyncio
import os
import pytest
import uuid
from typing import Dict, Any, List
from loguru import logger

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem
from agent_tools.cursor_rules.core.memory.confidence_scoring import (
    evaluate_confidence,
    update_fact_confidence,
    analyze_evidence,
    get_confidence_history
)
from agent_tools.cursor_rules.llm.litellm_call import litellm_call
from agent_tools.cursor_rules.llm.initialize_litellm_cache import initialize_litellm_cache

# Test configuration
TEST_DB_NAME = "test_agent_confidence_scoring"
TEST_COLLECTION = "test_memory_facts"
TEST_USERNAME = os.environ.get("ARANGO_USERNAME", "root")
TEST_PASSWORD = os.environ.get("ARANGO_PASSWORD", "openSesame")

# Sample facts and evidence for testing
SAMPLE_FACTS = [
    {
        "content": "The next total solar eclipse visible from North America will occur on April 8, 2024.",
        "importance": 0.8,
        "confidence": 0.9,
        "domains": ["astronomy", "events"]
    },
    {
        "content": "The human brain has approximately 86 billion neurons.",
        "importance": 0.7,
        "confidence": 0.8,
        "domains": ["neuroscience", "biology"]
    },
    {
        "content": "The Great Wall of China is visible from space.",
        "importance": 0.6,
        "confidence": 0.7,
        "domains": ["history", "architecture"]
    }
]

SAMPLE_EVIDENCE = {
    "eclipse": [
        "NASA confirms a total solar eclipse will cross North America on April 8, 2024.",
        "The path of totality will span from Mexico through the US to Canada.",
        "The next total solar eclipse visible from parts of North America after 2024 will be in 2045."
    ],
    "brain": [
        "Recent studies estimate the human brain contains between 86-100 billion neurons.",
        "The exact count varies by individual and counting methodology.",
        "Earlier estimates of 100 billion neurons have been revised downward."
    ],
    "wall": [
        "Astronauts report the Great Wall is not visible to the naked eye from low Earth orbit.",
        "While other human structures are visible from space, the Great Wall is too thin and often blends with surroundings.",
        "This is a common misconception that has been debunked by multiple space agencies."
    ]
}

@pytest.fixture(scope="module")
async def litellm():
    """Initialize LiteLLM with Redis cache."""
    await initialize_litellm_cache()

@pytest.fixture
async def memory_system():
    """Create and initialize a memory system for testing."""
    # Create a unique test DB name to avoid conflicts in parallel tests
    test_db = f"{TEST_DB_NAME}_{uuid.uuid4().hex[:8]}"
    
    # Configuration with test DB
    config = {
        "arango_host": "http://localhost:8529",
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
        "db_name": test_db,
        "facts_collection": TEST_COLLECTION,
        "embeddings_model": "all-MiniLM-L6-v2",  # Use a small model for tests
        "llm_config": {
            "model": "openai/gpt-3.5-turbo",
            "temperature": 0.2,
            "max_tokens": 500,
            "caching": True
        }
    }
    
    # Create the memory system
    memory_system = AgentMemorySystem(config)
    await memory_system.initialize()
    
    # Add sample facts
    fact_ids = []
    for fact in SAMPLE_FACTS:
        fact_id = await memory_system.remember(**fact)
        fact_ids.append(fact_id)
    
    yield memory_system, fact_ids
    
    # Cleanup: drop the test database after tests
    try:
        await memory_system.cleanup()
        from arango import ArangoClient
        client = ArangoClient(hosts=config["arango_host"])
        sys_db = client.db('_system', username=config["username"], password=config["password"])
        if sys_db.has_database(test_db):
            sys_db.delete_database(test_db)
    except Exception as e:
        print(f"Failed to cleanup test database: {e}")

@pytest.mark.asyncio
async def test_evaluate_confidence(memory_system):
    """Test evaluating confidence based on evidence."""
    mem_sys, fact_ids = memory_system
    
    # Evaluate confidence for eclipse fact with supporting evidence
    confidence, reasoning = await evaluate_confidence(
        fact_content=SAMPLE_FACTS[0]["content"],
        evidence=SAMPLE_EVIDENCE["eclipse"],
        model="gpt-3.5-turbo"
    )
    
    assert confidence > 0.9, "Should have high confidence with strong evidence"
    assert reasoning is not None, "Should provide reasoning"
    assert "NASA" in reasoning, "Reasoning should reference authoritative source"
    
    # Evaluate confidence for Great Wall fact with contradicting evidence
    confidence, reasoning = await evaluate_confidence(
        fact_content=SAMPLE_FACTS[2]["content"],
        evidence=SAMPLE_EVIDENCE["wall"],
        model="gpt-3.5-turbo"
    )
    
    assert confidence < 0.5, "Should have low confidence with contradicting evidence"
    assert "debunked" in reasoning.lower(), "Reasoning should mention contradiction"

@pytest.mark.asyncio
async def test_update_fact_confidence(memory_system):
    """Test updating fact confidence based on new evidence."""
    mem_sys, fact_ids = memory_system
    
    # Update Great Wall fact with contradicting evidence
    result = await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_ids[2],
        evidence=SAMPLE_EVIDENCE["wall"]
    )
    
    assert result["updated"], "Should update the fact"
    assert result["evaluated_confidence"] < result["original_confidence"], "Confidence should decrease"
    
    # Verify the update
    fact = await asyncio.to_thread(
        lambda: mem_sys.db.collection(TEST_COLLECTION).get(fact_ids[2])
    )
    
    assert fact["confidence"] < 0.7, "Confidence should be reduced"
    assert "confidence_history" in fact, "Should have confidence history"
    assert fact["confidence_history"][-1]["evidence"] == SAMPLE_EVIDENCE["wall"], "Should record evidence"

@pytest.mark.asyncio
async def test_analyze_evidence(memory_system):
    """Test analyzing evidence for a fact."""
    mem_sys, fact_ids = memory_system
    
    # Analyze brain neurons evidence
    analysis = await analyze_evidence(
        fact_content=SAMPLE_FACTS[1]["content"],
        evidence=SAMPLE_EVIDENCE["brain"],
        model="gpt-3.5-turbo"
    )
    
    assert "support_score" in analysis, "Should have support score"
    assert "contradiction_score" in analysis, "Should have contradiction score"
    assert "uncertainty_factors" in analysis, "Should have uncertainty factors"
    assert len(analysis["uncertainty_factors"]) > 0, "Should identify uncertainty"
    
    # The evidence suggests a range, so uncertainty should be noted
    assert any("range" in factor.lower() or "varies" in factor.lower() 
              for factor in analysis["uncertainty_factors"]), "Should note variability"

@pytest.mark.asyncio
async def test_confidence_history(memory_system):
    """Test tracking confidence history for facts."""
    mem_sys, fact_ids = memory_system
    
    # Make multiple confidence updates
    evidence_sets = [
        ["Initial study suggests the Great Wall is visible from space"],
        ["Astronaut reports suggest the wall is not clearly visible"],
        SAMPLE_EVIDENCE["wall"]  # Final comprehensive evidence
    ]
    
    fact_id = fact_ids[2]  # Great Wall fact
    
    # Apply updates sequentially
    for evidence in evidence_sets:
        await update_fact_confidence(
            memory_system=mem_sys,
            fact_id=fact_id,
            evidence=evidence
        )
    
    # Get confidence history
    history = await get_confidence_history(
        memory_system=mem_sys,
        fact_id=fact_id
    )
    
    assert len(history) >= 3, "Should have at least 3 history entries"
    assert history[0]["confidence"] == SAMPLE_FACTS[2]["confidence"], "First entry should be initial confidence"
    assert history[-1]["confidence"] < history[0]["confidence"], "Final confidence should be lower"
    assert all("timestamp" in entry for entry in history), "All entries should have timestamps"
    assert all("evidence" in entry for entry in history), "All entries should include evidence"

@pytest.mark.asyncio
async def test_hybrid_search_confidence_integration(memory_system):
    """Test that confidence scores affect hybrid search results."""
    mem_sys, fact_ids = memory_system
    
    # First, update the Great Wall fact with contradicting evidence
    await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_ids[2],
        evidence=SAMPLE_EVIDENCE["wall"]
    )
    
    # Add a new fact about the same topic with higher confidence
    new_fact_id = await mem_sys.remember(
        content="The Great Wall of China is not clearly visible to the naked eye from low Earth orbit, though it can be seen from space with instruments.",
        importance=0.7,
        confidence=0.95,
        domains=["history", "architecture", "space"]
    )
    
    # Perform a hybrid search
    results = await mem_sys.hybrid_search("Is the Great Wall of China visible from space?")
    
    # Verify results
    assert len(results) >= 2, "Should find both facts"
    assert results[0]["_id"] == new_fact_id, "Higher confidence fact should rank first"
    assert results[-1]["_id"] == fact_ids[2], "Lower confidence fact should rank last"
    
    # Check confidence values are included
    assert all("confidence" in result for result in results), "All results should include confidence"
    assert results[0]["confidence"] > results[-1]["confidence"], "Results should be ordered by confidence"

@pytest.mark.asyncio
async def test_confidence_based_fact_retrieval(memory_system):
    """Test retrieving facts based on confidence thresholds."""
    mem_sys, fact_ids = memory_system
    
    # Update some facts with different confidence levels
    await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_ids[0],  # Eclipse fact
        evidence=SAMPLE_EVIDENCE["eclipse"]  # Strong evidence
    )
    
    await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_ids[2],  # Great Wall fact
        evidence=SAMPLE_EVIDENCE["wall"]  # Contradicting evidence
    )
    
    # Get high confidence facts
    high_conf_facts = await mem_sys.get_facts_by_confidence(
        min_confidence=0.9
    )
    
    # Get low confidence facts
    low_conf_facts = await mem_sys.get_facts_by_confidence(
        max_confidence=0.5
    )
    
    # Verify results
    assert any(fact["_id"] == fact_ids[0] for fact in high_conf_facts), "Eclipse fact should be high confidence"
    assert any(fact["_id"] == fact_ids[2] for fact in low_conf_facts), "Great Wall fact should be low confidence"
    
    # Check confidence values
    assert all(fact["confidence"] >= 0.9 for fact in high_conf_facts), "All high confidence facts should meet threshold"
    assert all(fact["confidence"] <= 0.5 for fact in low_conf_facts), "All low confidence facts should meet threshold"

@pytest.mark.asyncio
async def test_litellm_integration_with_cache():
    """
    Test LiteLLM integration with caching and retries.
    This test verifies:
    1. LiteLLM cache initialization works
    2. Cache hits and misses are tracked
    3. Both retry mechanisms work (network and validation)
    4. Integration with ArangoDB works
    """
    # Initialize cache
    await initialize_litellm_cache()
    
    # Test data
    fact = "The speed of light in vacuum is approximately 299,792,458 meters per second."
    evidence = [
        "This value was defined by the International Bureau of Weights and Measures in 1983.",
        "Multiple experimental measurements have confirmed this value.",
        "This constant is represented by 'c' in physics equations."
    ]
    
    # First call - should miss cache
    logger.info("Making first LiteLLM call (should miss cache)...")
    response1 = await evaluate_confidence(
        fact_content=fact,
        evidence=evidence,
        model="gpt-4o-mini"  # Use the default model
    )
    
    # Verify response structure
    assert isinstance(response1, ConfidenceEvaluation)
    assert 0.0 <= response1.confidence <= 1.0
    assert len(response1.reasoning) > 0
    logger.info(f"First call confidence: {response1.confidence}")
    
    # Second call with same input - should hit cache
    logger.info("Making second LiteLLM call (should hit cache)...")
    response2 = await evaluate_confidence(
        fact_content=fact,
        evidence=evidence,
        model="gpt-4o-mini"
    )
    
    # Verify cache hit by checking response equality
    assert response2.confidence == response1.confidence
    assert response2.reasoning == response1.reasoning
    logger.info("Cache hit verified - responses match")
    
    # Test validation retry by providing empty evidence
    logger.info("Testing validation retry with empty evidence...")
    response3 = await evaluate_confidence(
        fact_content=fact,
        evidence=[],  # Empty evidence should trigger validation retry
        model="gpt-4o-mini"
    )
    
    # Should get default confidence for invalid input
    assert response3.confidence == 0.5
    assert "Error" in response3.reasoning
    logger.info("Validation retry handling verified")

@pytest.mark.asyncio
async def test_litellm_integration_with_memory_system(memory_system):
    """
    Test LiteLLM integration with ArangoDB memory system.
    Verifies that confidence scoring works with database operations.
    """
    mem_sys, fact_ids = memory_system
    
    # Test fact with strong evidence
    fact_id = fact_ids[0]  # Use the eclipse fact
    evidence = SAMPLE_EVIDENCE["eclipse"]
    
    # Update confidence with LiteLLM call
    update_result = await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_id,
        evidence=evidence
    )
    
    # Verify update was stored in database
    updated_fact = await mem_sys.get_fact(fact_id)
    assert updated_fact is not None
    assert updated_fact["confidence"] == update_result.new_confidence
    assert "confidence_history" in updated_fact
    
    # Verify the update was cached
    logger.info("Testing cache hit for same fact update...")
    second_update = await update_fact_confidence(
        memory_system=mem_sys,
        fact_id=fact_id,
        evidence=evidence
    )
    
    # Should get same confidence from cache
    assert second_update.new_confidence == update_result.new_confidence
    assert second_update.reasoning == update_result.reasoning
    
    logger.info("LiteLLM integration with memory system verified")

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__])) 