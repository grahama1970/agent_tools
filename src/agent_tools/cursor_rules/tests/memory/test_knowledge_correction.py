#!/usr/bin/env python3
"""
Integration test for the knowledge correction functionality.

This test verifies that knowledge correction and updates work correctly using a real ArangoDB instance.
"""

import asyncio
import os
import pytest
import uuid
from typing import Dict, Any, List
import datetime

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem
from agent_tools.cursor_rules.core.memory.knowledge_correction import (
    update_knowledge,
    find_similar_facts,
    get_fact_history,
    resolve_contradictions,
    analyze_contradictions,
    merge_facts
)

# Test configuration
TEST_DB_NAME = "test_agent_knowledge_correction"
TEST_COLLECTION = "test_memory_facts"
TEST_USERNAME = os.environ.get("ARANGO_USERNAME", "root")
TEST_PASSWORD = os.environ.get("ARANGO_PASSWORD", "openSesame")

# Sample facts for testing
SAMPLE_FACTS = [
    {
        "content": "GPT-4 was released in March 2023.",
        "importance": 0.8,
        "confidence": 0.9,
        "domains": ["ai", "llm", "history"]
    },
    {
        "content": "The population of Tokyo is 37 million (2021 estimate).",
        "importance": 0.6,
        "confidence": 0.8,
        "domains": ["geography", "demographics"]
    },
    {
        "content": "The population of Tokyo metropolitan area is 39.5 million (2023 estimate).",
        "importance": 0.7,
        "confidence": 0.9,
        "domains": ["geography", "demographics"]
    }
]

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
        "associations_collection": "test_associations",
        "graph_name": "test_memory_graph",
        "embeddings_model": "all-MiniLM-L6-v2"  # Use a small model for tests
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
async def test_update_knowledge(memory_system):
    """Test updating existing knowledge with new information."""
    mem_sys, fact_ids = memory_system
    
    # Update GPT-4 fact with more precise information
    updated_content = "GPT-4 was released by OpenAI on March 14, 2023."
    fact_id, is_new, previous = await update_knowledge(
        memory_system=mem_sys,
        fact_content=updated_content,
        confidence=0.95,
        domains=["ai", "llm", "history"],
        metadata={"source": "OpenAI blog"}
    )
    
    assert not is_new, "Should update existing fact"
    assert previous is not None, "Should return previous version"
    assert previous["content"] == SAMPLE_FACTS[0]["content"], "Previous content should match"
    
    # Verify the update
    fact = await asyncio.to_thread(
        lambda: mem_sys.db.collection(TEST_COLLECTION).get(fact_id)
    )
    
    assert fact["content"] == updated_content, "Content should be updated"
    assert fact["confidence"] == 0.95, "Confidence should be updated"
    assert fact["metadata"]["source"] == "OpenAI blog", "Metadata should be added"
    assert fact["version"] > 1, "Version should be incremented"

@pytest.mark.asyncio
async def test_resolve_contradictions(memory_system):
    """Test resolving contradicting facts."""
    mem_sys, fact_ids = memory_system
    
    # Find contradictions in Tokyo population facts
    contradictions = await analyze_contradictions(mem_sys)
    
    assert len(contradictions) > 0, "Should find contradicting facts"
    
    # Resolve the contradiction
    resolution = await resolve_contradictions(
        memory_system=mem_sys,
        fact_id=fact_ids[1],  # Older Tokyo population fact
        resolution_content="The Tokyo metropolitan area has a population of 39.5 million (2023), while the city proper has 14 million.",
        confidence=0.95,
        resolution_notes="Merged and clarified city proper vs metropolitan area statistics"
    )
    
    assert resolution["resolved"], "Should successfully resolve contradiction"
    assert resolution["merged_facts"] == 2, "Should merge two facts"
    
    # Verify the resolution
    resolved_fact = await asyncio.to_thread(
        lambda: mem_sys.db.collection(TEST_COLLECTION).get(resolution["new_fact_id"])
    )
    
    assert resolved_fact["confidence"] == 0.95, "Resolved fact should have high confidence"
    assert "resolution_notes" in resolved_fact["metadata"], "Should include resolution notes"
    assert resolved_fact["version"] == 1, "Should be a new fact"

@pytest.mark.asyncio
async def test_merge_facts(memory_system):
    """Test merging related facts."""
    mem_sys, fact_ids = memory_system
    
    # Create some related facts about the same topic
    fact1_id = await mem_sys.remember(
        content="Python 3.9 introduced the union operator for dictionaries.",
        importance=0.7,
        confidence=0.9,
        domains=["programming", "python"]
    )
    
    fact2_id = await mem_sys.remember(
        content="The Python 3.9 union operator uses the | symbol for dictionaries.",
        importance=0.6,
        confidence=0.8,
        domains=["programming", "python"]
    )
    
    # Merge the facts
    merged = await merge_facts(
        memory_system=mem_sys,
        fact_ids=[fact1_id, fact2_id],
        merged_content="Python 3.9 introduced the union operator (|) for dictionaries.",
        confidence=0.95,
        merge_notes="Combined syntax information with feature description"
    )
    
    assert merged["success"], "Should successfully merge facts"
    assert len(merged["removed_ids"]) == 2, "Should remove both original facts"
    
    # Verify the merged fact
    merged_fact = await asyncio.to_thread(
        lambda: mem_sys.db.collection(TEST_COLLECTION).get(merged["new_fact_id"])
    )
    
    assert merged_fact["confidence"] == 0.95, "Merged fact should have high confidence"
    assert "merge_notes" in merged_fact["metadata"], "Should include merge notes"
    assert all(domain in merged_fact["domains"] for domain in ["programming", "python"]), "Should preserve domains"

@pytest.mark.asyncio
async def test_hybrid_search_integration(memory_system):
    """Test that knowledge correction properly updates search indices."""
    mem_sys, fact_ids = memory_system
    
    # Add a fact that will be updated
    original_id = await mem_sys.remember(
        content="React was created by Facebook.",
        importance=0.7,
        confidence=0.8,
        domains=["programming", "web"]
    )
    
    # Update the knowledge with more precise information
    updated_content = "React was created by Facebook (now Meta) and released in 2013."
    fact_id, is_new, previous = await update_knowledge(
        memory_system=mem_sys,
        fact_content=updated_content,
        confidence=0.9,
        domains=["programming", "web", "history"],
        metadata={"source": "React documentation"}
    )
    
    # Perform a hybrid search
    results = await mem_sys.hybrid_search("When was React created?")
    
    # Verify search results
    assert len(results) > 0, "Should find the updated fact"
    assert results[0]["content"] == updated_content, "Should return updated content"
    assert results[0]["confidence"] == 0.9, "Should have updated confidence"
    
    # Verify old content is not found
    old_results = await mem_sys.hybrid_search("Facebook created React")
    assert all(r["content"] != previous["content"] for r in old_results), "Old content should not be found"

@pytest.mark.asyncio
async def test_confidence_based_ranking(memory_system):
    """Test that confidence scores affect hybrid search ranking."""
    mem_sys, fact_ids = memory_system
    
    # Add multiple facts about the same topic with different confidence levels
    low_conf_id = await mem_sys.remember(
        content="TypeScript was created by Microsoft.",
        importance=0.7,
        confidence=0.7,
        domains=["programming", "languages"]
    )
    
    high_conf_id = await mem_sys.remember(
        content="TypeScript was created by Microsoft and first released in 2012.",
        importance=0.7,
        confidence=0.95,
        domains=["programming", "languages", "history"]
    )
    
    # Perform a hybrid search
    results = await mem_sys.hybrid_search("Who created TypeScript and when?")
    
    # Verify confidence-based ranking
    assert len(results) >= 2, "Should find both facts"
    assert results[0]["_id"] == high_conf_id, "Higher confidence fact should rank first"
    assert results[-1]["_id"] == low_conf_id, "Lower confidence fact should rank last"

def test_update_knowledge_new_fact(memory_system):
    """Test updating knowledge with a new fact."""
    fact_id, is_new, previous = update_knowledge(
        memory_system,
        "The sky is blue",
        confidence=0.9,
        domains=["nature", "colors"],
        ttl_days=365.0,
        importance=0.7
    )
    
    assert fact_id is not None
    assert is_new is True
    assert previous is None
    
    fact = memory_system.get_fact(fact_id)
    assert fact["content"] == "The sky is blue"
    assert fact["confidence"] == 0.9
    assert fact["domains"] == ["nature", "colors"]
    assert fact["ttl_days"] == 365.0
    assert fact["importance"] == 0.7

def test_update_knowledge_similar_fact(memory_system):
    """Test updating knowledge with a similar fact."""
    # Create initial fact
    fact_id = memory_system.remember(
        content="The sky appears blue",
        confidence=0.8,
        domains=["nature"],
        ttl_days=365.0,
        importance=0.5
    )
    
    # Update with similar fact
    new_id, is_new, previous = update_knowledge(
        memory_system,
        "The sky is blue in color",
        confidence=0.9,
        domains=["nature", "colors"],
        ttl_days=365.0,
        importance=0.7
    )
    
    assert new_id == fact_id  # Should update existing fact
    assert is_new is False
    assert previous is not None
    assert previous["content"] == "The sky appears blue"
    
    # Check updated fact
    fact = memory_system.get_fact(fact_id)
    assert fact["content"] == "The sky is blue in color"
    assert fact["confidence"] == 0.9
    assert set(fact["domains"]) == {"nature", "colors"}
    assert fact["importance"] > 0.5  # Should increase
    assert "correction_history" in fact
    assert len(fact["correction_history"]) == 1

def test_find_similar_facts(memory_system):
    """Test finding similar facts."""
    # Create some facts
    fact_ids = []
    facts = [
        {
            "content": "The sky is blue",
            "confidence": 0.9,
            "domains": ["nature", "colors"]
        },
        {
            "content": "The ocean appears blue",
            "confidence": 0.8,
            "domains": ["nature", "colors"]
        },
        {
            "content": "The Earth is round",
            "confidence": 0.95,
            "domains": ["science"]
        }
    ]
    
    for fact in facts:
        fact_id = memory_system.remember(**fact)
        fact_ids.append(fact_id)
    
    # Find similar facts
    similar = find_similar_facts(
        memory_system,
        "The sky looks blue",
        similarity_threshold=0.8
    )
    
    assert len(similar) > 0
    assert similar[0]["content"] == "The sky is blue"
    assert "similarity_score" in similar[0]
    assert similar[0]["similarity_score"] >= 0.8

def test_get_fact_history(memory_system):
    """Test getting fact history."""
    # Create a fact and update it
    fact_id = memory_system.remember(
        content="The sky is blue",
        confidence=0.8,
        domains=["nature"]
    )
    
    # Update it multiple times
    memory_system.remember(
        fact_id=fact_id,
        content="The sky appears blue",
        confidence=0.9,
        domains=["nature", "colors"]
    )
    
    memory_system.remember(
        fact_id=fact_id,
        content="The sky is colored blue",
        confidence=0.95,
        domains=["nature", "colors", "science"]
    )
    
    # Get history
    history = get_fact_history(memory_system, fact_id)
    
    assert history is not None
    assert "current" in history
    assert history["current"]["content"] == "The sky is colored blue"
    assert history["current"]["confidence"] == 0.95
    assert set(history["current"]["domains"]) == {"nature", "colors", "science"}

def test_resolve_contradictions(memory_system):
    """Test resolving contradictions between facts."""
    # Create a fact with contradictions
    fact_id = memory_system.remember(
        content="The Earth is flat",
        confidence=0.5,
        domains=["science"],
        alternatives=[
            {
                "content": "The Earth is round",
                "confidence": 0.9,
                "timestamp": datetime.datetime.now().isoformat()
            },
            {
                "content": "The Earth is spherical",
                "confidence": 0.8,
                "timestamp": datetime.datetime.now().isoformat()
            }
        ]
    )
    
    # Resolve the contradiction
    resolved_id = resolve_contradictions(
        memory_system,
        fact_id,
        "The Earth is an oblate spheroid",
        confidence=0.95,
        resolution_notes="Combined scientific observations"
    )
    
    # Check resolved fact
    resolved = memory_system.get_fact(resolved_id)
    assert resolved["content"] == "The Earth is an oblate spheroid"
    assert resolved["confidence"] == 0.95
    assert "resolved_from" in resolved
    assert resolved["resolved_from"] == fact_id
    
    # Check original fact
    original = memory_system.get_fact(fact_id)
    assert "resolved_by" in original
    assert original["resolved_by"] == resolved_id

def test_analyze_contradictions(memory_system):
    """Test analyzing contradictions in the memory system."""
    # Create facts with contradictions
    facts = [
        {
            "content": "The Earth is flat",
            "confidence": 0.5,
            "domains": ["science"],
            "alternatives": [
                {
                    "content": "The Earth is round",
                    "confidence": 0.9,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ]
        },
        {
            "content": "The sky is green",
            "confidence": 0.3,
            "domains": ["nature"],
            "alternatives": [
                {
                    "content": "The sky is blue",
                    "confidence": 0.8,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                {
                    "content": "The sky appears azure",
                    "confidence": 0.7,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ]
        }
    ]
    
    for fact in facts:
        memory_system.remember(**fact)
    
    # Analyze contradictions
    contradictions = analyze_contradictions(memory_system)
    
    assert len(contradictions) == 2
    assert contradictions[0]["contradiction_score"] > contradictions[1]["contradiction_score"]
    assert len(contradictions[0]["alternatives"]) == 2  # Sky fact has more alternatives
    assert len(contradictions[1]["alternatives"]) == 1  # Earth fact has one alternative

def test_merge_facts(memory_system):
    """Test merging related facts."""
    # Create facts to merge
    fact_ids = []
    facts = [
        {
            "content": "The sky appears blue during the day",
            "confidence": 0.8,
            "domains": ["nature", "time"]
        },
        {
            "content": "The sky is blue due to Rayleigh scattering",
            "confidence": 0.9,
            "domains": ["science", "physics"]
        },
        {
            "content": "Blue light is scattered more by the atmosphere",
            "confidence": 0.85,
            "domains": ["science", "optics"]
        }
    ]
    
    for fact in facts:
        fact_id = memory_system.remember(**fact)
        fact_ids.append(fact_id)
    
    # Merge the facts
    merged = merge_facts(
        memory_system,
        fact_ids,
        "The sky appears blue during the day due to Rayleigh scattering of light in the atmosphere",
        confidence=0.95,
        merge_notes="Combined scientific explanation"
    )
    
    assert merged is not None
    assert merged["confidence"] == 0.95
    assert len(merged["domains"]) == len(set(merged["domains"]))  # No duplicate domains
    assert "merged_from" in merged
    assert set(merged["merged_from"]) == set(fact_ids)
    
    # Check original facts
    for fact_id in fact_ids:
        fact = memory_system.get_fact(fact_id)
        assert "merged_into" in fact
        assert fact["merged_into"] == merged["_key"] 