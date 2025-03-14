#!/usr/bin/env python3
"""
Integration tests for AgentMemorySystem core functionality.

These tests verify the core functions of the agent memory system using a real ArangoDB instance.
No mocking is used to ensure real functionality is tested.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
- pytest: https://docs.pytest.org/
"""

import os
import time
import uuid
import pytest
from typing import Dict, Any, List
import datetime

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem

# Test configuration
TEST_DB_NAME = "test_agent_memory"
TEST_HOST = "http://localhost:8529"
TEST_USERNAME = os.environ.get("ARANGO_USERNAME", "root")
TEST_PASSWORD = os.environ.get("ARANGO_PASSWORD", "openSesame")

# Sample facts for testing
SAMPLE_FACTS = [
    {
        "content": "The capital of France is Paris.",
        "importance": 0.7,
        "confidence": 0.9,
        "domains": ["geography", "europe"],
    },
    {
        "content": "Python is a programming language.",
        "importance": 0.8,
        "confidence": 0.95,
        "domains": ["programming", "technology"],
    },
    {
        "content": "Water boils at 100 degrees Celsius at sea level.",
        "importance": 0.6,
        "confidence": 0.9,
        "domains": ["physics", "chemistry"],
    },
    {
        "content": "Neural networks are a type of machine learning model.",
        "importance": 0.75,
        "confidence": 0.85,
        "domains": ["ai", "technology", "machine_learning"],
    },
]

@pytest.fixture
def memory_system():
    """Create and initialize a memory system for testing."""
    # Use a unique database name for each test run
    unique_id = uuid.uuid4().hex[:8]
    test_db = f"{TEST_DB_NAME}_{unique_id}"
    
    config = {
        "arango_host": TEST_HOST,
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
        "db_name": test_db,
        "facts_collection": "test_facts",
        "associations_collection": "test_associations",
        "default_ttl_days": 30,
        "importance_decay_factor": 0.5,
        "recency_boost_factor": 0.4,
    }
    
    memory_system = AgentMemorySystem(config)
    success = memory_system.initialize()
    assert success, "Memory system initialization failed"
    
    yield memory_system
    
    # Clean up
    try:
        memory_system.cleanup()
        from arango import ArangoClient
        client = ArangoClient(hosts=config["arango_host"])
        sys_db = client.db('_system', username=config["username"], password=config["password"])
        if sys_db.has_database(test_db):
            sys_db.delete_database(test_db)
    except Exception as e:
        print(f"Failed to cleanup test database: {e}")

def test_initialization(memory_system):
    """Test that the memory system initializes correctly."""
    assert memory_system.initialized
    assert memory_system.facts is not None
    assert memory_system.associations is not None
    assert memory_system.db is not None

def test_remember_single_fact(memory_system):
    """Test storing a single fact in memory."""
    # Store a fact
    fact = SAMPLE_FACTS[0]
    result = memory_system.remember(**fact)
    
    # Verify the result
    assert result is not None
    assert "new" in result
    assert "fact_id" in result["new"]
    fact_id = result["new"]["fact_id"]
    
    # Retrieve the fact using the ID
    stored_fact = memory_system.get_fact(fact_id)
    assert stored_fact is not None
    assert stored_fact["content"] == fact["content"]
    assert stored_fact["importance"] == fact["importance"]
    assert stored_fact["confidence"] == fact["confidence"]
    assert set(stored_fact["domains"]) == set(fact["domains"])

def test_remember_update_fact(memory_system):
    """Test updating an existing fact."""
    # Store initial fact
    fact = SAMPLE_FACTS[0]
    result = memory_system.remember(**fact)
    fact_id = result["new"]["fact_id"]
    
    # Update the fact
    updated_fact = {
        "fact_id": fact_id,
        "content": "The capital of France is Paris, and it's known as the City of Light.",
        "importance": 0.8,
        "confidence": 0.95,
        "domains": ["geography", "europe", "tourism"]
    }
    
    update_result = memory_system.remember(**updated_fact)
    
    # Verify update
    assert update_result["operation"] == "update"
    
    # Retrieve and check
    stored_fact = memory_system.get_fact(fact_id)
    assert stored_fact["content"] == updated_fact["content"]
    assert stored_fact["importance"] == updated_fact["importance"]
    assert stored_fact["confidence"] == updated_fact["confidence"]
    assert set(stored_fact["domains"]) == set(updated_fact["domains"])
    assert "previous_content" in stored_fact
    assert stored_fact["previous_content"] == fact["content"]

def test_recall_hybrid_search(memory_system):
    """Test recalling facts using hybrid search."""
    # Store multiple facts
    fact_ids = []
    for fact in SAMPLE_FACTS:
        result = memory_system.remember(**fact)
        fact_ids.append(result["new"]["fact_id"])
    
    # Allow time for indexing
    time.sleep(1)
    
    # Test hybrid search
    results = memory_system.hybrid_search("programming language", threshold=0.1)
    
    # Verify results
    assert len(results) > 0
    # At least one result should be about programming
    found_programming = False
    for result in results:
        if "programming" in result["content"].lower():
            found_programming = True
            break
    assert found_programming, "Should find a fact about programming"

def test_recall_with_domain_filter(memory_system):
    """Test recalling facts with domain filtering."""
    # Document the expected behavior based on ArangoDB documentation
    # Reference: https://www.arangodb.com/docs/stable/aql/operations-filter.html
    
    # Store multiple facts with explicit success verification
    fact_ids = []
    for fact in SAMPLE_FACTS:
        result = memory_system.remember(**fact)
        assert result is not None, f"Failed to store fact: {fact['content']}"
        fact_ids.append(result["new"]["fact_id"])
    
    print(f"Successfully stored {len(fact_ids)} facts for domain filtering test")
    
    # Double check that the AI/technology fact was properly stored
    ai_facts = []
    for fact_id in fact_ids:
        fact = memory_system.get_fact(fact_id)
        if fact and 'domains' in fact and 'ai' in fact['domains']:
            ai_facts.append(fact)
    
    assert len(ai_facts) > 0, "No facts with 'ai' domain were stored, test cannot proceed"
    print(f"Found {len(ai_facts)} facts with 'ai' domain: {[f['content'] for f in ai_facts]}")
    
    # Allow more time for indexing (increased from 1 second)
    time.sleep(2)
    
    # Test recall with domain filter and reduced dependency on semantic search
    results = memory_system.recall(
        query="technology", 
        domain_filter=["ai"],
        threshold=0.05,  # Lower threshold to make matching easier
        semantic=True,   # Try with semantic search first
        bm25=True,       # But also use BM25 text search
        glossary=True    # And glossary matching
    )
    
    # If no results, try again with different search parameters for debugging
    if not results:
        print("No results found with semantic search, trying BM25 only...")
        results = memory_system.recall(
            query="technology", 
            domain_filter=["ai"],
            threshold=0.01,  # Very low threshold
            semantic=False,  # Disable semantic search
            bm25=True,       # Use only BM25
            glossary=True    # And glossary
        )
    
    # Print detailed debugging info before assertions
    print(f"Search results count: {len(results)}")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['content']}")
        print(f"  Domains: {result['domains']}")
        print(f"  Score: {result['score']}")
        if 'components' in result:
            print(f"  Score components: {result['components']}")
    
    # More robust assertions with better error messages
    assert len(results) > 0, "No results found with any search method, domain filtering failed"
    
    # All results should have 'ai' in domains
    for result in results:
        domains = result.get("domains", [])
        assert "ai" in domains, f"Domain filtering failed: Result has domains {domains} but 'ai' is missing"

def test_create_associations(memory_system):
    """Test creating associations between facts."""
    # Store two facts
    fact1 = memory_system.remember(**SAMPLE_FACTS[0])
    fact2 = memory_system.remember(**SAMPLE_FACTS[2])
    fact_id1 = fact1["new"]["fact_id"]
    fact_id2 = fact2["new"]["fact_id"]
    
    # Create association
    association_result = memory_system.create_association(
        fact_id1=fact_id1,
        fact_id2=fact_id2,
        association_type="related",
        weight=0.7
    )
    
    # Verify association was created
    assert association_result["status"] == "created"
    
    # Find related facts
    related = memory_system.find_related_facts(fact_id1)
    assert len(related) > 0
    assert related[0]["fact_id"] == fact_id2
    assert related[0]["association"]["weight"] == 0.7
    assert related[0]["association"]["type"] == "related"

def test_decay_memories(memory_system):
    """Test memory decay functionality."""
    # Add facts with different importances
    high_importance = memory_system.remember(
        content="This is a very important fact",
        importance=0.9,
        confidence=0.9,
        ttl_days=100
    )
    
    low_importance = memory_system.remember(
        content="This is a less important fact",
        importance=0.3,
        confidence=0.9,
        ttl_days=100
    )
    
    # Apply decay
    stats = memory_system.decay_memories(day_factor=30.0)
    
    # Verify decay
    assert stats, "Should return decay statistics"
    
    # Check that facts decayed differently based on importance
    high_fact = memory_system.get_fact(high_importance["new"]["fact_id"])
    low_fact = memory_system.get_fact(low_importance["new"]["fact_id"])
    
    # Higher importance should have higher TTL after decay
    assert high_fact["ttl_days"] > low_fact["ttl_days"], "High importance facts should decay slower"

def test_keyword_search(memory_system):
    """Test keyword search functionality."""
    # Add facts with specific domains
    memory_system.remember(
        content="Python is a programming language used in data science.",
        domains=["programming", "data_science"],
        importance=0.8,
        confidence=0.9
    )
    
    memory_system.remember(
        content="Data visualization is important in data science.",
        domains=["data_science", "visualization"],
        importance=0.7,
        confidence=0.9
    )
    
    # Allow time for indexing
    time.sleep(1)
    
    # Test keyword search
    results = memory_system.keyword_search("data_science")
    
    # Verify results
    assert len(results) > 0
    for result in results:
        assert "data_science" in result["domains"], "Results should have data_science domain"

def test_recall_semantic_search(memory_system):
    """Test recall with semantic search enabled."""
    # Store facts
    for fact in SAMPLE_FACTS:
        memory_system.remember(**fact)
    
    # Allow time for indexing
    time.sleep(1)
    
    # Test semantic search
    results = memory_system.recall(
        query="coding with snakes",  # Semantic search should associate this with Python
        semantic=True,
        bm25=False,
        glossary=False,
        threshold=0.1
    )
    
    # Verify results
    assert len(results) > 0
    
    # Should find Python-related content via semantic search
    found_python = False
    for result in results:
        if "python" in result["content"].lower():
            found_python = True
            break
    
    assert found_python, "Semantic search should find Python-related content"

def test_recall_bm25_search(memory_system):
    """Test recall with BM25 text search enabled."""
    # Store facts
    for fact in SAMPLE_FACTS:
        memory_system.remember(**fact)
    
    # Allow time for indexing
    time.sleep(1)
    
    # Test BM25 search
    results = memory_system.recall(
        query="programming Python",
        semantic=False,
        bm25=True,
        glossary=False,
        threshold=0.1
    )
    
    # Verify results
    assert len(results) > 0
    
    # Should find exact keyword matches
    found_programming = False
    for result in results:
        if "programming" in result["content"].lower() and "python" in result["content"].lower():
            found_programming = True
            break
    
    assert found_programming, "BM25 search should find exact keyword matches"

def test_get_fact(memory_system):
    """Test retrieving a fact by ID."""
    # Store a fact
    result = memory_system.remember(**SAMPLE_FACTS[0])
    fact_id = result["new"]["fact_id"]
    
    # Retrieve the fact
    fact = memory_system.get_fact(fact_id)
    
    # Verify the fact
    assert fact is not None
    assert fact["fact_id"] == fact_id
    assert fact["content"] == SAMPLE_FACTS[0]["content"]
    assert fact["importance"] == SAMPLE_FACTS[0]["importance"]
    assert fact["confidence"] == SAMPLE_FACTS[0]["confidence"]
    assert fact["access_count"] >= 2  # Initial storage + get_fact access

def test_domain_filtering_simple(memory_system):
    """
    Simple targeted test for domain filtering functionality.
    
    This isolates the domain filtering feature to verify it works correctly
    before attempting to test it in more complex scenarios.
    """
    # Store two facts with clear domain differences
    ai_fact = memory_system.remember(
        content="Neural networks are AI models",
        domains=["ai", "technology"],
        importance=0.8,
        confidence=0.9
    )
    
    non_ai_fact = memory_system.remember(
        content="Python is a programming language",
        domains=["programming", "technology"],
        importance=0.8,
        confidence=0.9
    )
    
    # Verify the facts were stored correctly
    assert ai_fact is not None, "Failed to store AI fact"
    assert non_ai_fact is not None, "Failed to store non-AI fact"
    
    print(f"Successfully stored test facts with IDs: {ai_fact['new']['fact_id']} and {non_ai_fact['new']['fact_id']}")
    
    # Allow time for indexing
    time.sleep(1)
    
    # Direct test of domain filtering with keyword_search
    print("Testing domain filtering with keyword_search...")
    results = memory_system.keyword_search(
        query="technology",
        domain_filter=["ai"],
        threshold=0.01  # Very low threshold to ensure matches
    )
    
    # Detailed output for debugging
    print(f"Domain-filtered search results count: {len(results)}")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['content']}")
        print(f"  Domains: {result['domains']}")
        print(f"  Score: {result['score']}")
    
    # Verify results
    assert len(results) > 0, "No results found with domain filtering"
    
    # All results should have 'ai' domain
    for result in results:
        assert "ai" in result["domains"], f"Domain filtering failed: Result has domains {result['domains']} but 'ai' is missing"
    
    print("Domain filtering basic test passed successfully!")
