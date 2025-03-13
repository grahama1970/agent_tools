#!/usr/bin/env python3
"""
Integration test for the associative memory functionality.

This test verifies that the associative memory works correctly using a real ArangoDB instance.
"""

import os
import pytest
import uuid
from typing import Dict, Any, List
import datetime

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem
from agent_tools.cursor_rules.core.memory.associative_memory import (
    AssociationType,
    create_association,
    get_associations,
    delete_association,
    traverse_associations,
    find_common_associations,
    create_semantic_associations,
    find_path,
    find_clusters,
    prune_associations,
    strengthen_association
)
from agent_tools.cursor_rules.llm.initialize_litellm_cache import initialize_litellm_cache

# Test configuration
TEST_DB_NAME = "test_agent_associative_memory"
TEST_FACTS_COLLECTION = "test_memory_facts"
TEST_ASSOC_COLLECTION = "test_memory_associations"
TEST_USERNAME = os.environ.get("ARANGO_USERNAME", "root")
TEST_PASSWORD = os.environ.get("ARANGO_PASSWORD", "openSesame")

# Sample facts for testing
SAMPLE_FACTS = [
    {
        "content": "Python is a high-level programming language.",
        "importance": 0.7,
        "confidence": 0.9,
        "domains": ["programming", "languages"]
    },
    {
        "content": "TensorFlow is a machine learning framework developed by Google.",
        "importance": 0.6,
        "confidence": 0.9,
        "domains": ["machine_learning", "programming", "google"]
    },
    {
        "content": "PyTorch is a machine learning framework developed by Facebook.",
        "importance": 0.6,
        "confidence": 0.9,
        "domains": ["machine_learning", "programming", "facebook"]
    },
    {
        "content": "Natural language processing is a subfield of artificial intelligence.",
        "importance": 0.7,
        "confidence": 0.9,
        "domains": ["ai", "nlp"]
    },
    {
        "content": "Computer vision is a field of AI that trains computers to interpret visual information.",
        "importance": 0.7,
        "confidence": 0.9,
        "domains": ["ai", "computer_vision"]
    }
]

# Initialize LiteLLM cache if needed for semantic associations
initialize_litellm_cache()

@pytest.fixture
def memory_system():
    """Create and initialize a memory system for testing."""
    # Create a unique test DB name to avoid conflicts in parallel tests
    test_db = f"{TEST_DB_NAME}_{uuid.uuid4().hex[:8]}"
    
    # Configuration with test DB
    config = {
        "arango_host": "http://localhost:8529",
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
        "db_name": test_db,
        "facts_collection": TEST_FACTS_COLLECTION,
        "associations_collection": TEST_ASSOC_COLLECTION,
        "embeddings_model": "all-MiniLM-L6-v2"  # Use a small model for tests
    }
    
    # Create the memory system
    memory_system = AgentMemorySystem(config)
    memory_system.initialize()
    
    # Add sample facts
    fact_ids = []
    for fact in SAMPLE_FACTS:
        fact_id = memory_system.remember(**fact)
        fact_ids.append(fact_id)
    
    yield memory_system, fact_ids
    
    # Cleanup: drop the test database after tests
    try:
        # We need to disconnect first to avoid "database in use" errors
        memory_system.cleanup()
        
        # Connect to _system database to drop the test DB
        from arango import ArangoClient
        client = ArangoClient(hosts=config["arango_host"])
        sys_db = client.db('_system', username=config["username"], password=config["password"])
        if sys_db.has_database(test_db):
            sys_db.delete_database(test_db)
    except Exception as e:
        print(f"Failed to cleanup test database: {e}")

@pytest.mark.asyncio
async def test_create_association(memory_system):
    """Test creating an association between two facts."""
    mem_sys, fact_ids = memory_system
    
    # Create an association between Python and TensorFlow
    assoc = await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[0],  # Python
        fact_id_2=fact_ids[1],  # TensorFlow
        assoc_type=AssociationType.RELATED,
        strength=0.8,
        metadata={"reason": "Python is often used with TensorFlow"}
    )
    
    # Verify the association was created
    assert assoc is not None, "Should return an association"
    
    # Check association details
    assert assoc["type"] == AssociationType.RELATED, "Association type should match"
    assert assoc["strength"] == 0.8, "Association strength should match"
    assert "metadata" in assoc, "Should have metadata"
    assert assoc["metadata"]["reason"] == "Python is often used with TensorFlow", "Metadata should match"

@pytest.mark.asyncio
async def test_bidirectional_association(memory_system):
    """Test creating a bidirectional association."""
    mem_sys, fact_ids = memory_system
    
    # Create a bidirectional association between NLP and AI
    assoc = await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[3],  # NLP
        fact_id_2=fact_ids[4],  # Computer Vision
        assoc_type=AssociationType.RELATED,
        bidirectional=True,
        strength=0.9
    )
    
    # Verify the association was created
    assert assoc is not None, "Should return an association"
    
    # Get associations from NLP to Computer Vision
    nlp_to_cv = get_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[3],
        direction="outbound"
    )
    
    # Get associations from Computer Vision to NLP
    cv_to_nlp = get_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[4],
        direction="outbound"
    )
    
    # Verify bidirectional associations exist
    assert len(nlp_to_cv) > 0, "Should have association from NLP to CV"
    assert len(cv_to_nlp) > 0, "Should have association from CV to NLP"
    
    # They should reference each other as inverses
    assert "inverse_of" in nlp_to_cv[0], "Forward association should reference inverse"
    assert "inverse_of" in cv_to_nlp[0], "Backward association should reference inverse"
    
    # The inverse references should match
    assert nlp_to_cv[0]["inverse_of"] == cv_to_nlp[0]["_key"], "Inverse references should match"
    assert cv_to_nlp[0]["inverse_of"] == nlp_to_cv[0]["_key"], "Inverse references should match"

@pytest.mark.asyncio
async def test_get_associations(memory_system):
    """Test getting associations for a fact."""
    mem_sys, fact_ids = memory_system
    
    # Create multiple associations from Python to other facts
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[0],  # Python
        fact_id_2=fact_ids[1],  # TensorFlow
        assoc_type=AssociationType.RELATED,
        strength=0.8
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[0],  # Python
        fact_id_2=fact_ids[2],  # PyTorch
        assoc_type=AssociationType.RELATED,
        strength=0.7
    )
    
    # Create an association to Python
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[3],  # NLP
        fact_id_2=fact_ids[0],  # Python
        assoc_type=AssociationType.EXAMPLE,
        strength=0.6
    )
    
    # Get outbound associations
    outbound = get_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        direction="outbound"
    )
    
    # Get inbound associations
    inbound = get_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        direction="inbound"
    )
    
    # Get all associations
    all_assocs = get_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        direction="any"
    )
    
    # Verify counts
    assert len(outbound) == 2, "Should have 2 outbound associations"
    assert len(inbound) == 1, "Should have 1 inbound association"
    assert len(all_assocs) == 3, "Should have 3 total associations"
    
    # Verify association details
    assert all(assoc["direction"] == "outbound" for assoc in outbound), "All should be outbound"
    assert all(assoc["direction"] == "inbound" for assoc in inbound), "All should be inbound"
    
    # Verify other fact content is included
    for assoc in all_assocs:
        assert "other_fact" in assoc, "Should include other fact"

@pytest.mark.asyncio
async def test_delete_association(memory_system):
    """Test deleting an association."""
    mem_sys, fact_ids = memory_system
    
    # Create a bidirectional association
    assoc = await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[0],  # Python
        fact_id_2=fact_ids[1],  # TensorFlow
        assoc_type=AssociationType.RELATED,
        bidirectional=True
    )
    
    assoc_id = assoc["_key"]
    inverse_id = assoc["inverse_of"]
    
    # Verify both associations exist
    assoc_collection = mem_sys.db.collection(TEST_ASSOC_COLLECTION)
    assert assoc_collection.has(assoc_id), "Association should exist"
    assert assoc_collection.has(inverse_id), "Inverse association should exist"
    
    # Delete the association and its inverse
    result = delete_association(
        memory_system=mem_sys,
        assoc_id=assoc_id,
        delete_inverse=True
    )
    
    assert result, "Delete operation should succeed"
    
    # Verify both associations are gone
    assert not assoc_collection.has(assoc_id), "Association should be deleted"
    assert not assoc_collection.has(inverse_id), "Inverse should be deleted"

@pytest.mark.asyncio
async def test_traverse_associations(memory_system):
    """Test traversing associations to find connected facts."""
    mem_sys, fact_ids = memory_system
    
    # Create a chain of associations
    # Python -> TensorFlow -> PyTorch -> NLP -> Computer Vision
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[0],  # Python
        fact_id_2=fact_ids[1],  # TensorFlow
        assoc_type=AssociationType.RELATED
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[1],  # TensorFlow
        fact_id_2=fact_ids[2],  # PyTorch
        assoc_type=AssociationType.SIMILAR
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[2],  # PyTorch
        fact_id_2=fact_ids[3],  # NLP
        assoc_type=AssociationType.PART_OF
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[3],  # NLP
        fact_id_2=fact_ids[4],  # Computer Vision
        assoc_type=AssociationType.RELATED
    )
    
    # Traverse with depth 1 - should find TensorFlow
    depth_1 = traverse_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        max_depth=1
    )
    
    # Traverse with depth 2 - should find TensorFlow and PyTorch
    depth_2 = traverse_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        max_depth=2
    )
    
    # Traverse with depth 4 - should find all connected facts
    depth_4 = traverse_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[0],
        max_depth=4
    )
    
    # Verify results
    assert len(depth_1) == 1, "Depth 1 should find 1 fact"
    assert len(depth_2) == 2, "Depth 2 should find 2 facts"
    assert len(depth_4) == 4, "Depth 4 should find 4 facts"
    
    # Verify path information is included
    for result in depth_4:
        assert "path" in result, "Should include path information"
        assert isinstance(result["path"], list), "Path should be a list"
        assert len(result["path"]) > 0, "Path should not be empty"

@pytest.mark.asyncio
async def test_common_associations(memory_system):
    """Test finding common associations between facts."""
    mem_sys, fact_ids = memory_system
    
    # Create a common fact that relates to multiple others
    common_fact_content = "Deep learning is a subset of machine learning that uses neural networks."
    common_fact_id = mem_sys.remember(
        content=common_fact_content,
        importance=0.8,
        confidence=0.9,
        domains=["ai", "machine_learning", "deep_learning"]
    )
    
    # Create associations from multiple facts to the common fact
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[1],  # TensorFlow
        fact_id_2=common_fact_id,
        assoc_type=AssociationType.PART_OF
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[2],  # PyTorch
        fact_id_2=common_fact_id,
        assoc_type=AssociationType.PART_OF
    )
    
    await create_association(
        memory_system=mem_sys,
        fact_id_1=fact_ids[3],  # NLP
        fact_id_2=common_fact_id,
        assoc_type=AssociationType.EXAMPLE
    )
    
    # Find common associations between TensorFlow and PyTorch
    common = find_common_associations(
        memory_system=mem_sys,
        fact_ids=[fact_ids[1], fact_ids[2]]
    )
    
    # Find common associations between TensorFlow, PyTorch, and NLP
    common_three = find_common_associations(
        memory_system=mem_sys,
        fact_ids=[fact_ids[1], fact_ids[2], fact_ids[3]]
    )
    
    # Verify results
    assert len(common) > 0, "Should find common associations"
    assert len(common_three) > 0, "Should find common associations among three facts"
    
    # Verify common fact is found
    common_fact = next((f for f in common if f["fact"]["_key"] == common_fact_id), None)
    assert common_fact is not None, "Should find the common fact"
    assert "connection_strength" in common_fact, "Should have connection strength"
    
    # Verify the common fact is also found for all three inputs
    common_fact_three = next((f for f in common_three if f["fact"]["_key"] == common_fact_id), None)
    assert common_fact_three is not None, "Should find the common fact for three inputs"
    assert "connection_strength" in common_fact_three, "Should have connection strength"

@pytest.mark.asyncio
async def test_find_path(memory_system):
    """Test finding paths between facts."""
    system, fact_ids = memory_system
    
    # Create a chain of associations
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[0],
        fact_id_2=fact_ids[1],
        assoc_type=AssociationType.SIMILAR,
        strength=0.8
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[1],
        fact_id_2=fact_ids[2],
        assoc_type=AssociationType.RELATED,
        strength=0.7
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[2],
        fact_id_2=fact_ids[3],
        assoc_type=AssociationType.RELATED,
        strength=0.9
    )
    
    # Find path between first and last facts
    path = find_path(
        memory_system=system,
        start_fact_id=fact_ids[0],
        end_fact_id=fact_ids[3],
        max_depth=3,
        min_strength=0.5
    )
    
    assert path is not None, "Should find a path"
    assert path["length"] == 3, "Should find 3-hop path"
    assert all(edge["strength"] >= 0.5 for edge in path["edges"]), "All edges should meet strength threshold"
    
    # Test with higher strength threshold
    high_path = find_path(
        memory_system=system,
        start_fact_id=fact_ids[0],
        end_fact_id=fact_ids[3],
        max_depth=3,
        min_strength=0.8
    )
    
    assert high_path is None, "Should not find path with high strength threshold"

@pytest.mark.asyncio
async def test_find_clusters(memory_system):
    """Test finding clusters of associated facts."""
    system, fact_ids = memory_system
    
    # Create a cluster of strongly associated facts
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[0],
        fact_id_2=fact_ids[1],
        assoc_type=AssociationType.SIMILAR,
        strength=0.9
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[1],
        fact_id_2=fact_ids[2],
        assoc_type=AssociationType.RELATED,
        strength=0.8
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[2],
        fact_id_2=fact_ids[0],
        assoc_type=AssociationType.RELATED,
        strength=0.8
    )
    
    # Find clusters
    clusters = find_clusters(
        memory_system=system,
        min_strength=0.7,
        min_cluster_size=3
    )
    
    assert len(clusters) > 0, "Should find clusters"
    cluster = clusters[0]
    assert len(cluster["facts"]) >= 3, "Cluster should have at least 3 facts"
    assert cluster["avg_strength"] >= 0.7, "Average strength should meet threshold"
    
    # Test with higher strength threshold
    high_clusters = find_clusters(
        memory_system=system,
        min_strength=0.9,
        min_cluster_size=3
    )
    
    assert len(high_clusters) == 0, "Should not find clusters with very high strength threshold"

@pytest.mark.asyncio
async def test_prune_associations(memory_system):
    """Test pruning weak associations."""
    system, fact_ids = memory_system
    
    # Create some weak and strong associations
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[0],
        fact_id_2=fact_ids[1],
        assoc_type=AssociationType.SIMILAR,
        strength=0.9
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[1],
        fact_id_2=fact_ids[2],
        assoc_type=AssociationType.RELATED,
        strength=0.3
    )
    
    await create_association(
        memory_system=system,
        fact_id_1=fact_ids[2],
        fact_id_2=fact_ids[3],
        assoc_type=AssociationType.RELATED,
        strength=0.2
    )
    
    # Prune weak associations
    removed = prune_associations(
        memory_system=system,
        min_strength=0.5
    )
    
    assert removed == 2, "Should remove two weak associations"
    
    # Verify only strong association remains
    associations = get_associations(
        memory_system=system,
        fact_id=fact_ids[0]
    )
    
    assert len(associations) == 1, "Should have one remaining association"
    assert associations[0]["strength"] >= 0.5, "Remaining association should be strong"

@pytest.mark.asyncio
async def test_strengthen_association(memory_system):
    """Test strengthening associations."""
    system, fact_ids = memory_system
    
    # Create an association
    assoc = await create_association(
        memory_system=system,
        fact_id_1=fact_ids[0],
        fact_id_2=fact_ids[1],
        assoc_type=AssociationType.SIMILAR,
        strength=0.5
    )
    
    # Strengthen it
    result = strengthen_association(
        memory_system=system,
        assoc_id=assoc["_key"],
        amount=0.2
    )
    
    assert result["strength"] == 0.7, "Strength should increase by 0.2"
    
    # Test capping at 1.0
    result = strengthen_association(
        memory_system=system,
        assoc_id=assoc["_key"],
        amount=0.5
    )
    
    assert result["strength"] == 1.0, "Strength should be capped at 1.0"

@pytest.mark.asyncio
async def test_create_semantic_associations(memory_system):
    """Test creating semantic associations based on content similarity."""
    mem_sys, fact_ids = memory_system
    
    # Add a few more related facts to increase chances of semantic matches
    related_facts = [
        "Machine learning is a field of AI focused on creating systems that learn from data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Python libraries like NumPy and Pandas are commonly used in data science."
    ]
    
    for content in related_facts:
        mem_sys.remember(
            content=content,
            importance=0.7,
            confidence=0.9,
            domains=["ai", "machine_learning"]
        )
    
    # Create semantic associations for the TensorFlow fact
    assoc_ids = await create_semantic_associations(
        memory_system=mem_sys,
        fact_id=fact_ids[1],  # TensorFlow
        min_similarity=0.65,
        max_associations=3
    )
    
    # Verify associations were created
    assert len(assoc_ids) > 0, "Should create semantic associations"
    
    # Check that the associations exist and have correct properties
    assoc_collection = mem_sys.db.collection(TEST_ASSOC_COLLECTION)
    for assoc in assoc_ids:
        assert assoc["type"] == AssociationType.SIMILAR, "Should be SIMILAR type"
        assert "metadata" in assoc, "Should have metadata"
        assert "semantic_similarity" in assoc["metadata"], "Should have similarity score" 