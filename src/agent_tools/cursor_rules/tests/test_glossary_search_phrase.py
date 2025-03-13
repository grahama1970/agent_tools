#!/usr/bin/env python3
"""
Test script for the enhanced glossary search functionality.

This test validates the glossary search functions in the glossary_search module,
with a focus on the PHRASE-based search approach.

Documentation references:
- ArangoDB Search: https://www.arangodb.com/docs/stable/arangosearch.html
- ArangoDB AQL PHRASE: https://www.arangodb.com/docs/stable/aql/functions-arangosearch.html#phrase
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

import asyncio
import os
import json
import pytest
import pytest_asyncio
import logging
from typing import Dict, List, Any
from arango import ArangoClient

from agent_tools.cursor_rules.core.glossary_search import (
    search_glossary_terms,
    apply_filter_pattern,
    extract_terms_from_text,
    auto_identify_glossary_terms,
    semantic_glossary_search,
    hybrid_glossary_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test sample data
SAMPLE_GLOSSARY = [
    {
        "term": "Azure",
        "definition": "Microsoft's cloud computing platform",
        "category": "cloud",
        "related_terms": ["Cloud Computing", "Microsoft"],
        "source": "Microsoft Glossary"
    },
    {
        "term": "Teams",
        "definition": "Microsoft's collaboration and communication platform",
        "category": "productivity",
        "related_terms": ["Microsoft 365", "Collaboration"],
        "source": "Microsoft Glossary"
    },
    {
        "term": "Windows",
        "definition": "Microsoft's operating system for personal computers",
        "category": "operating_system",
        "related_terms": ["OS", "Microsoft"],
        "source": "Microsoft Glossary"
    },
    {
        "term": "PowerPoint",
        "definition": "Microsoft's presentation software",
        "category": "productivity",
        "related_terms": ["Microsoft Office", "Presentation"],
        "source": "Microsoft Glossary"
    }
]

# Mock embeddings (these would normally be generated from the term+definition)
MOCK_EMBEDDINGS = {
    "Azure": [0.1, 0.2, 0.3, 0.4],
    "Teams": [0.2, 0.3, 0.4, 0.5],
    "Windows": [0.3, 0.4, 0.5, 0.6],
    "PowerPoint": [0.4, 0.5, 0.6, 0.7]
}

# This fixture is module-scoped to match the event_loop fixture
@pytest_asyncio.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module.
    
    As per LESSONS_LEARNED.md, we match the event_loop scope with database fixtures.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def test_db():
    """Fixture to create a test database with a glossary collection.
    
    This follows the best practices from LESSONS_LEARNED.md for database setup.
    """
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    
    # Connect to system database with correct credentials
    # Note: Using password from the example you provided
    sys_db = await asyncio.to_thread(
        client.db, "_system", username="root", password="openSesame"
    )
    
    # Create test database if it doesn't exist
    test_db_name = "test_glossary_search_db"
    db_exists = await asyncio.to_thread(sys_db.has_database, test_db_name)
    
    if not db_exists:
        await asyncio.to_thread(sys_db.create_database, test_db_name)
        logger.info(f"Created test database: {test_db_name}")
    
    # Connect to test database
    db = await asyncio.to_thread(
        client.db, test_db_name, username="root", password="openSesame"
    )
    
    # Create glossary collection if it doesn't exist
    collection_name = "test_glossary"
    collection_exists = await asyncio.to_thread(db.has_collection, collection_name)
    
    if not collection_exists:
        collection = await asyncio.to_thread(db.create_collection, collection_name)
        logger.info(f"Created collection: {collection_name}")
    else:
        collection = await asyncio.to_thread(db.collection, collection_name)
    
    # Insert sample data
    await asyncio.to_thread(lambda: collection.truncate())  # Clear existing data
    
    for entry in SAMPLE_GLOSSARY:
        # Add mock embedding
        mock_embedding = MOCK_EMBEDDINGS.get(entry["term"], [0.1, 0.2, 0.3, 0.4])
        entry["embedding"] = mock_embedding
        
        # Insert document
        await asyncio.to_thread(lambda entry=entry: collection.insert(entry))
    
    logger.info(f"Inserted {len(SAMPLE_GLOSSARY)} glossary entries")
    
    # Create ArangoSearch view if it doesn't exist
    view_name = "test_glossary_view"
    view_exists = await asyncio.to_thread(db.has_view, view_name)
    
    if not view_exists:
        view_definition = {
            "links": {
                collection_name: {
                    "includeAllFields": False,
                    "fields": {
                        "term": {"analyzers": ["text_en"]},
                        "definition": {"analyzers": ["text_en"]},
                        "category": {"analyzers": ["identity"]},
                        "related_terms": {"analyzers": ["text_en"]},
                    }
                }
            }
        }
        
        await asyncio.to_thread(
            db.create_arangosearch_view, view_name, view_definition
        )
        logger.info(f"Created ArangoSearch view: {view_name}")
    
    # Return the database handle and configuration
    config = {
        "db": db,
        "collection_name": collection_name,
        "view_name": view_name
    }
    
    yield config
    
    # No cleanup - we'll leave the test database for inspection and reuse

@pytest.mark.asyncio
async def test_glossary_search_exact_match(test_db):
    """Test searching for exact glossary terms using PHRASE."""
    db = test_db["db"]
    view_name = test_db["view_name"]
    
    # Test single term search
    results = await search_glossary_terms(db, view_name, "Azure")
    
    # Verify results
    assert results, "Should find at least one result for 'Azure'"
    assert len(results) == 1, f"Should find exactly one result, got {len(results)}"
    assert results[0]["term"] == "Azure", f"Expected 'Azure', got {results[0]['term']}"
    
    # Test multiple terms search
    results = await search_glossary_terms(db, view_name, ["Azure", "Teams"])
    
    # Verify results
    assert len(results) == 2, f"Should find 2 results, got {len(results)}"
    terms = [r["term"] for r in results]
    assert "Azure" in terms, "Azure should be in results"
    assert "Teams" in terms, "Teams should be in results"

@pytest.mark.asyncio
async def test_glossary_search_nonexistent_term(test_db):
    """Test searching for terms that don't exist in the glossary."""
    db = test_db["db"]
    view_name = test_db["view_name"]
    
    # Test with a term that doesn't exist
    results = await search_glossary_terms(db, view_name, "NonexistentTerm")
    
    # Verify results
    assert not results, "Should not find any results for a nonexistent term"
    
    # Test with mixed existing and nonexistent terms
    results = await search_glossary_terms(db, view_name, ["Azure", "NonexistentTerm"])
    
    # Verify results
    assert len(results) == 1, f"Should find only 1 result, got {len(results)}"
    assert results[0]["term"] == "Azure", f"Expected 'Azure', got {results[0]['term']}"

@pytest.mark.asyncio
async def test_filter_pattern(test_db):
    """Test filtering glossary results with regex patterns."""
    db = test_db["db"]
    view_name = test_db["view_name"]
    
    # Get all glossary terms
    all_results = await search_glossary_terms(
        db, view_name, ["Azure", "Teams", "Windows", "PowerPoint"]
    )
    
    # Filter for terms starting with 'P'
    p_results = await apply_filter_pattern(all_results, r"^P.*")
    assert len(p_results) == 1, f"Should find 1 result starting with 'P', got {len(p_results)}"
    assert p_results[0]["term"] == "PowerPoint", f"Expected 'PowerPoint', got {p_results[0]['term']}"
    
    # Filter for terms starting with 'T'
    t_results = await apply_filter_pattern(all_results, r"^T.*")
    assert len(t_results) == 1, f"Should find 1 result starting with 'T', got {len(t_results)}"
    assert t_results[0]["term"] == "Teams", f"Expected 'Teams', got {t_results[0]['term']}"
    
    # Filter with a pattern that matches nothing
    no_results = await apply_filter_pattern(all_results, r"^Z.*")
    assert not no_results, f"Should find 0 results starting with 'Z', got {len(no_results)}"

@pytest.mark.asyncio
async def test_extract_terms_from_text():
    """Test extracting potential glossary terms from text."""
    # Simple text with known terms
    text = "I want to know about Azure and Teams, but I hate all the bugs in Windows"
    terms = extract_terms_from_text(text)
    
    assert "Azure" in terms, "Should extract 'Azure' from the text"
    assert "Teams" in terms, "Should extract 'Teams' from the text"
    assert "Windows" in terms, "Should extract 'Windows' from the text"
    
    # Text with capitalized phrases (potential multi-word terms)
    text = "Microsoft Azure is a cloud platform. PowerPoint Presentation is useful."
    terms = extract_terms_from_text(text)
    
    assert "Microsoft Azure" in terms, "Should extract 'Microsoft Azure' as a phrase"
    assert "PowerPoint Presentation" in terms, "Should extract 'PowerPoint Presentation' as a phrase"

@pytest.mark.asyncio
async def test_auto_identify_glossary_terms(test_db):
    """Test automatically identifying glossary terms in text."""
    db = test_db["db"]
    view_name = test_db["view_name"]
    
    # Text containing known glossary terms
    text = "I use Azure for cloud computing and Teams for collaboration."
    results = await auto_identify_glossary_terms(db, view_name, text)
    
    # Verify results
    assert len(results) > 0, "Should find at least one glossary term"
    terms = [r["term"] for r in results]
    assert "Azure" in terms or "Teams" in terms, "Should identify at least one of 'Azure' or 'Teams'"

@pytest.mark.asyncio
async def test_semantic_glossary_search(test_db):
    """Test semantic search with mock embeddings."""
    db = test_db["db"]
    collection_name = test_db["collection_name"]
    
    # Since we're using mock embeddings, we'll search with a vector close to Azure's embedding
    query_vector = [0.1, 0.2, 0.3, 0.4]  # Same as Azure's mock embedding
    
    results = await semantic_glossary_search(db, collection_name, query_vector, limit=2)
    
    # Verify results
    assert len(results) > 0, "Should find at least one semantically similar term"
    # The closest match should be Azure since we used its exact embedding
    assert results[0]["term"] == "Azure", f"Expected most similar term to be 'Azure', got {results[0]['term']}"

@pytest.mark.asyncio
async def test_hybrid_glossary_search(test_db):
    """Test hybrid search combining BM25 and semantic similarity."""
    db = test_db["db"]
    collection_name = test_db["collection_name"]
    view_name = test_db["view_name"]
    
    # Search text and vector that should match Azure
    search_text = "Azure cloud platform"
    query_vector = [0.1, 0.2, 0.3, 0.4]  # Azure's mock embedding
    
    results = await hybrid_glossary_search(
        db, 
        collection_name, 
        view_name,
        search_text,
        query_vector,
        limit=3
    )
    
    # Verify results
    assert len(results) > 0, "Should find at least one result with hybrid search"
    # First result should be Azure due to both keyword and vector match
    if results:
        assert results[0]["term"] == "Azure", f"Expected most relevant term to be 'Azure', got {results[0]['term']}"

if __name__ == "__main__":
    # For manual testing - run this file directly
    asyncio.run(pytest.main(["-xvs", __file__])) 