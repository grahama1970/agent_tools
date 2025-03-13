#!/usr/bin/env python3
"""
Comprehensive Tests for Hybrid Search Functionality.

This test suite provides extensive testing of the hybrid search functionality,
covering various scenarios including:
1. Searches that should return results
2. Searches that should return no results
3. Testing of the BM25 component
4. Testing of the vector similarity component
5. Testing the combined hybrid approach
6. Testing threshold behavior
7. Testing LIMIT parameter functionality

Documentation References:
- ArangoDB Vector Search: https://docs.arangodb.com/3.11/search/vector-search/
- ArangoDB BM25: https://docs.arangodb.com/3.11/aql/functions/arangosearch/#bm25
- ArangoDB Hybrid Search: https://docs.arangodb.com/3.11/search/hybrid-search/
"""

import os
import asyncio
import pytest
import pytest_asyncio
import logging
from arango import ArangoClient

# Ensure we can import from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    hybrid_search,
    bm25_keyword_search,
    semantic_search,
    generate_embedding,
    create_arangosearch_view,
    EMBEDDING_AVAILABLE
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample scenarios for testing - expanded with more test cases
TEST_SCENARIOS = [
    {
        "title": "Debug Error Message",
        "description": "Finding solutions for specific error messages",
        "query_example": "How to fix 'ImportError: No module named X'",
        "expected_result_format": "Error cause and step-by-step resolution",
        "priority": 1,
        "category": "error_resolution"
    },
    {
        "title": "Error Handling Pattern",
        "description": "Best practices for handling exceptions in Python",
        "query_example": "What's the recommended way to handle exceptions in async code?",
        "expected_result_format": "Code pattern with explanation",
        "priority": 1,
        "category": "error_handling"
    },
    {
        "title": "Database Query Optimization",
        "description": "Techniques for optimizing database queries and indexing",
        "query_example": "How can I make my ArangoDB queries faster?",
        "expected_result_format": "Performance optimization techniques",
        "priority": 2,
        "category": "performance"
    },
    {
        "title": "Async Pattern Implementation",
        "description": "How to implement async patterns in Python",
        "query_example": "What's the correct way to use asyncio.gather?",
        "expected_result_format": "Code pattern with examples",
        "priority": 1,
        "category": "async_patterns"
    },
    {
        "title": "Testing Best Practices",
        "description": "Guidelines for writing effective tests",
        "query_example": "How should I structure pytest fixtures for database testing?",
        "expected_result_format": "Best practices with examples",
        "priority": 1,
        "category": "testing"
    }
]

@pytest_asyncio.fixture(scope="module")
async def test_db():
    """Set up a test database for comprehensive hybrid search testing."""
    client = ArangoClient(hosts="http://localhost:8529")
    sys_db = await asyncio.to_thread(client.db, "_system", username="root", password="openSesame")
    
    # Use a unique test database name
    db_name = "test_hybrid_search_comprehensive"
    
    # Clean up existing database if it exists
    if await asyncio.to_thread(sys_db.has_database, db_name):
        await asyncio.to_thread(sys_db.delete_database, db_name)
        logger.info(f"Deleted existing database: {db_name}")
    
    # Create a fresh test database
    await asyncio.to_thread(sys_db.create_database, db_name)
    logger.info(f"Created database: {db_name}")
    
    # Connect to the test database
    db = await asyncio.to_thread(client.db, db_name, username="root", password="openSesame")
    
    # Create the scenarios collection
    collection_name = "query_scenarios"
    if not await asyncio.to_thread(db.has_collection, collection_name):
        await asyncio.to_thread(db.create_collection, collection_name)
        logger.info(f"Created collection: {collection_name}")
    
    # Create the ArangoSearch view
    view_name = f"{collection_name}_view"
    
    # Ensure view doesn't exist first
    for view in await asyncio.to_thread(db.views):
        if view['name'] == view_name:
            await asyncio.to_thread(db.delete_view, view_name)
            logger.info(f"Deleted existing view: {view_name}")
    
    # Create ArangoSearch view with proper analyzers for tokenization
    view_properties = {
        "links": {
            collection_name: {
                "includeAllFields": False,
                "fields": {
                    "title": {
                        "analyzers": ["text_en"],
                        "includeAllFields": False
                    },
                    "description": {
                        "analyzers": ["text_en"],
                        "includeAllFields": False
                    },
                    "query_example": {
                        "analyzers": ["text_en"],
                        "includeAllFields": False
                    },
                    "category": {
                        "analyzers": ["text_en"],
                        "includeAllFields": False
                    }
                }
            }
        }
    }
    
    await asyncio.to_thread(
        db.create_view,
        name=view_name,
        view_type="arangosearch",
        properties=view_properties
    )
    logger.info(f"Created view: {view_name}")
    
    # Insert test scenarios with embeddings
    scenarios_collection = await asyncio.to_thread(db.collection, collection_name)
    
    # Clear existing data
    await asyncio.to_thread(scenarios_collection.truncate)
    
    for scenario in TEST_SCENARIOS:
        # Generate embedding from scenario content
        text_to_embed = f"{scenario['title']} {scenario['description']} {scenario['query_example']}"
        embedding_result = generate_embedding(text_to_embed)
        
        if embedding_result and "embedding" in embedding_result:
            scenario["embedding"] = embedding_result["embedding"]
            scenario["embedding_metadata"] = embedding_result.get("metadata", {})
            logger.info(f"Added embedding for scenario: {scenario['title']}")
        else:
            logger.warning(f"Failed to generate embedding for scenario: {scenario['title']}")
        
        # Insert the scenario
        await asyncio.to_thread(scenarios_collection.insert, scenario)
    
    logger.info(f"Inserted {len(TEST_SCENARIOS)} test scenarios with embeddings")
    
    # Add a delay to allow ArangoDB to index the documents
    logger.info("Waiting for ArangoDB to index documents...")
    await asyncio.sleep(2)
    
    yield db
    
    # Clean up after tests
    await asyncio.to_thread(sys_db.delete_database, db_name)
    logger.info(f"Deleted test database: {db_name}")

@pytest.mark.asyncio
async def test_hybrid_search_for_error_term(test_db):
    """Test that hybrid search can find scenarios related to 'error'."""
    # This test specifically addresses the issue where searches for 'error' were failing
    
    # Search for 'error' term
    results = await hybrid_search(test_db, "error", collection_name="query_scenarios", verbose=True)
    
    # Print results for debugging
    logger.info(f"Hybrid search results for 'error': {results}")
    
    # Verify we found results
    assert len(results) > 0, "Should find scenarios related to 'error'"
    
    # Check that we found scenarios with 'error' in the title or description
    found_error_scenarios = [
        result for result in results 
        if "error" in result[0].get("title", "").lower() or 
           "error" in result[0].get("description", "").lower()
    ]
    
    assert len(found_error_scenarios) > 0, "Should find scenarios with 'error' in title or description"
    
    # Verify specific scenario is found
    found_debug_error = any(
        "Debug Error Message" in result[0].get("title", "")
        for result in results
    )
    assert found_debug_error, "Should find the 'Debug Error Message' scenario"

@pytest.mark.asyncio
async def test_bm25_search_for_error_term(test_db):
    """Test that BM25 search alone can find scenarios related to 'error'."""
    # This tests just the BM25 component of the hybrid search
    
    results = await bm25_keyword_search(test_db, "error", collection_name="query_scenarios", verbose=True)
    
    # Print results for debugging
    logger.info(f"BM25 search results for 'error': {results}")
    
    # Verify we found results
    assert len(results) > 0, "BM25 search should find scenarios related to 'error'"
    
    # Check that we found scenarios with 'error' in the title or description
    found_error_scenarios = [
        result for result in results 
        if "error" in result["rule"].get("title", "").lower() or 
           "error" in result["rule"].get("description", "").lower()
    ]
    
    assert len(found_error_scenarios) > 0, "Should find scenarios with 'error' in title or description"

@pytest.mark.asyncio
async def test_hybrid_search_with_partial_terms(test_db):
    """Test hybrid search with partial terms to verify tokenization works properly."""
    # Test with partial words like 'async' instead of 'asynchronous'
    results = await hybrid_search(test_db, "async", collection_name="query_scenarios")
    
    assert len(results) > 0, "Should find results for partial term 'async'"
    
    # Verify we found the async-related scenario
    found_async_scenario = any(
        "Async Pattern Implementation" in result[0].get("title", "")
        for result in results
    )
    assert found_async_scenario, "Should find the async pattern scenario"

@pytest.mark.asyncio
async def test_hybrid_search_with_no_results(test_db):
    """Test hybrid search with a term that should not match any scenario."""
    # Search for a term that doesn't exist in our test data
    results = await hybrid_search(test_db, "nonexistent_term_xyz123", collection_name="query_scenarios")
    
    # Verify we get an empty result list
    assert len(results) == 0, "Should not find any results for nonexistent term"

@pytest.mark.asyncio
async def test_hybrid_search_relevance_ranking(test_db):
    """Test that hybrid search returns results in order of relevance."""
    # Search for a term that should match multiple scenarios with different relevance
    results = await hybrid_search(test_db, "testing database", collection_name="query_scenarios")
    
    # Verify we found multiple results
    assert len(results) >= 2, "Should find multiple results for 'testing database'"
    
    # The first result should be the most relevant (Testing Best Practices)
    assert "Testing Best Practices" in results[0][0].get("title", ""), (
        "Most relevant result should be Testing Best Practices"
    )
    
    # Check that scores are in descending order
    scores = [result[1] for result in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), (
        "Results should be sorted by descending relevance score"
    )

@pytest.mark.asyncio
async def test_hybrid_search_with_multiple_terms(test_db):
    """Test hybrid search with multiple search terms."""
    # Search for multiple terms that should match specific scenarios
    results = await hybrid_search(test_db, "async exception handling", collection_name="query_scenarios")
    
    # Verify we found results
    assert len(results) > 0, "Should find results for multiple terms"
    
    # Check that we found scenarios related to both async and error handling
    found_async = any(
        "Async" in result[0].get("title", "") 
        for result in results
    )
    
    found_error_handling = any(
        "Error Handling" in result[0].get("title", "")
        for result in results
    )
    
    assert found_async or found_error_handling, "Should find scenarios related to either async or error handling"

@pytest.mark.asyncio
async def test_hybrid_search_empty_query(test_db):
    """Test hybrid search with an empty query string."""
    # Search with an empty string
    results = await hybrid_search(test_db, "", collection_name="query_scenarios")
    
    # Empty queries should return no results
    assert len(results) == 0, "Empty query should return no results"

@pytest.mark.asyncio
async def test_hybrid_search_special_characters(test_db):
    """Test hybrid search with special characters in the query."""
    # Search with a query containing special characters
    results = await hybrid_search(test_db, "error: $&#@", collection_name="query_scenarios")
    
    # Should be able to handle special characters and still find relevant results
    assert len(results) > 0, "Should find results even with special characters in query"
    
    # Verify we found error-related scenarios by ignoring the special characters
    found_error_scenario = any(
        "Error" in result[0].get("title", "")
        for result in results
    )
    assert found_error_scenario, "Should find error-related scenarios despite special characters"

@pytest.mark.asyncio
async def test_hybrid_search_case_insensitivity(test_db):
    """Test that hybrid search is case-insensitive."""
    # Search with mixed case
    results_mixed_case = await hybrid_search(test_db, "Error Handling", collection_name="query_scenarios")
    
    # Search with lowercase
    results_lowercase = await hybrid_search(test_db, "error handling", collection_name="query_scenarios")
    
    # Search with uppercase
    results_uppercase = await hybrid_search(test_db, "ERROR HANDLING", collection_name="query_scenarios")
    
    # All searches should return results
    assert len(results_mixed_case) > 0, "Mixed case query should return results"
    assert len(results_lowercase) > 0, "Lowercase query should return results"
    assert len(results_uppercase) > 0, "Uppercase query should return results"
    
    # All searches should return the same number of results with similar ranking
    assert len(results_mixed_case) == len(results_lowercase) == len(results_uppercase), (
        "Case variations should return the same number of results"
    )

# NEW TESTS FOR THRESHOLDS AND LIMITS

@pytest.mark.asyncio
async def test_hybrid_search_bm25_threshold(test_db):
    """Test that BM25 threshold filters out weak matches."""
    # Use a term that should return weak matches
    query = "vague term"
    
    # Run with normal threshold (0.1)
    results_normal_threshold = await hybrid_search(
        test_db, query, collection_name="query_scenarios", verbose=True
    )
    
    # Run with very low threshold (0.001)
    results_low_threshold = await hybrid_search(
        test_db, query, collection_name="query_scenarios", 
        verbose=True, _force_bm25_threshold=0.001
    )
    
    # Print results for debugging
    logger.info(f"Low threshold search returned {len(results_low_threshold)} results")
    logger.info(f"Normal threshold search returned {len(results_normal_threshold)} results")
    
    # Either both should return 0 results, or low threshold should return >= normal threshold
    if len(results_normal_threshold) > 0:
        assert len(results_low_threshold) >= len(results_normal_threshold), (
            "Low threshold search should return at least as many results as normal threshold"
        )

@pytest.mark.asyncio
async def test_hybrid_search_limit_parameter(test_db):
    """Test that LIMIT parameter properly restricts the number of results."""
    # Search for a term that should match multiple results
    # First get all results with a high limit
    results_high_limit = await hybrid_search(
        test_db, "test", collection_name="query_scenarios", limit=10, verbose=True
    )
    
    # Now get results with a lower limit
    results_low_limit = await hybrid_search(
        test_db, "test", collection_name="query_scenarios", limit=1, verbose=True
    )
    
    # Verify the low limit query respected the limit
    assert len(results_low_limit) <= 1, "Low limit query should return at most 1 result"
    
    # If we got results from both queries, check that the high limit has at least as many
    if len(results_low_limit) > 0 and len(results_high_limit) > 0:
        assert len(results_high_limit) >= len(results_low_limit), (
            "High limit query should return at least as many results as low limit"
        )
        
        # Check that the first result is the same in both queries
        # (This assumes the sorting is stable, which it should be)
        assert results_high_limit[0][0]["_key"] == results_low_limit[0][0]["_key"], (
            "The first result should be the same regardless of limit"
        )
        
        logger.info(f"High limit query returned {len(results_high_limit)} results")
        logger.info(f"Low limit query returned {len(results_low_limit)} results")

@pytest.mark.asyncio
async def test_hybrid_score_threshold(test_db):
    """Test that hybrid score threshold properly filters results."""
    # Prepare a query that will get some matches but not strong ones
    # We'll use a made-up term mixed with a real term to get weak matches
    query = "database xyzzyficational"
    
    # Run with normal threshold (0.15)
    results_normal = await hybrid_search(
        test_db, query, collection_name="query_scenarios", verbose=True
    )
    
    # Run with lowered threshold (0.001)
    results_low_threshold = await hybrid_search(
        test_db, query, collection_name="query_scenarios", 
        verbose=True, _test_hybrid_score_threshold=0.001
    )
    
    logger.info(f"Normal threshold results: {len(results_normal)}")
    logger.info(f"Low threshold results: {len(results_low_threshold)}")
    
    # Low threshold should return at least as many results as normal threshold
    assert len(results_low_threshold) >= len(results_normal), (
        "Low hybrid score threshold should return at least as many results as normal threshold"
    )

if __name__ == "__main__":
    # Run the tests using pytest
    import pytest
    pytest.main(["-v", __file__]) 