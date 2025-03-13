#!/usr/bin/env python3
"""
Test script for hybrid search threshold parameters.

This script demonstrates how the threshold parameters affect the search results
and follows ArangoDB best practices for async operations and queries.
"""

import asyncio
import sys
import os
import logging
from arango import ArangoClient
from src.agent_tools.cursor_rules.core.cursor_rules import (
    hybrid_search,
    generate_embedding,
    create_arangosearch_view
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample test scenarios
TEST_SCENARIOS = [
    {
        "title": "Debug Error Message",
        "description": "Finding solutions for specific error messages",
        "query_example": "How to fix 'ImportError: No module named X'",
        "priority": 1,
        "category": "error_resolution"
    },
    {
        "title": "Error Handling Pattern",
        "description": "Best practices for handling exceptions in Python",
        "query_example": "What's the recommended way to handle exceptions in async code?",
        "priority": 1,
        "category": "error_handling"
    },
    {
        "title": "Database Query Optimization",
        "description": "Techniques for optimizing database queries and indexing",
        "query_example": "How can I make my ArangoDB queries faster?",
        "priority": 2,
        "category": "performance"
    },
    {
        "title": "Async Pattern Implementation",
        "description": "How to implement async patterns in Python",
        "query_example": "What's the correct way to use asyncio.gather?",
        "priority": 1,
        "category": "async_patterns"
    },
    {
        "title": "Testing Best Practices",
        "description": "Guidelines for writing effective tests",
        "query_example": "How should I structure pytest fixtures for database testing?",
        "priority": 1,
        "category": "testing"
    }
]

async def setup_test_db():
    """Set up a test database with sample data for threshold testing."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = await asyncio.to_thread(
            client.db, "_system", username="root", password="openSesame"
        )
        
        db_name = "test_hybrid_search_comprehensive"
        
        # Create the database if it doesn't exist
        db_exists = await asyncio.to_thread(sys_db.has_database, db_name)
        if not db_exists:
            logger.info(f"Creating database: {db_name}")
            await asyncio.to_thread(sys_db.create_database, db_name)
        else:
            logger.info(f"Database {db_name} already exists")
        
        # Connect to the test database
        db = await asyncio.to_thread(
            client.db, db_name, username="root", password="openSesame"
        )
        
        # Create the collection if it doesn't exist
        collection_name = "query_scenarios"
        collection_exists = await asyncio.to_thread(db.has_collection, collection_name)
        if not collection_exists:
            logger.info(f"Creating collection: {collection_name}")
            await asyncio.to_thread(db.create_collection, collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists")
        
        # Create the ArangoSearch view
        view_name = f"{collection_name}_view"
        await asyncio.to_thread(create_arangosearch_view, db, collection_name, view_name)
        
        # Insert test scenarios with embeddings if collection is empty
        scenarios_collection = await asyncio.to_thread(db.collection, collection_name)
        
        # Check if collection is empty
        count_query = "RETURN LENGTH(FOR doc IN query_scenarios RETURN 1)"
        cursor = await asyncio.to_thread(db.aql.execute, count_query)
        count = await asyncio.to_thread(next, cursor, 0)
        
        if count == 0:
            logger.info("Inserting test scenarios")
            for scenario in TEST_SCENARIOS:
                # Generate embedding from scenario content
                text_to_embed = f"{scenario['title']} {scenario['description']} {scenario['query_example']}"
                embedding_result = generate_embedding(text_to_embed)
                
                if embedding_result and "embedding" in embedding_result:
                    scenario["embedding"] = embedding_result["embedding"]
                    scenario["embedding_metadata"] = embedding_result.get("metadata", {})
                    
                # Insert the scenario - using await with asyncio.to_thread as per best practices
                await asyncio.to_thread(scenarios_collection.insert, scenario)
                
            logger.info(f"Inserted {len(TEST_SCENARIOS)} test scenarios")
        else:
            logger.info(f"Collection already contains {count} documents")
        
        return db
    except Exception as e:
        logger.error(f"Error setting up test database: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_various_thresholds():
    """Test hybrid search with various threshold parameters."""
    try:
        # Set up the test database
        logger.info("Setting up test database...")
        db = await setup_test_db()
        
        if not db:
            logger.error("Failed to set up test database")
            return
        
        print('\n===== Testing nonexistent term =====')
        nonexistent_results = await hybrid_search(
            db, 'completely_nonexistent_xyz789', 
            collection_name='query_scenarios', 
            verbose=True
        )
        print(f'Results for nonexistent term: {len(nonexistent_results)}')
        
        print('\n===== Testing with default threshold =====')
        default_results = await hybrid_search(
            db, 'database', 
            collection_name='query_scenarios', 
            verbose=True
        )
        print(f'Results with default threshold: {len(default_results)}')
        
        print('\n===== Testing with very low BM25 threshold =====')
        low_threshold_results = await hybrid_search(
            db, 'database', 
            collection_name='query_scenarios', 
            _force_bm25_threshold=0.001, 
            verbose=True
        )
        print(f'Results with low BM25 threshold: {len(low_threshold_results)}')
        
        print('\n===== Testing with narrow limit =====')
        limited_results = await hybrid_search(
            db, 'database', 
            collection_name='query_scenarios', 
            limit=1, 
            verbose=True
        )
        print(f'Results with limit=1: {len(limited_results)}')
        
        print('\n===== Testing with low hybrid score threshold =====')
        hybrid_low_results = await hybrid_search(
            db, 'database vaguely_related', 
            collection_name='query_scenarios', 
            _test_hybrid_score_threshold=0.01, 
            verbose=True
        )
        print(f'Results with low hybrid score threshold: {len(hybrid_low_results)}')
        
        # Add a test demonstrating proper use of TOKENS instead of LIKE as per best practices
        print('\n===== Testing with proper TOKENS usage =====')
        tokens_results = await hybrid_search(
            db, 'optimiz', 
            collection_name='query_scenarios', 
            verbose=True
        )
        print(f'Results with tokenized search (should find "optimization"): {len(tokens_results)}')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Running hybrid search threshold tests...")
    asyncio.run(test_various_thresholds()) 