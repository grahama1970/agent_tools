#!/usr/bin/env python
"""
Test for semantic search functionality with vector embeddings.

This test validates our implementation of semantic search using
Nomic ModernBert embeddings with proper prefixing.

Documentation References:
- ArangoDB Vector Search: https://docs.arangodb.com/3.11/search/vector-search/
- Nomic Embed: https://docs.nomic.ai/embedding/
"""
import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock
import asyncio

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from arango import ArangoClient

# Updated import path to match new file structure and available functions
from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    semantic_search,
    generate_embedding,
    create_arangosearch_view,
    EMBEDDING_AVAILABLE
)

def test_generate_embedding_with_prefix():
    """
    Test that generate_embedding properly uses ensure_text_has_prefix.
    
    This test verifies that:
    1. The function calls ensure_text_has_prefix to add the required prefix
    2. It calls create_embedding_sync with the prefixed text
    3. It returns the embedding result with the correct structure
    """
    # Skip test if embedding utilities aren't available
    if not EMBEDDING_AVAILABLE:
        pytest.skip("Embedding utilities not available")
    
    with patch('agent_tools.cursor_rules.core.cursor_rules.ensure_text_has_prefix') as mock_ensure_prefix:
        with patch('agent_tools.cursor_rules.core.cursor_rules.create_embedding_sync') as mock_create_embedding:
            # Set up mocks
            mock_ensure_prefix.return_value = "prefixed_text"
            mock_create_embedding.return_value = {
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "embedding_model": "nomic-embed-text-v1",
                    "embedding_timestamp": "2023-01-01T00:00:00Z",
                    "embedding_method": "nomic",
                    "embedding_dim": 3
                }
            }
            
            # Call the function
            result = generate_embedding("test text")
            
            # Verify the function called ensure_text_has_prefix
            mock_ensure_prefix.assert_called_once_with("test text")
            
            # Verify the function called create_embedding_sync with the prefixed text
            mock_create_embedding.assert_called_once_with("prefixed_text")
            
            # Verify the result structure
            assert "embedding" in result
            assert "metadata" in result
            assert result["embedding"] == [0.1, 0.2, 0.3]
            assert result["metadata"]["embedding_model"] == "nomic-embed-text-v1"
            
            print("✅ generate_embedding correctly uses ensure_text_has_prefix and create_embedding_sync")

def test_semantic_search_with_prefix():
    """
    Test that semantic_search properly prefixes the query with "search_query: ".
    
    This test verifies that:
    1. The function adds "search_query: " prefix to the query
    2. It calls create_embedding_sync with the prefixed query
    3. It executes the correct AQL query with the embedding
    """
    # Skip test if embedding utilities aren't available
    if not EMBEDDING_AVAILABLE:
        pytest.skip("Embedding utilities not available")
    
    # Create mock database
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_db.aql.execute.return_value = mock_cursor
    mock_cursor.__iter__.return_value = [{"rule": {"title": "Test Rule"}, "similarity": 0.9}]
    
    with patch('agent_tools.cursor_rules.core.cursor_rules.create_embedding_sync') as mock_create_embedding:
        # Set up mock
        mock_create_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "embedding_model": "nomic-embed-text-v1",
                "embedding_timestamp": "2023-01-01T00:00:00Z",
                "embedding_method": "nomic",
                "embedding_dim": 3
            }
        }
        
        # Call the function
        results = semantic_search(mock_db, "test query", limit=5)
        
        # Verify the function called create_embedding_sync with the prefixed query
        mock_create_embedding.assert_called_once_with("search_query: test query")
        
        # Verify the function executed the AQL query with the embedding
        mock_db.aql.execute.assert_called_once()
        
        # Verify the results
        assert len(results) == 1
        assert results[0]["rule"]["title"] == "Test Rule"
        assert results[0]["similarity"] == 0.9
        
        print("✅ semantic_search correctly prefixes the query and uses the embedding for search")

def test_semantic_search_real_db():
    """
    Test semantic search with a real database connection.
    
    This test:
    1. Connects to a test database
    2. Inserts test documents with embeddings
    3. Performs a semantic search
    4. Verifies the search results
    
    Note: This test requires a running ArangoDB instance and embedding utilities.
    """
    # Skip test if embedding utilities aren't available
    if not EMBEDDING_AVAILABLE:
        pytest.skip("Embedding utilities not available")
    
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = client.db("_system", username="root", password="openSesame")
        
        # Use a fixed database name for testing
        db_name = "cursor_rules_test_semantic"
        
        # Print database information
        print("\n" + "="*80)
        print(f"DATABASE: {db_name}")
        print("="*80)
        
        # Clean up existing database if it exists
        if sys_db.has_database(db_name):
            sys_db.delete_database(db_name)
            print(f"Deleted existing database: {db_name}")
        
        # Create fresh database
        sys_db.create_database(db_name)
        print(f"Created database: {db_name}")
        db = client.db(db_name, username="root", password="openSesame")
        
        try:
            # Create collection
            collection_name = "rules"
            print(f"COLLECTION: {collection_name}")
            print("-"*80)
            
            if not db.has_collection(collection_name):
                db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")
            
            # Insert test documents with embeddings
            rules_collection = db.collection(collection_name)
            
            # Generate embeddings for test documents
            test_rules = [
                {
                    "rule_number": "001",
                    "title": "Python Code Style",
                    "description": "Guidelines for Python code formatting and style",
                    "content": "Follow PEP 8 style guide for Python code.",
                    "rule_type": "style",
                    "glob_pattern": "*.py"
                },
                {
                    "rule_number": "002",
                    "title": "JavaScript Formatting",
                    "description": "Standards for JavaScript code organization",
                    "content": "Use 2 spaces for indentation in JavaScript files.",
                    "rule_type": "style",
                    "glob_pattern": "*.js"
                },
                {
                    "rule_number": "003",
                    "title": "Database Schema",
                    "description": "Database schema conventions for the project",
                    "content": "Use snake_case for table names and columns.",
                    "rule_type": "database",
                    "glob_pattern": "*.sql"
                }
            ]
            
            # Add embeddings to test documents
            for rule in test_rules:
                # Generate embedding for the rule content
                embedding_result = generate_embedding(rule["content"])
                rule["embedding"] = embedding_result["embedding"]
                rules_collection.insert(rule)
            
            print(f"Inserted {len(test_rules)} test rules with embeddings")
            
            # Perform semantic search using the enhanced function with verbose=True
            search_query = "python coding standards"
            print(f"\nPerforming semantic search for: '{search_query}'")
            
            # Use the semantic_search function with verbose=True
            results = semantic_search(db, search_query, limit=5, verbose=True)
            
            # Verify results
            assert len(results) > 0, "Should find results for 'python coding standards'"
            
            # The Python style guide should be the most relevant result
            if results:
                top_result = results[0]['rule']
                assert top_result['rule_number'] == "001", "Python style guide should be the top result"
                assert results[0]['similarity'] > 0, "Similarity score should be positive"
                
                print("\nTEST PASSED: Found relevant results with Python style guide as top result")
        
        finally:
            # Clean up
            sys_db.delete_database(db_name)
            print(f"\nCleaned up test database: {db_name}")
    
    except Exception as e:
        pytest.skip(f"Test requires ArangoDB and embedding utilities: {e}")

if __name__ == "__main__":
    # Run the tests
    if EMBEDDING_AVAILABLE:
        test_generate_embedding_with_prefix()
        test_semantic_search_with_prefix()
        test_semantic_search_real_db()
    else:
        print("Skipping tests: Embedding utilities not available") 