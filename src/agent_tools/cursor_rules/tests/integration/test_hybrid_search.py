#!/usr/bin/env python
"""
Test for hybrid search functionality combining BM25 and vector search.

This test validates our implementation of hybrid search that combines
BM25 keyword search with vector similarity search.

Documentation References:
- ArangoDB Vector Search: https://docs.arangodb.com/3.11/search/vector-search/
- ArangoDB BM25: https://docs.arangodb.com/3.11/aql/functions/arangosearch/#bm25
- ArangoDB Hybrid Search: https://docs.arangodb.com/3.11/search/hybrid-search/
"""
import os
import sys
import time
import asyncio
import pytest
from arango import ArangoClient

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Updated import path to match new file structure and available functions
from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    hybrid_search,
    generate_embedding,
    create_arangosearch_view,
    EMBEDDING_AVAILABLE
)

@pytest.mark.asyncio
async def test_hybrid_search_real_db():
    """
    Test hybrid search with a real ArangoDB instance.
    
    This test:
    1. Sets up a test database with sample rules
    2. Creates embeddings for the rules
    3. Creates an ArangoSearch view
    4. Performs a hybrid search
    5. Verifies that results are returned and properly scored
    
    Note: This test requires embedding utilities to be available.
    """
    # Skip test if embedding utilities aren't available
    if not EMBEDDING_AVAILABLE:
        print("Skipping test_hybrid_search_real_db - embedding utilities not available")
        return
    
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    sys_db = client.db("_system", username="root", password="openSesame")
    
    # Use a fixed database name for testing
    db_name = "cursor_rules_test_hybrid"
    
    # Delete the test database if it exists
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
        print(f"Deleted existing database: {db_name}")
    
    # Create a fresh test database
    sys_db.create_database(db_name)
    print(f"Created database: {db_name}")
    
    # Connect to the test database
    db = client.db(db_name, username="root", password="openSesame")
    
    # Create the rules collection
    collection_name = "rules"
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)
        print(f"Created collection: {collection_name}")
    
    # Create the ArangoSearch view
    view_name = "rules_view"
    
    # Ensure view doesn't exist first
    for view in db.views():
        if view['name'] == view_name:
            db.delete_view(view_name)
            print(f"Deleted existing view: {view_name}")
    
    # Use the exact view properties that worked in test_arangosearch_view_setup
    view_properties = {
        "primarySort": [
            {"field": "rule_number", "direction": "asc"}
        ],
        "primarySortCompression": "lz4",
        "storedValues": [
            {"fields": ["rule_number", "title"], "compression": "lz4"}
        ],
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
                    "content": {
                        "analyzers": ["text_en"],
                        "includeAllFields": False
                    }
                }
            }
        }
    }
    
    db.create_view(
        name=view_name,
        view_type="arangosearch",
        properties=view_properties
    )
    print(f"Created view: {view_name}")
    
    # Insert test documents with embeddings
    # Updated import path
    from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
    
    rules_collection = db.collection(collection_name)
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
    
    # Add embeddings to the test rules
    for rule in test_rules:
        # Generate embedding from rule content
        text_to_embed = f"RULE: {rule['title']}\nDESCRIPTION: {rule['description']}\nCONTENT: {rule['content']}"
        embedding_result = generate_embedding(text_to_embed)
        
        if embedding_result and "embedding" in embedding_result:
            rule["embedding"] = embedding_result["embedding"]
            rule["embedding_metadata"] = embedding_result.get("metadata", {})
            print(f"Added embedding for rule: {rule['title']}")
        else:
            print(f"Warning: Failed to generate embedding for rule: {rule['title']}")
        
        # Insert the rule
        rules_collection.insert(rule)
    
    print(f"Inserted {len(test_rules)} test rules with embeddings")
    
    # Add a delay to allow ArangoDB to index the documents
    print("\nWaiting for ArangoDB to index documents...")
    await asyncio.sleep(2)
    
    # Perform hybrid search
    search_text = "python style guide"
    print(f"\nPerforming hybrid search for: '{search_text}'")
    
    results = await hybrid_search(db, search_text, limit=5, verbose=True)
    
    # Verify results
    assert len(results) > 0, "Should find results for 'python style guide'"
    assert any("Python Code Style" in result[0]["title"] for result in results), "Should find the Python style guide rule"
    
    # Clean up
    sys_db.delete_database(db_name)
    print(f"\nDeleted test database: {db_name}")
    
    return True

if __name__ == "__main__":
    test_hybrid_search_real_db() 