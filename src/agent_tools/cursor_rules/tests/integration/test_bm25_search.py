#!/usr/bin/env python3
"""Test BM25 keyword search functionality."""
import pytest
import pytest_asyncio
from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    bm25_keyword_search
)
import os

@pytest.mark.asyncio
async def test_bm25_keyword_search():
    """Test the BM25 keyword search functionality with stemming."""
    # Setup test environment
    config = {
        "arango": {
            "hosts": ["http://localhost:8529"],
            "username": "root",
            "password": "openSesame"
        }
    }
    
    # Connect to database
    db = setup_cursor_rules_db(config)
    
    # Test different keyword variations to demonstrate stemming
    test_queries = [
        "database", 
        "databases",
        "connect", 
        "connection",
        "connecting",
        "test", 
        "testing",
        "tests"
    ]
    
    # Run each query
    print("\n=== BM25 Keyword Search Testing ===")
    print("Testing stemming functionality with different word forms")
    print("---------------------------------------")
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = await bm25_keyword_search(db, query)
        
        if results:
            print(f"Found {len(results)} results:")
            for result in results:
                print(f"- Score: {result['score']:.2f}")
                print(f"  Title: {result['rule']['title']}")
                print(f"  Description: {result['rule']['description'][:100]}...")
        else:
            print("No results found")
    
    # Test a more complex query with multiple words
    complex_query = "database connections and operations"
    print(f"\n--- Complex Query: '{complex_query}' ---")
    results = await bm25_keyword_search(db, complex_query)
    
    if results:
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"- Score: {result['score']:.2f}")
            print(f"  Title: {result['rule']['title']}")
            print(f"  Description: {result['rule']['description'][:100]}...")
    else:
        print("No results found")
    
    print("\n=== BM25 Search Testing Complete ===")

if __name__ == "__main__":
    test_bm25_keyword_search() 