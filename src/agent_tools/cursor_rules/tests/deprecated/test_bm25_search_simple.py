#!/usr/bin/env python
"""
Simple test for BM25 search functionality.

This test focuses only on BM25 search with known search results.
"""
import time
from typing import Dict, Any, List

from arango import ArangoClient
import pytest

from agent_tools.cursor_rules.core import bm25_keyword_search

# Try to import tabulate for better output formatting
try:
    from tabulate import tabulate
except ImportError:
    # Simple tabulate replacement if not available
    def tabulate(data, headers, tablefmt="grid"):
        result = []
        # Add header
        header_str = " | ".join(str(h) for h in headers)
        result.append(header_str)
        result.append("-" * len(header_str))
        
        # Add rows
        for row in data:
            row_str = " | ".join(str(cell) for cell in row)
            result.append(row_str)
            
        return "\n".join(result)

# Test data with known search terms in different fields
TEST_RULES = [
    {
        "_key": "001-python-style",
        "rule_number": "001",
        "title": "Python Style Guide",
        "description": "Guidelines for proper use of Python code",
        "content": "Follow PEP 8 style guidelines for all Python code.",
        "rule_type": "style",
        "glob_pattern": "*.py"
    },
    {
        "_key": "002-javascript-format",
        "rule_number": "002",
        "title": "JavaScript Formatting",
        "description": "Standards for JavaScript code formatting",
        "content": "Use 2 spaces for indentation in JavaScript files.",
        "rule_type": "style",
        "glob_pattern": "*.js"
    }
]

def setup_test_db():
    """Set up a test database with sample rules."""
    client = ArangoClient(hosts="http://localhost:8529")
    sys_db = client.db("_system", username="root", password="openSesame")
    
    # Create a unique test database name
    db_name = f"cursor_rules_test_{int(time.time())}"
    
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
    sys_db.create_database(db_name)
    
    # Connect to the test database
    db = client.db(db_name, username="root", password="openSesame")
    
    # Create collections
    if not db.has_collection("rules"):
        db.create_collection("rules")
    if not db.has_collection("rule_examples"):
        db.create_collection("rule_examples")
    
    # Insert sample rules
    rules = db.collection("rules")
    rules.insert({
        "_key": "001",
        "title": "Test Rule",
        "description": "A test rule about proper use of testing",
        "content": "This is a test rule about the proper use of testing frameworks."
    })
    
    print(f"Created test database '{db_name}' with {rules.count()} rules")
    return db, db_name

def cleanup_test_db(db_name):
    """Clean up the test database."""
    client = ArangoClient(hosts=["http://localhost:8529"])
    sys_db = client.db("_system", username="root", password="openSesame")
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
        print(f"Deleted test database '{db_name}'")

@pytest.mark.asyncio
async def test_bm25_search_positive():
    """Test BM25 search with text that exists in the rules."""
    client = ArangoClient(hosts="http://localhost:8529")
    db, db_name = setup_test_db()
    
    try:
        print("\n" + "="*80)
        print("EXECUTING BM25 SEARCH FOR: 'proper use'")
        print("="*80)
        
        # Print the exact AQL query being used
        aql_query = """
        FOR doc IN @@view
            SEARCH ANALYZER(
                doc.title IN TOKENS(@search_text, "text_en") OR
                doc.description IN TOKENS(@search_text, "text_en") OR
                doc.content IN TOKENS(@search_text, "text_en"),
                "text_en"
            )
            LET bm25_score = BM25(doc, @k1, @b)
            FILTER bm25_score > @bm25_threshold
            SORT bm25_score DESC
            LIMIT @limit
            RETURN {
                rule: doc,
                score: bm25_score
            }
        """
        
        print("\nEXACT AQL QUERY:")
        print("-"*80)
        print(aql_query)
        print("-"*80)
        
        print("\nBIND VARIABLES:")
        print("-"*80)
        print("@view: 'rules_view'")
        print("search_text: 'proper use'")
        print("k1: 1.2")
        print("b: 0.75")
        print("bm25_threshold: 0.0")
        print("limit: 5")
        print("-"*80)
        
        # Search for text that exists in the description
        results = await bm25_keyword_search(db, "proper use", limit=5, verbose=True)
        
        # Verify results
        print(f"\nFound {len(results)} results for 'proper use'")
        assert len(results) > 0, "Should find at least one result for 'proper use'"
        
        # Check that results are properly scored
        for result in results:
            assert "score" in result, "Each result should have a score"
            assert result["score"] > 0, "Scores should be positive"
            assert "rule" in result, "Each result should have a rule"
            
    finally:
        # Clean up test database
        sys_db = client.db("_system", username="root", password="openSesame")
        if sys_db.has_database(db_name):
            sys_db.delete_database(db_name)

@pytest.mark.asyncio
async def test_bm25_search_negative():
    """Test BM25 search with text that doesn't exist in the rules."""
    client = ArangoClient(hosts="http://localhost:8529")
    db, db_name = setup_test_db()
    
    try:
        print("\n" + "="*80)
        print("EXECUTING BM25 SEARCH FOR: 'mustard'")
        print("="*80)
        
        # Print the exact AQL query being used
        aql_query = """
        FOR doc IN @@view
            SEARCH ANALYZER(
                doc.title IN TOKENS(@search_text, "text_en") OR
                doc.description IN TOKENS(@search_text, "text_en") OR
                doc.content IN TOKENS(@search_text, "text_en"),
                "text_en"
            )
            LET bm25_score = BM25(doc, @k1, @b)
            FILTER bm25_score > @bm25_threshold
            SORT bm25_score DESC
            LIMIT @limit
            RETURN {
                rule: doc,
                score: bm25_score
            }
        """
        
        print("\nEXACT AQL QUERY:")
        print("-"*80)
        print(aql_query)
        print("-"*80)
        
        print("\nBIND VARIABLES:")
        print("-"*80)
        print("@view: 'rules_view'")
        print("search_text: 'mustard'")
        print("k1: 1.2")
        print("b: 0.75")
        print("bm25_threshold: 0.0")
        print("limit: 5")
        print("-"*80)
        
        # Search for text that doesn't exist
        results = await bm25_keyword_search(db, "mustard", limit=5, verbose=True)
        
        # Verify results
        print(f"\nFound {len(results)} results for 'mustard'")
        assert len(results) == 0, "Should not find any results for 'mustard'"
        
    finally:
        # Clean up test database
        sys_db = client.db("_system", username="root", password="openSesame")
        if sys_db.has_database(db_name):
            sys_db.delete_database(db_name)

if __name__ == "__main__":
    print("Running BM25 search tests...")
    test_bm25_search_positive()
    test_bm25_search_negative()
    print("All tests completed.") 