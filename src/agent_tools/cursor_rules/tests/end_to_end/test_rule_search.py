#!/usr/bin/env python3
"""
Test file for the rule_search module.

This file tests the functionality of the rule_search module, which provides
functions to search for rules related to user queries or reasoning tasks.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any, List
from arango import ArangoClient

from agent_tools.cursor_rules.utils.rule_search import (
    search_related_rules,
    format_rules_for_agent,
    RuleSearchResult,
    RuleSearchCache,
    get_related_rules,
    search_for_user_query,
    search_for_reasoning_task
)

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def db():
    """Setup test database."""
    client = ArangoClient(hosts='http://localhost:8529')
    # Connect to system database for management duties
    sys_db = client.db('_system', username='root', password='openSesame')
    test_db_name = "TestRuleSearchDB"
    
    # Create test database if it doesn't exist
    if not sys_db.has_database(test_db_name):
        sys_db.create_database(test_db_name)
    
    # Connect to the test database
    db = client.db(test_db_name, username='root', password='openSesame')
    
    # Create or clear the 'rules' collection
    if db.has_collection("rules"):
        db.delete_collection("rules")
    rules_collection = db.create_collection("rules")
    
    # Create test documents
    test_docs = [
        {
            "_key": "001",
            "rule_number": "001",
            "title": "Async Database Operations",
            "description": "Always use asyncio.to_thread for database operations",
            "content": "When working with synchronous database drivers like ArangoDB's Python driver, always use asyncio.to_thread to prevent blocking the event loop. Example: await asyncio.to_thread(database_function, var1, var2, varN)",
            "priority": 1
        },
        {
            "_key": "002",
            "rule_number": "002",
            "title": "Error Handling Patterns",
            "description": "Proper error handling for async operations",
            "content": "Always use try/except blocks around database operations and handle specific exceptions appropriately.",
            "priority": 2
        },
        {
            "_key": "003",
            "rule_number": "003",
            "title": "Code Documentation",
            "description": "Guidelines for documenting code",
            "content": "All functions should have docstrings explaining their purpose, parameters, and return values.",
            "priority": 3
        }
    ]
    
    # Insert test documents
    for doc in test_docs:
        rules_collection.insert(doc)
    
    yield db
    
    # Teardown: drop the test database
    sys_db.delete_database(test_db_name)

@pytest.mark.asyncio
async def test_search_related_rules(db):
    """Test searching for rules related to a query."""
    # Search for rules related to async operations
    result = await search_related_rules(db, "async database operations", limit=5)
    
    # Check that we got a RuleSearchResult
    assert isinstance(result, RuleSearchResult)
    
    # Check that we got some results
    assert len(result.rules) > 0
    
    # Check that the query was stored
    assert result.query == "async database operations"
    
    # Check that we can get rule titles
    titles = result.get_rule_titles()
    assert len(titles) > 0
    
    # Check that we can get rule descriptions
    descriptions = result.get_rule_descriptions()
    assert len(descriptions) > 0

@pytest.mark.asyncio
async def test_format_rules_for_agent(db):
    """Test formatting rules for presentation to the agent."""
    # Search for rules
    result = await search_related_rules(db, "async database operations", limit=5)
    
    # Format the rules
    formatted = format_rules_for_agent(result)
    
    # Check that the formatted string contains the query
    assert "async database operations" in formatted
    
    # Check that the formatted string contains rule titles
    for title in result.get_rule_titles():
        assert title in formatted
    
    # Check that the formatted string contains rule descriptions
    for description in result.get_rule_descriptions():
        assert description in formatted

@pytest.mark.asyncio
async def test_rule_search_cache():
    """Test the rule search cache."""
    # Create a cache
    cache = RuleSearchCache(max_size=2)
    
    # Create some test results
    result1 = RuleSearchResult("query1", [{"title": "Rule 1"}])
    result2 = RuleSearchResult("query2", [{"title": "Rule 2"}])
    result3 = RuleSearchResult("query3", [{"title": "Rule 3"}])
    
    # Add results to cache
    cache.add("query1", result1)
    cache.add("query2", result2)
    
    # Check that we can get results from cache
    assert cache.get("query1") == result1
    assert cache.get("query2") == result2
    
    # Add another result, which should evict the oldest one
    cache.add("query3", result3)
    
    # Check that the oldest result was evicted
    assert cache.get("query1") is None
    assert cache.get("query2") == result2
    assert cache.get("query3") == result3
    
    # Clear the cache
    cache.clear()
    
    # Check that the cache is empty
    assert cache.get("query2") is None
    assert cache.get("query3") is None

@pytest.mark.asyncio
async def test_get_related_rules(db):
    """Test getting rules related to a query, using cache if available."""
    # Get rules for a query
    result1 = await get_related_rules(db, "async database operations", use_cache=True, limit=5)
    
    # Get rules for the same query again, which should use the cache
    result2 = await get_related_rules(db, "async database operations", use_cache=True, limit=5)
    
    # Check that we got the same results
    assert result1.query == result2.query
    assert len(result1.rules) == len(result2.rules)
    
    # Get rules for the same query without using the cache
    result3 = await get_related_rules(db, "async database operations", use_cache=False, limit=5)
    
    # Check that we got the same query but potentially different results
    assert result1.query == result3.query

@pytest.mark.asyncio
async def test_search_for_user_query(db):
    """Test searching for rules related to a user query."""
    # Search for rules related to a user query
    formatted = await search_for_user_query(db, "How should I handle database operations?", limit=5)
    
    # Check that the formatted string contains relevant information
    assert "Rules related to 'How should I handle database operations?'" in formatted
    assert "Async Database Operations" in formatted

@pytest.mark.asyncio
async def test_search_for_reasoning_task(db):
    """Test searching for rules related to a reasoning task."""
    # Search for rules related to a reasoning task
    formatted = await search_for_reasoning_task(db, "I need to implement error handling for database operations", limit=5)
    
    # Check that the formatted string contains relevant information
    assert "Rules related to 'I need to implement error handling for database operations'" in formatted
    assert "Error Handling Patterns" in formatted 