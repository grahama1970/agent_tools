#!/usr/bin/env python3
"""
Test common database queries.

Documentation References:
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
- python-arango: https://docs.python-arango.com/
"""

import os
import pytest
import pytest_asyncio
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    query_by_rule_number,
    query_by_title,
    query_by_description,
    bm25_keyword_search,
    semantic_search,
    hybrid_search,
    generate_embedding
)

from agent_tools.cursor_rules.scenarios.common_queries import (
    store_scenario,
    get_scenario_by_title,
    list_all_scenarios,
    search_scenarios,
    import_scenarios_from_file
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample scenarios for testing
SAMPLE_SCENARIOS = [
    {
        "title": "Find method signature",
        "description": "How to find the correct signature for a specific method in a package",
        "query_example": "What parameters does requests.get accept?",
        "expected_result_format": "Method signature with parameter types and documentation",
        "priority": 1,
        "category": "method_usage"
    },
    {
        "title": "Debug error message",
        "description": "Finding solutions for specific error messages",
        "query_example": "How to fix 'ImportError: No module named X'",
        "expected_result_format": "Error cause and step-by-step resolution",
        "priority": 1,
        "category": "error_resolution"
    },
    {
        "title": "Implementation pattern",
        "description": "Finding the correct implementation pattern for a specific task",
        "query_example": "What's the right way to implement async file reading?",
        "expected_result_format": "Code snippet with explanation and documentation references",
        "priority": 2,
        "category": "implementation_pattern"
    },
    {
        "title": "Find Correct Usage of asyncio.to_thread",
        "description": "How to properly use asyncio.to_thread with ArangoDB operations",
        "query_example": "What's the correct pattern for using asyncio.to_thread with ArangoDB?",
        "expected_result_format": "Code pattern with examples and common pitfalls",
        "priority": 1,
        "category": "async_patterns"
    },
    {
        "title": "Package Version Compatibility",
        "description": "Finding compatible package versions and dependency requirements",
        "query_example": "Which version of pytest-asyncio works with Python 3.10?",
        "expected_result_format": "Version compatibility matrix with known issues",
        "priority": 2,
        "category": "dependency_management"
    },
    {
        "title": "Testing Best Practices",
        "description": "Guidelines for writing effective tests",
        "query_example": "How should I structure pytest fixtures for database testing?",
        "expected_result_format": "Best practices with examples and rationale",
        "priority": 1,
        "category": "testing"
    }
]

# Sample rules for testing
SAMPLE_RULES = [
    {
        "rule_number": "001",
        "title": "Code Advice Rules",
        "description": "Project Code Advice Rules for AI Code Generation",
        "content": "# Code Advice Rules\n\nThese rules must be followed..."
    }
]

@pytest_asyncio.fixture(scope="module")
async def db(event_loop):
    """Set up a test database for scenarios."""
    logger.debug("Setting up database connection...")
    config = {
        'arango': {
            'hosts': 'http://localhost:8529',
            'username': 'root',
            'password': 'openSesame'
        }
    }
    try:
        logger.debug("Attempting to connect to ArangoDB...")
        db_instance = await asyncio.to_thread(setup_cursor_rules_db, config, db_name='cursor_rules_test_scenarios')
        logger.debug("Successfully connected to ArangoDB")
        
        # Set up rules collection and insert sample rules
        collections = await asyncio.to_thread(db_instance.collections)
        collection_names = [c['name'] for c in collections]
        
        if 'rules' not in collection_names:
            logger.debug("Creating rules collection...")
            await asyncio.to_thread(db_instance.create_collection, 'rules')
        
        rules_collection = db_instance.collection('rules')
        await asyncio.to_thread(rules_collection.truncate)
        
        for rule in SAMPLE_RULES:
            logger.debug(f"Inserting rule: {rule['rule_number']}")
            await asyncio.to_thread(rules_collection.insert, rule)
        
        yield db_instance
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {str(e)}")
        raise

@pytest_asyncio.fixture
async def scenarios_collection(db):
    """Create and populate the scenarios collection."""
    collections = await asyncio.to_thread(db.collections)
    collection_names = [c['name'] for c in collections]
    
    if 'query_scenarios' not in collection_names:
        await asyncio.to_thread(db.create_collection, 'query_scenarios')
    
    collection = db.collection('query_scenarios')
    await asyncio.to_thread(collection.truncate)
    
    for scenario in SAMPLE_SCENARIOS:
        # Generate embedding for the scenario
        scenario_text = f"{scenario['title']} {scenario['description']} {scenario['query_example']}"
        embedding_result = generate_embedding(scenario_text)
        scenario['embedding'] = embedding_result['embedding']
        scenario['embedding_metadata'] = embedding_result['metadata']
        
        # Insert scenario with embedding
        await asyncio.to_thread(collection.insert, scenario)
    
    yield collection

@pytest.mark.asyncio
async def test_store_query_scenarios(scenarios_collection):
    """Test that query scenarios can be stored in the database."""
    count = await asyncio.to_thread(scenarios_collection.count)
    assert count == len(SAMPLE_SCENARIOS), "All scenarios should be stored in the database"
    
    cursor = await asyncio.to_thread(scenarios_collection.find, {"title": "Find method signature"})
    results = await asyncio.to_thread(list, cursor)
    
    assert len(results) == 1, "Should find exactly one scenario with this title"
    assert results[0]["priority"] == 1, "Priority should match the original data"

@pytest.mark.asyncio
async def test_search_scenarios_by_keyword(db, scenarios_collection):
    """Test that we can search scenarios using BM25 keyword search."""
    results = await bm25_keyword_search(db, "error", collection_name="query_scenarios")
    assert len(results) > 0, "Should find at least one scenario related to errors"
    
    found_error_scenario = any(
        "Debug error message" in result["rule"].get("title", "")
        for result in results
    )
    assert found_error_scenario, "Should find the 'Debug error message' scenario"

@pytest.mark.asyncio
async def test_scenarios_have_required_fields(scenarios_collection):
    """Test that all scenarios have the required fields for effective retrieval."""
    cursor = await asyncio.to_thread(scenarios_collection.all)
    scenarios = await asyncio.to_thread(list, cursor)
    
    required_fields = ["title", "description", "query_example", "expected_result_format", "category"]
    for scenario in scenarios:
        for field in required_fields:
            assert field in scenario, f"Scenario '{scenario.get('title', 'Unknown')}' is missing required field '{field}'"

@pytest.mark.asyncio
async def test_query_by_rule_number(db):
    """Test querying rules by their rule number."""
    rule_number = "001"
    expected_title = "Code Advice Rules"
    result = await query_by_rule_number(db, rule_number)
    assert result is not None, "Should find a rule with number 001"
    assert result['title'] == expected_title, "Rule title should match"

@pytest.mark.asyncio
async def test_query_by_title(db):
    """Test querying rules by their title."""
    title = "Code Advice Rules"
    expected_rule_number = "001"
    result = await query_by_title(db, title)
    assert result is not None, "Should find a rule with title 'Code Advice Rules'"
    assert result['rule_number'] == expected_rule_number, "Rule number should match"

@pytest.mark.asyncio
async def test_query_by_description(db):
    """Test querying rules by their description."""
    description = "Project Code Advice Rules for AI Code Generation"
    expected_rule_number = "001"
    result = await query_by_description(db, description)
    assert result is not None, "Should find a rule with the given description"
    assert result['rule_number'] == expected_rule_number, "Rule number should match"

@pytest.mark.asyncio
async def test_search_async_patterns(db, scenarios_collection):
    """Test finding async-related patterns and best practices."""
    results = await hybrid_search(db, "asyncio.to_thread pattern", collection_name="query_scenarios")
    assert len(results) > 0, "Should find scenarios related to asyncio patterns"
    
    found_async_scenario = any(
        "asyncio.to_thread" in result[0].get("title", "")
        for result in results
    )
    assert found_async_scenario, "Should find the asyncio.to_thread usage scenario"

@pytest.mark.asyncio
async def test_search_testing_practices(db, scenarios_collection):
    """Test finding testing best practices."""
    results = await hybrid_search(db, "pytest fixture database testing", collection_name="query_scenarios")
    assert len(results) > 0, "Should find scenarios related to testing practices"
    
    found_testing_scenario = any(
        "Testing Best Practices" in result[0].get("title", "")
        for result in results
    )
    assert found_testing_scenario, "Should find the testing best practices scenario"

@pytest.mark.asyncio
async def test_search_package_compatibility(db, scenarios_collection):
    """Test finding package version compatibility information."""
    results = await hybrid_search(db, "pytest-asyncio Python 3.10 compatibility", collection_name="query_scenarios")
    assert len(results) > 0, "Should find scenarios related to package compatibility"
    
    found_compatibility_scenario = any(
        "Package Version Compatibility" in result[0].get("title", "")
        for result in results
    )
    assert found_compatibility_scenario, "Should find the package compatibility scenario"

@pytest.mark.asyncio
async def test_scenario_categories(scenarios_collection):
    """Test that scenarios are properly categorized for effective retrieval."""
    cursor = await asyncio.to_thread(scenarios_collection.all)
    scenarios = await asyncio.to_thread(list, cursor)
    
    expected_categories = {
        "method_usage",
        "error_resolution",
        "implementation_pattern",
        "async_patterns",
        "dependency_management",
        "testing"
    }
    
    found_categories = {scenario["category"] for scenario in scenarios}
    assert found_categories == expected_categories, "All expected categories should be present"
    
    # Verify priority distribution
    high_priority_scenarios = [s for s in scenarios if s["priority"] == 1]
    assert len(high_priority_scenarios) >= 3, "Should have at least 3 high-priority scenarios"

# Additional test scenarios can be added here following the same pattern 