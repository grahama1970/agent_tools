#!/usr/bin/env python3
"""
Test file for managing AI retrieval scenarios in the cursor rules database.

Documentation References:
- ArangoDB Python Driver: https://docs.python-arango.com/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

import asyncio
import pytest
import pytest_asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from agent_tools.cursor_rules.core.cursor_rules import setup_cursor_rules_db
from agent_tools.cursor_rules.scenarios.scenario_management import (
    store_scenario,
    get_scenario_by_title,
    list_all_scenarios,
    search_scenarios,
    import_scenarios_from_file,
    validate_scenario,
    ensure_collection,
    COLLECTION_NAME,
    get_scenario,
    update_scenario,
    insert_sample_scenarios,
    hybrid_search
)
import os
from arango import ArangoClient

@pytest.fixture(scope="function")
def db():
    """Setup test database."""
    config = {"db_name": "TestCursorRulesDB", "username": "root", "password": "openSesame", "host": "http://localhost:8529"}
    db = setup_cursor_rules_db(config)
    # Clean up any existing test collection
    if db.has_collection(COLLECTION_NAME):
        db.delete_collection(COLLECTION_NAME)
    return db

@pytest.fixture
def sample_scenario():
    """Provide a valid test scenario."""
    return {
        "title": "Common Code Pattern Query",
        "description": "AI needs to find the correct pattern for implementing async operations",
        "query_example": "What's the best pattern for handling async database operations?",
        "expected_result_type": "code_pattern",
        "validation_criteria": [
            "Must include error handling",
            "Must show proper async/await usage",
            "Should include transaction management"
        ],
        "priority": 1
    }

@pytest.mark.asyncio
async def test_ensure_collection(db):
    """Test ensuring collection exists."""
    # First ensure it doesn't exist
    if db.has_collection(COLLECTION_NAME):
        db.delete_collection(COLLECTION_NAME)
    
    # Test creating collection
    await ensure_collection(db)
    exists = db.has_collection(COLLECTION_NAME)
    assert exists, "Collection should exist after ensure_collection"
    
    # Test idempotency - should not error when called again
    await ensure_collection(db)
    assert db.has_collection(COLLECTION_NAME)

@pytest.mark.asyncio
async def test_validate_scenario_valid(sample_scenario):
    """Test validating a valid scenario."""
    await validate_scenario(sample_scenario)  # Should not raise

@pytest.mark.asyncio
async def test_validate_scenario_invalid():
    """Test validating an invalid scenario."""
    invalid_scenario = {
        "title": "Incomplete Scenario",
        "description": "Missing required fields"
    }
    
    with pytest.raises(ValueError) as exc_info:
        await validate_scenario(invalid_scenario)
    assert "Missing required fields" in str(exc_info.value)

@pytest.mark.asyncio
async def test_store_and_get_scenario(db, sample_scenario):
    """Test storing and retrieving a scenario."""
    # Store scenario
    key = await store_scenario(db, sample_scenario)
    assert key, "Should return a valid document key"
    
    # Retrieve and verify
    stored = await get_scenario(db, key)
    assert stored["title"] == sample_scenario["title"]
    assert stored["description"] == sample_scenario["description"]
    assert "embedding" in stored, "Should include generated embedding"

@pytest.mark.asyncio
async def test_search_scenarios(db, sample_scenario):
    """Test searching for scenarios."""
    # Store a test scenario
    await store_scenario(db, sample_scenario)
    
    # Search with matching query
    results = await search_scenarios(db, "async database operations")
    assert len(results) > 0, "Should find matching scenario"
    assert any("async" in r["description"].lower() for r in results), "Should find scenario with 'async' in the description"
    
    # Search with non-matching query
    results = await search_scenarios(db, "quantum physics theories")
    assert len(results) == 0, "Should not find unrelated scenarios"

@pytest.mark.asyncio
async def test_update_scenario(db, sample_scenario):
    """Test updating a scenario."""
    # Store initial scenario
    key = await store_scenario(db, sample_scenario)
    
    # Update scenario
    updates = {
        "title": "Updated Pattern Query",
        "priority": 2
    }
    
    await update_scenario(db, key, updates)
    
    # Verify updates
    updated = await get_scenario(db, key)
    assert updated["title"] == updates["title"]
    assert updated["priority"] == updates["priority"]
    assert updated["description"] == sample_scenario["description"]  # Unchanged field
    assert "embedding" in updated, "Should maintain embedding"

@pytest.mark.asyncio
async def test_update_scenario_invalid_key(db):
    """Test updating a non-existent scenario."""
    with pytest.raises(ValueError) as exc_info:
        await update_scenario(db, "nonexistent", {"title": "New Title"})
    assert "No scenario found with key" in str(exc_info.value)

@pytest.fixture(scope="module")
def scenarios_collection():
    """Set up a test database and a scenarios collection, then tear it down after tests."""
    client = ArangoClient(hosts='http://localhost:8529')
    # Connect to system database for management duties
    sys_db = client.db('_system', username='root', password='openSesame')
    test_db_name = "TestScenarioDB"
    # Create test database if it doesn't exist
    if not sys_db.has_database(test_db_name):
        sys_db.create_database(test_db_name)
    # Connect to the test database
    db = client.db(test_db_name, username='root', password='openSesame')
    
    # Create or clear the 'scenarios' collection
    if db.has_collection("scenarios"):
        collection = db.collection("scenarios")
        collection.truncate()
    else:
        collection = db.create_collection("scenarios")
    
    yield collection
    
    # Teardown: drop the test database
    sys_db.delete_database(test_db_name)

@pytest.mark.asyncio
async def test_insert_scenarios(scenarios_collection):
    """Test that sample scenarios are correctly inserted into the collection."""
    # Determine the path to sample_scenarios.json file
    scenarios_file_path = os.path.join(os.path.dirname(__file__), "..", "sample_scenarios.json")
    with open(scenarios_file_path, "r") as f:
        scenarios = json.load(f)

    # Insert scenarios into the collection
    inserted_count = await insert_sample_scenarios(scenarios, scenarios_collection)

    # Assert that the number of inserted documents matches the number in the file
    assert inserted_count == len(scenarios)

    # Verify by checking the document count in the collection
    count = await asyncio.to_thread(scenarios_collection.count)
    assert count == len(scenarios)

@pytest.mark.asyncio
async def test_hybrid_search(scenarios_collection):
    """Test hybrid search functionality by querying for a keyword present in one of the scenarios."""
    # Get the database directly
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('TestScenarioDB', username='root', password='openSesame')
    
    # Create a rules collection with the same data as scenarios
    if db.has_collection("rules"):
        db.delete_collection("rules")
    rules_collection = db.create_collection("rules")
    
    # Create a test document with the keyword "asyncio"
    test_doc = {
        "_key": "test_asyncio",
        "title": "Find Correct Usage of asyncio.to_thread",
        "description": "AI needs to find the correct pattern for implementing asyncio.to_thread operations",
        "query_example": "What's the best pattern for handling async database operations with asyncio.to_thread?",
        "expected_result_type": "code_pattern",
        "validation_criteria": [
            "Must include error handling",
            "Must show proper async/await usage",
            "Should include transaction management"
        ],
        "priority": 1
    }
    rules_collection.insert(test_doc)
    
    # Use a keyword that we expect to find
    search_term = "asyncio"
    results = await hybrid_search(search_term, rules_collection, db, limit=5)
    
    # Print the actual results for debugging
    print(f"Raw results: {results}")
    
    # Check that we got some results
    assert len(results) > 0, "Should find at least one result"
    
    # For this test, we'll just check that we got some results
    assert results, "Should find at least one result" 