#!/usr/bin/env python3
"""
Test async patterns with ArangoDB.

This module tests the async patterns used with ArangoDB operations.

Key Lessons:
1. ArangoDB driver is synchronous - all DB operations must be wrapped in asyncio.to_thread
2. Never mix sync and async code directly
3. Database connections should be properly managed with async context managers
4. Always clean up database resources properly
5. CRITICAL: Async generators must be properly awaited!
6. CRITICAL: pytest-asyncio fixtures require special handling!

Documentation References:
- ArangoDB Python Driver: https://docs.python-arango.com/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import os
import pytest
import asyncio
import pytest_asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List, Optional
from pathlib import Path

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import CollectionCreateError, GraphCreateError, ViewCreateError

from agent_tools.cursor_rules.core.cursor_rules import setup_cursor_rules_db
from agent_tools.cursor_rules.core.ai_knowledge_db import (
    setup_ai_knowledge_db,
    create_document_collections,
    create_edge_collections,
    create_named_graphs,
    create_views,
    create_analyzers,
    store_schema_doc,
    get_schema_doc
)

# Test constants
TEST_COLLECTION = 'async_pattern_test'
TEST_DB_NAME = f"cursor_rules_async_test_{int(time.time())}"
TEST_CONFIG = {
    "arango": {
        "hosts": "http://localhost:8529",
        "username": "root",
        "password": "openSesame"
    }
}

@asynccontextmanager
async def get_db():
    """
    Async context manager for database connection.
    
    Key Pattern:
    1. Use asynccontextmanager for proper async resource management
    2. Ensure cleanup in both success and failure cases
    3. Use asyncio.to_thread for ALL database operations
    """
    # Pass the required config parameter to setup_cursor_rules_db
    db = await asyncio.to_thread(setup_cursor_rules_db, TEST_CONFIG, db_name=TEST_DB_NAME)
    try:
        # Check if db is None before proceeding
        if db is None:
            pytest.skip("Could not connect to database or setup failed")
            yield None
            return
            
        # Clean up any existing test collection using to_thread
        exists = await asyncio.to_thread(db.has_collection, TEST_COLLECTION)
        if exists:
            await asyncio.to_thread(db.delete_collection, TEST_COLLECTION)
        yield db
    finally:
        # Cleanup after tests using to_thread - only if db is not None
        if db is not None:
            exists = await asyncio.to_thread(db.has_collection, TEST_COLLECTION)
            if exists:
                await asyncio.to_thread(db.delete_collection, TEST_COLLECTION)

@pytest_asyncio.fixture
async def db():
    """
    Setup test database using async context manager.
    
    Key Pattern:
    1. Use pytest_asyncio.fixture for proper async fixture handling
    2. Yield the database connection from the context manager
    3. Use async with for proper resource management
    """
    async with get_db() as database:
        yield database

async def create_test_collection(db) -> None:
    """
    Demonstrate proper async collection creation pattern.
    
    Key Pattern:
    1. Use asyncio.to_thread for the synchronous database operation
    2. Handle the operation atomically
    """
    await asyncio.to_thread(db.create_collection, TEST_COLLECTION)

async def insert_test_document(db, doc: dict) -> str:
    """
    Demonstrate proper async document insertion pattern.
    
    Key Pattern:
    1. Use asyncio.to_thread for the synchronous database operation
    2. Return values should be captured directly
    """
    collection = await asyncio.to_thread(db.collection, TEST_COLLECTION)
    result = await asyncio.to_thread(collection.insert, doc)
    return result['_key']

async def get_test_document(db, key: str) -> dict:
    """
    Demonstrate proper async document retrieval pattern.
    
    Key Pattern:
    1. Use asyncio.to_thread for the synchronous database operation
    2. Handle potential None returns properly
    """
    collection = await asyncio.to_thread(db.collection, TEST_COLLECTION)
    doc = await asyncio.to_thread(collection.get, key)
    if doc is None:
        raise ValueError(f"No document found with key {key}")
    return doc

@pytest.mark.asyncio
async def test_collection_creation(db):
    """Test proper async pattern for collection creation."""
    # Verify collection doesn't exist
    exists = await asyncio.to_thread(db.has_collection, TEST_COLLECTION)
    assert not exists, "Collection should not exist initially"
    
    # Create collection
    await create_test_collection(db)
    
    # Verify creation
    exists = await asyncio.to_thread(db.has_collection, TEST_COLLECTION)
    assert exists, "Collection should exist after creation"

@pytest.mark.asyncio
async def test_document_operations(db):
    """Test proper async patterns for document operations."""
    # Setup
    await create_test_collection(db)
    
    # Test document insertion
    doc = {"test_field": "test_value"}
    key = await insert_test_document(db, doc)
    assert key, "Should return a valid document key"
    
    # Test document retrieval
    stored = await get_test_document(db, key)
    assert stored["test_field"] == doc["test_field"]
    
    # Test non-existent document
    with pytest.raises(ValueError) as exc_info:
        await get_test_document(db, "nonexistent")
    assert "No document found with key" in str(exc_info.value)

@pytest.mark.asyncio
async def test_concurrent_operations(db):
    """
    Test proper async patterns for concurrent operations.
    
    Key Pattern:
    1. Use asyncio.gather for concurrent operations
    2. Each operation should be properly wrapped in to_thread
    """
    # Setup
    await create_test_collection(db)
    
    # Create multiple documents concurrently
    docs = [
        {"index": i, "value": f"test_{i}"} 
        for i in range(5)
    ]
    
    # Use gather for concurrent insertion
    keys = await asyncio.gather(
        *[insert_test_document(db, doc) for doc in docs]
    )
    assert len(keys) == len(docs), "Should insert all documents"
    
    # Use gather for concurrent retrieval
    stored_docs = await asyncio.gather(
        *[get_test_document(db, key) for key in keys]
    )
    
    # Verify all documents
    assert len(stored_docs) == len(docs)
    for i, doc in enumerate(stored_docs):
        assert doc["index"] == i
        assert doc["value"] == f"test_{i}" 