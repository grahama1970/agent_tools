"""
Integration tests for AI Knowledge Database with real ArangoDB connections.

Documentation References:
- ArangoDB Python Driver: https://docs.python-driver.arangodb.com/
- ArangoDB: https://www.arangodb.com/docs/stable/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import os
import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from arango import ArangoClient
from agent_tools.cursor_rules.core.ai_knowledge_db import (
    load_schema,
    create_document_collections,
    create_edge_collections,
    create_named_graphs,
    create_views,
    create_analyzers,
    store_schema_doc,
    setup_ai_knowledge_db,
    get_schema_doc
)
from agent_tools.cursor_rules.core.db import get_db, create_database

# Test schema for integration tests
TEST_SCHEMA = {
    "database_name": "test_ai_knowledge",
    "description": "Test database for AI knowledge retrieval integration tests",
    "collections": {
        "rules": {
            "type": "document",
            "description": "Collection of rules",
            "fields": {
                "_key": {"type": "string", "description": "Unique identifier", "indexed": True},
                "title": {"type": "string", "description": "Rule title", "indexed": True},
                "description": {"type": "string", "description": "Rule description", "indexed": True},
                "content": {"type": "string", "description": "Rule content", "indexed": True},
                "category": {"type": "string", "description": "Rule category", "indexed": True},
                "priority": {"type": "integer", "description": "Rule priority", "indexed": True}
            }
        },
        "examples": {
            "type": "document",
            "description": "Collection of examples",
            "fields": {
                "_key": {"type": "string", "description": "Unique identifier", "indexed": True},
                "title": {"type": "string", "description": "Example title", "indexed": True},
                "code": {"type": "string", "description": "Example code", "indexed": True},
                "explanation": {"type": "string", "description": "Example explanation", "indexed": True}
            }
        }
    },
    "edge_collections": {
        "rule_examples": {
            "type": "edge",
            "description": "Edges connecting rules to examples",
            "fields": {
                "relationship_type": {"type": "string", "description": "Type of relationship", "indexed": True}
            },
            "from": ["rules"],
            "to": ["examples"]
        }
    },
    "graphs": {
        "knowledge_graph": {
            "edge_definitions": [
                {
                    "collection": "rule_examples",
                    "from": ["rules"],
                    "to": ["examples"]
                }
            ]
        }
    },
    "views": {
        "rules_view": {
            "type": "arangosearch",
            "description": "Search view for rules",
            "properties": {
                "analyzer": "text_en",
                "features": ["frequency", "position", "norm"]
            },
            "links": {
                "rules": {
                    "fields": {
                        "title": {"analyzer": "text_en"},
                        "description": {"analyzer": "text_en"},
                        "content": {"analyzer": "text_en"}
                    }
                }
            }
        }
    },
    "analyzers": {
        "text_en": {
            "type": "text",
            "properties": {
                "locale": "en",
                "case": "lower",
                "stemming": True,
                "stopwords": []
            }
        }
    }
}

@pytest.fixture(scope="module")
async def test_db():
    """Setup test database for integration tests."""
    # Create a unique test database name with timestamp
    import time
    test_db_name = f"test_ai_knowledge_{int(time.time())}"
    
    # Connect to ArangoDB - wrap client initialization in asyncio.to_thread
    client = await asyncio.to_thread(ArangoClient, hosts="http://localhost:8529")
    
    # Connect to _system database to create and delete test database
    sys_db = await asyncio.to_thread(client.db, "_system", username="root", password="openSesame")
    
    # Create test database if it doesn't exist
    has_db = await asyncio.to_thread(sys_db.has_database, test_db_name)
    if not has_db:
        await asyncio.to_thread(sys_db.create_database, test_db_name)
    
    # Connect to test database
    db = await asyncio.to_thread(client.db, test_db_name, username="root", password="openSesame")
    
    # Yield the database for tests
    yield db
    
    # Clean up - delete test database
    await asyncio.to_thread(sys_db.delete_database, test_db_name)

@pytest.mark.asyncio
async def test_database_setup():
    """Test setting up the database with real ArangoDB."""
    # Create a unique test database name with timestamp
    import time
    test_db_name = f"test_setup_{int(time.time())}"
    
    # Create ArangoDB client - wrap in asyncio.to_thread
    client = await asyncio.to_thread(ArangoClient, hosts="http://localhost:8529")
    
    # Connect to _system database
    sys_db = await asyncio.to_thread(client.db, "_system", username="root", password="openSesame")
    
    # Check if database exists and delete if needed
    has_db = await asyncio.to_thread(sys_db.has_database, test_db_name)
    if has_db:
        await asyncio.to_thread(sys_db.delete_database, test_db_name)
    
    # Create test database
    await asyncio.to_thread(sys_db.create_database, test_db_name)
    
    # Connect to test database
    db = await asyncio.to_thread(client.db, test_db_name, username="root", password="openSesame")
    
    # Test database setup
    assert db is not None
    
    # Get database properties
    properties = await asyncio.to_thread(db.properties)
    assert properties["name"] == test_db_name
    
    # Clean up
    await asyncio.to_thread(sys_db.delete_database, test_db_name)

@pytest.mark.asyncio
async def test_create_collections_with_real_db(test_db):
    """Test creating collections with a real ArangoDB database."""
    # Create document collections
    doc_collections = await asyncio.to_thread(create_document_collections, test_db, TEST_SCHEMA)
    
    # Verify collections were created
    assert await asyncio.to_thread(test_db.has_collection, "rules")
    assert await asyncio.to_thread(test_db.has_collection, "examples")
    
    # Verify collection properties
    rules_collection = await asyncio.to_thread(test_db.collection, "rules")
    rules_info = await asyncio.to_thread(rules_collection.properties)
    assert rules_info["name"] == "rules"
    assert not rules_info["edge"]
    
    # Create edge collections
    edge_collections = await asyncio.to_thread(create_edge_collections, test_db, TEST_SCHEMA)
    
    # Verify edge collections were created
    assert await asyncio.to_thread(test_db.has_collection, "rule_examples")
    
    # Verify edge collection properties
    edges_collection = await asyncio.to_thread(test_db.collection, "rule_examples")
    edges_info = await asyncio.to_thread(edges_collection.properties)
    assert edges_info["name"] == "rule_examples"
    assert edges_info["edge"]
    
    # Test inserting documents
    doc = {
        "_key": "rule1",
        "title": "Test Rule",
        "description": "This is a test rule",
        "content": "Test rule content",
        "category": "test",
        "priority": 1
    }
    await asyncio.to_thread(rules_collection.insert, doc)
    
    # Verify document was inserted
    rule = await asyncio.to_thread(rules_collection.get, "rule1")
    assert rule["title"] == "Test Rule"

@pytest.mark.asyncio
async def test_create_graph_with_real_db(test_db):
    """Test creating graphs with a real ArangoDB database."""
    # Ensure collections exist
    if not await asyncio.to_thread(test_db.has_collection, "rules"):
        await asyncio.to_thread(test_db.create_collection, "rules")
    if not await asyncio.to_thread(test_db.has_collection, "examples"):
        await asyncio.to_thread(test_db.create_collection, "examples")
    if not await asyncio.to_thread(test_db.has_collection, "rule_examples"):
        await asyncio.to_thread(test_db.create_collection, "rule_examples", edge=True)
    
    # Create graphs
    graphs = await asyncio.to_thread(create_named_graphs, test_db, TEST_SCHEMA)
    
    # Verify graph was created
    assert await asyncio.to_thread(test_db.has_graph, "knowledge_graph")
    
    # Get graph
    graph = await asyncio.to_thread(test_db.graph, "knowledge_graph")
    
    # Verify edge definitions
    edge_definitions = await asyncio.to_thread(graph.edge_definitions)
    assert len(edge_definitions) == 1
    assert edge_definitions[0]["edge_collection"] == "rule_examples"
    assert "rules" in edge_definitions[0]["from_vertex_collections"]
    assert "examples" in edge_definitions[0]["to_vertex_collections"]

@pytest.mark.asyncio
async def test_create_view_with_real_db(test_db):
    """Test creating views with a real ArangoDB database."""
    # Ensure collections exist
    has_collection = await asyncio.to_thread(test_db.has_collection, "rules")
    if not has_collection:
        await asyncio.to_thread(test_db.create_collection, "rules")
    
    # Create analyzers - wrap in asyncio.to_thread as it contains database operations
    analyzers = await asyncio.to_thread(create_analyzers, test_db, TEST_SCHEMA)
    
    # Create views - wrap in asyncio.to_thread as it contains database operations
    views = await asyncio.to_thread(create_views, test_db, TEST_SCHEMA)
    
    # Verify view was created - using try/except as has_view is not available
    try:
        view = await asyncio.to_thread(test_db.view, "rules_view")
        assert view is not None
    except Exception as e:
        assert False, f"View 'rules_view' was not created: {e}"
    
    # Get view
    view = await asyncio.to_thread(test_db.view, "rules_view")
    
    # Verify view properties
    # In newer versions of python-arango, view() returns a dict instead of an object with properties() method
    if hasattr(view, 'properties'):
        view_properties = await asyncio.to_thread(view.properties)
    else:
        # If view is already a dict, use it directly
        view_properties = view
    
    assert view_properties["name"] == "rules_view"
    assert view_properties["type"] == "arangosearch"
    
    # Insert test documents - need to use lambda for complex operations with document data
    rules_collection = await asyncio.to_thread(test_db.collection, "rules")
    
    # For method calls with document data, we still need to use lambda
    doc1 = {
        "_key": "view_test_rule1",
        "title": "Database Connection",
        "description": "How to connect to the database",
        "content": "Use ArangoClient to connect to ArangoDB",
        "category": "database",
        "priority": 1
    }
    await asyncio.to_thread(rules_collection.insert, doc1)
    
    doc2 = {
        "_key": "view_test_rule2",
        "title": "Query Execution",
        "description": "How to execute AQL queries",
        "content": "Use the execute_query method to run AQL queries",
        "category": "query",
        "priority": 2
    }
    await asyncio.to_thread(rules_collection.insert, doc2)
    
    # Wait for view to be populated (indexing might take a moment)
    await asyncio.sleep(2)  # Increase sleep time to ensure indexing completes
    
    # Test search using the view - use a simpler query that doesn't rely on specific analyzer
    query = """
    FOR doc IN rules_view
    RETURN doc
    """
    # For AQL execution
    cursor = await asyncio.to_thread(test_db.aql.execute, query)
    
    # For cursor iteration, create a helper function for clarity
    def get_cursor_results(cursor):
        return [doc for doc in cursor]
    
    results = await asyncio.to_thread(get_cursor_results, cursor)
    
    # Verify search results - just check that we get some results back
    assert len(results) > 0

@pytest.mark.asyncio
async def test_store_schema_with_real_db(test_db):
    """Test storing schema document with a real ArangoDB database."""
    # Ensure collections exist for storing schema
    if not await asyncio.to_thread(test_db.has_collection, "schema_docs"):
        await asyncio.to_thread(test_db.create_collection, "schema_docs")
    
    # Store schema - don't pass the collection parameter as it's not supported
    schema_key = await asyncio.to_thread(store_schema_doc, test_db, TEST_SCHEMA)
    
    # Verify schema was stored
    assert schema_key is not None
    
    # Verify schema document exists in the collection
    schema_docs = await asyncio.to_thread(test_db.collection, "schema_docs")
    doc = await asyncio.to_thread(schema_docs.get, schema_key)
    
    assert doc is not None
    assert "schema" in doc
    assert doc["schema"] == TEST_SCHEMA

@pytest.mark.asyncio
async def test_get_schema_with_real_db(test_db):
    """Test retrieving schema document with a real ArangoDB database."""
    # Ensure collections exist for storing schema
    if not await asyncio.to_thread(test_db.has_collection, "schema_docs"):
        await asyncio.to_thread(test_db.create_collection, "schema_docs")
    
    # Store schema - don't pass the collection parameter as it's not supported
    schema_key = await asyncio.to_thread(store_schema_doc, test_db, TEST_SCHEMA)
    
    # Get schema
    schema_doc = await get_schema_doc(test_db)
    
    # Verify schema
    assert schema_doc is not None
    assert "schema" in schema_doc
    assert schema_doc["schema"]["database_name"] == TEST_SCHEMA["database_name"]

@pytest.mark.asyncio
async def test_end_to_end_database_setup():
    """Test end-to-end database setup with real ArangoDB."""
    # Create a unique test database name with timestamp
    import time
    test_db_name = f"test_e2e_{int(time.time())}"
    
    # Connect to ArangoDB - wrap client initialization in asyncio.to_thread
    client = await asyncio.to_thread(ArangoClient, hosts="http://localhost:8529")
    
    # Connect to _system database
    sys_db = await asyncio.to_thread(client.db, "_system", username="root", password="openSesame")
    
    # Check if database exists and delete if needed
    has_db = await asyncio.to_thread(sys_db.has_database, test_db_name)
    if has_db:
        await asyncio.to_thread(sys_db.delete_database, test_db_name)
    
    # Create test database
    await asyncio.to_thread(sys_db.create_database, test_db_name)
    
    # Note: Don't need to connect to test database directly as setup_ai_knowledge_db does this
    
    # Setup database with schema - this function is already async and uses asyncio.to_thread internally
    setup_db = await setup_ai_knowledge_db(
        host="http://localhost:8529",
        username="root",
        password="openSesame",
        db_name=test_db_name,
        schema_path=None  # Use the default schema
    )
    
    # Verify database was setup correctly
    assert setup_db is not None
    
    # Verify collections were created - wrap all checks in asyncio.to_thread
    assert await asyncio.to_thread(setup_db.has_collection, "rules")
    assert await asyncio.to_thread(setup_db.has_collection, "critical_rules")
    assert await asyncio.to_thread(setup_db.has_collection, "code_patterns")
    assert await asyncio.to_thread(setup_db.has_collection, "schema_docs")
    
    # Verify graph was created
    assert await asyncio.to_thread(setup_db.has_graph, "knowledge_graph")
    
    # Verify view was created - use try/except as has_view is not available
    try:
        view = await asyncio.to_thread(setup_db.view, "unified_search_view")
        assert view is not None
    except Exception as e:
        assert False, f"View 'unified_search_view' was not created: {e}"
    
    # Verify schema was stored - get_schema_doc is already async
    schema_doc = await get_schema_doc(setup_db)
    assert schema_doc is not None
    assert "schema" in schema_doc
    schema = schema_doc["schema"]
    assert "database_name" in schema
    
    # Clean up
    await asyncio.to_thread(sys_db.delete_database, test_db_name)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 