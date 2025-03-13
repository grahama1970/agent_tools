"""
Basic tests for the AI Knowledge Database module.

This module tests the basic functionality of ai_knowledge_db.py.

Documentation reference: https://python-driver.arangodb.com/
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, mock_open

from agent_tools.cursor_rules.core.ai_knowledge_db import (
    load_schema,
    create_document_collections,
    create_edge_collections,
    create_views,
    create_analyzers
)
from agent_tools.cursor_rules.core.db import get_db, create_database

# Test data
TEST_SCHEMA = {
    "database_name": "test_cursor_rules",
    "description": "Test database for AI knowledge retrieval",
    "collections": {
        "test_rules": {
            "type": "document",
            "description": "Test rules collection",
            "fields": {
                "_key": {"type": "string", "description": "Unique identifier", "indexed": True},
                "title": {"type": "string", "description": "Rule title", "indexed": True}
            }
        }
    },
    "edge_collections": {
        "test_edges": {
            "type": "edge",
            "description": "Test edge collection",
            "fields": {
                "relationship_type": {"type": "string", "description": "Type of relationship", "indexed": True}
            },
            "from": ["test_rules"],
            "to": ["test_rules"]
        }
    },
    "views": {
        "test_view": {
            "type": "arangosearch",
            "description": "Test search view",
            "properties": {
                "analyzer": "text_en",
                "features": ["frequency", "position"]
            },
            "links": {
                "test_rules": {
                    "fields": {
                        "title": {"analyzer": "text_en"}
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
                "stemming": True
            }
        }
    }
}

def test_load_schema():
    """Test loading the schema from a file."""
    schema_json = json.dumps(TEST_SCHEMA)
    with patch("builtins.open", mock_open(read_data=schema_json)):
        schema = load_schema("dummy_path.json")
        assert schema == TEST_SCHEMA

def test_create_collections_with_mock():
    """Test creating collections with a mock database."""
    # Create a mock database
    mock_db = MagicMock()
    
    # For document collections test
    # Mock has_collection to return False (collection doesn't exist)
    mock_db.has_collection.return_value = False
    # Mock create_collection to return a mock collection
    mock_collection = MagicMock()
    mock_db.create_collection.return_value = mock_collection
    
    # Test creating document collections
    # The function should create the collection since it doesn't exist
    collections = create_document_collections(mock_db, TEST_SCHEMA)
    
    # Verify the function tried to create the collection
    mock_db.create_collection.assert_called_once()
    
    # For edge collections test
    # Reset the mocks
    mock_db.has_collection.return_value = False
    mock_db.create_collection.reset_mock()
    
    # Test creating edge collections
    edge_collections = create_edge_collections(mock_db, TEST_SCHEMA)
    
    # Verify the function tried to create the edge collection
    mock_db.create_collection.assert_called_once_with("test_edges", edge=True)

def test_create_views_with_mock():
    """Test creating views with a mock database."""
    # Create a mock database
    mock_db = MagicMock()
    
    # Mock has_view to return False (view doesn't exist)
    mock_db.has_view.return_value = False
    
    # Test creating views
    views = create_views(mock_db, TEST_SCHEMA)
    
    # Verify the function tried to create the view
    mock_db.create_arangosearch_view.assert_called_once()

def test_create_analyzers_with_mock():
    """Test creating analyzers with a mock database."""
    # Create a mock database
    mock_db = MagicMock()
    
    # Mock analyzers method to return empty list (no analyzers exist)
    mock_db.analyzers.return_value = []
    
    # Test creating analyzers
    analyzers = create_analyzers(mock_db, TEST_SCHEMA)
    
    # Verify the function tried to create the analyzer
    mock_db.create_analyzer.assert_called_once()

@pytest.mark.skipif(True, reason="Skipping database connection test")
def test_db_connection():
    """Test connecting to the database."""
    # This test is skipped by default to avoid requiring a real database
    # Remove the skipif decorator to run this test with a real database
    db = get_db()
    assert db is not None
    
    # Check if we can get database info
    info = db.properties()
    assert "name" in info
    assert info["name"] == "cursor_rules"

@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="Skipping database creation test")
async def test_create_database():
    """Test creating a database."""
    # This test is skipped by default to avoid requiring a real database
    # Remove the skipif decorator to run this test with a real database
    db_name = "test_cursor_rules_temp"
    db = create_database(db_name=db_name)
    assert db is not None
    
    # Check if we can get database info
    info = db.properties()
    assert "name" in info
    assert info["name"] == db_name
    
    # Clean up - delete the test database
    client = db.conn.client
    sys_db = client.db("_system", username="root", password="openSesame")
    sys_db.delete_database(db_name)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 