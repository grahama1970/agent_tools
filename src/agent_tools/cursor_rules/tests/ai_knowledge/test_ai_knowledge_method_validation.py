"""
Tests to validate ArangoDB method existence and signatures using method_validator.

Documentation References:
- ArangoDB Python Driver: https://docs.python-driver.arangodb.com/
- Method Validator: See src/agent_tools/method_validator/README.md
"""

import pytest
import os
from pathlib import Path

from agent_tools.method_validator.analyzer import validate_method

def test_validate_arango_client_methods():
    """Test that critical ArangoDB client methods exist with correct signatures."""
    # ArangoClient constructor
    is_valid, message = validate_method("arango", "ArangoClient")
    assert is_valid, f"ArangoClient validation failed: {message}"
    
    # Database connection method
    is_valid, message = validate_method("arango.client", "ArangoClient.db")
    assert is_valid, f"ArangoClient.db validation failed: {message}"

def test_validate_database_methods():
    """Test that critical ArangoDB database methods exist with correct signatures."""
    # Collection methods
    is_valid, message = validate_method("arango.database", "StandardDatabase.has_collection")
    assert is_valid, f"StandardDatabase.has_collection validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_collection")
    assert is_valid, f"StandardDatabase.create_collection validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.collection")
    assert is_valid, f"StandardDatabase.collection validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.collections")
    assert is_valid, f"StandardDatabase.collections validation failed: {message}"
    
    # Graph methods
    is_valid, message = validate_method("arango.database", "StandardDatabase.has_graph")
    assert is_valid, f"StandardDatabase.has_graph validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_graph")
    assert is_valid, f"StandardDatabase.create_graph validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.graph")
    assert is_valid, f"StandardDatabase.graph validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.graphs")
    assert is_valid, f"StandardDatabase.graphs validation failed: {message}"
    
    # View methods
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_arangosearch_view")
    assert is_valid, f"StandardDatabase.create_arangosearch_view validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.views")
    assert is_valid, f"StandardDatabase.views validation failed: {message}"
    
    # Analyzer methods
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_analyzer")
    assert is_valid, f"StandardDatabase.create_analyzer validation failed: {message}"
    
    is_valid, message = validate_method("arango.database", "StandardDatabase.analyzers")
    assert is_valid, f"StandardDatabase.analyzers validation failed: {message}"

def test_validate_collection_methods():
    """Test that critical ArangoDB collection methods exist with correct signatures."""
    is_valid, message = validate_method("arango.collection", "StandardCollection.properties")
    assert is_valid, f"StandardCollection.properties validation failed: {message}"
    
    is_valid, message = validate_method("arango.collection", "StandardCollection.insert")
    assert is_valid, f"StandardCollection.insert validation failed: {message}"
    
    is_valid, message = validate_method("arango.collection", "StandardCollection.get")
    assert is_valid, f"StandardCollection.get validation failed: {message}"
    
    is_valid, message = validate_method("arango.collection", "StandardCollection.all")
    assert is_valid, f"StandardCollection.all validation failed: {message}"

def test_validate_cursor_methods():
    """Test that critical ArangoDB cursor methods exist with correct signatures."""
    is_valid, message = validate_method("arango.cursor", "Cursor.batch")
    assert is_valid, f"Cursor.batch validation failed: {message}"

def test_validate_graph_methods():
    """Test that critical ArangoDB graph methods exist with correct signatures."""
    is_valid, message = validate_method("arango.graph", "Graph.properties")
    assert is_valid, f"Graph.properties validation failed: {message}"

def test_validate_critical_method_parameters():
    """Test that critical methods have expected parameters."""
    # StandardDatabase.create_collection should have 'name' and 'edge' parameters
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_collection")
    assert is_valid, f"StandardDatabase.create_collection validation failed: {message}"
    
    # StandardDatabase.create_arangosearch_view should have 'name' and 'properties' parameters
    is_valid, message = validate_method("arango.database", "StandardDatabase.create_arangosearch_view")
    assert is_valid, f"StandardDatabase.create_arangosearch_view validation failed: {message}"

def test_validate_critical_asyncio_methods():
    """Test that critical asyncio methods exist with correct signatures."""
    is_valid, message = validate_method("asyncio", "to_thread")
    assert is_valid, f"asyncio.to_thread validation failed: {message}" 