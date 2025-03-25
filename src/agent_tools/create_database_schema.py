#!/usr/bin/env python3
"""
create_database_schema.py

This module provides utilities for creating and managing database schemas,
with a particular focus on ArangoDB schemas and CRUD operations.

Key features:
- ArangoDB schema creation and validation
- Collection definition and indexing
- Graph relationship modeling
- CRUD operations for documents and graphs
- Query generation for common operations
- Schema export and import
- Database configuration management

Requirements:
- python-arango for ArangoDB interaction
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path

# Configure logging
logger = logging.getLogger("dualipa.create_database_schema")

try:
    from arango import ArangoClient
    from arango.database import Database
    from arango.collection import StandardCollection
    from arango.graph import Graph
    from arango.exceptions import (
        DocumentInsertError, 
        DocumentUpdateError, 
        DocumentDeleteError,
        CollectionCreateError,
        GraphCreateError
    )
    HAVE_ARANGO = True
except ImportError:
    HAVE_ARANGO = False
    logger.warning("python-arango not found. Install with 'pip install python-arango'")


# Schema definition types
CollectionSchema = Dict[str, Any]
GraphSchema = Dict[str, Any]
DatabaseSchema = Dict[str, Any]


class ArangoDBManager:
    """
    Manager class for ArangoDB operations, schema creation, and CRUD functions.
    """
    
    def __init__(self, 
                 host: str = "http://localhost:8529", 
                 username: str = "root", 
                 password: str = "",
                 database: str = "_system"):
        """
        Initialize the ArangoDB manager.
        
        Args:
            host: ArangoDB host URL
            username: Database username
            password: Database password
            database: Database name
        """
        if not HAVE_ARANGO:
            raise ImportError("python-arango is required but not installed")
        
        self.client = ArangoClient(hosts=host)
        self.sys_db = self.client.db("_system", username=username, password=password)
        
        if database != "_system":
            # Create the database if it doesn't exist
            if not self.sys_db.has_database(database):
                self.sys_db.create_database(database)
            
            # Connect to the specified database
            self.db = self.client.db(database, username=username, password=password)
        else:
            self.db = self.sys_db
        
        self.database_name = database
    
    def create_collection(self, 
                         name: str, 
                         schema: Optional[CollectionSchema] = None,
                         edge: bool = False,
                         indexes: Optional[List[Dict[str, Any]]] = None) -> StandardCollection:
        """
        Create a collection with optional schema validation and indexes.
        
        Args:
            name: Collection name
            schema: JSON Schema for document validation
            edge: Whether this is an edge collection
            indexes: List of indexes to create
            
        Returns:
            The created collection object
        """
        try:
            # Check if collection already exists
            if self.db.has_collection(name):
                collection = self.db.collection(name)
                logger.info(f"Collection {name} already exists, using existing collection")
            else:
                # Create collection (document or edge)
                if edge:
                    collection = self.db.create_collection(name, edge=True)
                    logger.info(f"Created edge collection: {name}")
                else:
                    collection = self.db.create_collection(name)
                    logger.info(f"Created document collection: {name}")
                
                # Apply schema if provided
                if schema:
                    collection.configure(schema=schema)
                    logger.info(f"Applied schema to collection: {name}")
            
            # Create indexes if provided
            if indexes:
                for index_def in indexes:
                    index_type = index_def.pop("type")
                    fields = index_def.pop("fields")
                    
                    if index_type == "persistent":
                        collection.add_persistent_index(fields, **index_def)
                    elif index_type == "fulltext":
                        collection.add_fulltext_index(fields, **index_def)
                    elif index_type == "geo":
                        collection.add_geo_index(fields, **index_def)
                    elif index_type == "hash":
                        collection.add_hash_index(fields, **index_def)
                    elif index_type == "skiplist":
                        collection.add_skiplist_index(fields, **index_def)
                    
                    logger.info(f"Created {index_type} index on {', '.join(fields)} for collection: {name}")
            
            return collection
            
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise
    
    def create_graph(self, 
                    name: str, 
                    edge_definitions: List[Dict[str, Any]],
                    orphan_collections: Optional[List[str]] = None) -> Graph:
        """
        Create a graph with defined edge relationships.
        
        Args:
            name: Graph name
            edge_definitions: List of edge definitions
            orphan_collections: List of orphaned collections
            
        Returns:
            The created graph object
        """
        try:
            # Check if graph already exists
            if self.db.has_graph(name):
                graph = self.db.graph(name)
                logger.info(f"Graph {name} already exists, using existing graph")
            else:
                # Create graph
                graph = self.db.create_graph(
                    name, 
                    edge_definitions=edge_definitions,
                    orphan_collections=orphan_collections or []
                )
                logger.info(f"Created graph: {name} with {len(edge_definitions)} edge definitions")
            
            return graph
            
        except Exception as e:
            logger.error(f"Error creating graph {name}: {e}")
            raise
    
    def create_database_schema(self, schema: DatabaseSchema) -> Dict[str, Any]:
        """
        Create a complete database schema from a definition.
        
        Args:
            schema: Database schema definition
            
        Returns:
            Dictionary of created collections and graphs
        """
        created = {
            "collections": [],
            "graphs": []
        }
        
        # Create document collections
        for collection_def in schema.get("collections", []):
            name = collection_def.pop("name")
            edge = collection_def.pop("edge", False)
            schema_def = collection_def.pop("schema", None)
            indexes = collection_def.pop("indexes", None)
            
            collection = self.create_collection(
                name=name,
                schema=schema_def,
                edge=edge,
                indexes=indexes
            )
            
            created["collections"].append(name)
        
        # Create graph definitions
        for graph_def in schema.get("graphs", []):
            name = graph_def.pop("name")
            edge_definitions = graph_def.pop("edge_definitions")
            orphan_collections = graph_def.pop("orphan_collections", None)
            
            graph = self.create_graph(
                name=name,
                edge_definitions=edge_definitions,
                orphan_collections=orphan_collections
            )
            
            created["graphs"].append(name)
        
        logger.info(f"Created database schema with {len(created['collections'])} collections and {len(created['graphs'])} graphs")
        return created
    
    def load_schema_from_file(self, schema_file: Union[str, Path]) -> DatabaseSchema:
        """
        Load a database schema from a JSON file.
        
        Args:
            schema_file: Path to the schema file
            
        Returns:
            The loaded schema definition
        """
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            logger.info(f"Loaded database schema from {schema_file}")
            return schema
        except Exception as e:
            logger.error(f"Error loading schema from {schema_file}: {e}")
            raise
    
    def export_schema_to_file(self, schema: DatabaseSchema, output_file: Union[str, Path]) -> None:
        """
        Export a database schema to a JSON file.
        
        Args:
            schema: Database schema to export
            output_file: Path to save the schema
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Exported database schema to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting schema to {output_file}: {e}")
            raise
    
    def create_index(self, collection_name: str, index_type: str, fields: List[str], **kwargs) -> Dict[str, Any]:
        """
        Create an index on a collection.
        
        Args:
            collection_name: Name of the collection
            index_type: Type of index to create
            fields: Fields to index
            **kwargs: Additional index options
            
        Returns:
            Index information
        """
        try:
            collection = self.db.collection(collection_name)
            
            if index_type == "persistent":
                result = collection.add_persistent_index(fields, **kwargs)
            elif index_type == "fulltext":
                result = collection.add_fulltext_index(fields, **kwargs)
            elif index_type == "geo":
                result = collection.add_geo_index(fields, **kwargs)
            elif index_type == "hash":
                result = collection.add_hash_index(fields, **kwargs)
            elif index_type == "skiplist":
                result = collection.add_skiplist_index(fields, **kwargs)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            logger.info(f"Created {index_type} index on {', '.join(fields)} for collection: {collection_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating index on {collection_name}: {e}")
            raise
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a document into a collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            The inserted document with _id, _key, and _rev
        """
        try:
            collection = self.db.collection(collection_name)
            result = collection.insert(document, return_new=True)
            return result
        except Exception as e:
            logger.error(f"Error inserting document into {collection_name}: {e}")
            raise
    
    def update_document(self, collection_name: str, document_key: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a document in a collection.
        
        Args:
            collection_name: Name of the collection
            document_key: Key of the document to update
            document: Updated document content
            
        Returns:
            The updated document
        """
        try:
            collection = self.db.collection(collection_name)
            result = collection.update({"_key": document_key}, document, return_new=True)
            return result
        except Exception as e:
            logger.error(f"Error updating document {document_key} in {collection_name}: {e}")
            raise
    
    def delete_document(self, collection_name: str, document_key: str) -> Dict[str, Any]:
        """
        Delete a document from a collection.
        
        Args:
            collection_name: Name of the collection
            document_key: Key of the document to delete
            
        Returns:
            Deletion result
        """
        try:
            collection = self.db.collection(collection_name)
            result = collection.delete({"_key": document_key}, return_old=True)
            return result
        except Exception as e:
            logger.error(f"Error deleting document {document_key} from {collection_name}: {e}")
            raise
    
    def get_document(self, collection_name: str, document_key: str) -> Dict[str, Any]:
        """
        Get a document from a collection.
        
        Args:
            collection_name: Name of the collection
            document_key: Key of the document to retrieve
            
        Returns:
            The document
        """
        try:
            collection = self.db.collection(collection_name)
            result = collection.get(document_key)
            return result
        except Exception as e:
            logger.error(f"Error retrieving document {document_key} from {collection_name}: {e}")
            raise
    
    def create_edge(self, edge_collection: str, from_id: str, to_id: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an edge between two documents.
        
        Args:
            edge_collection: Name of the edge collection
            from_id: _id of the source document
            to_id: _id of the target document
            data: Optional edge data
            
        Returns:
            The created edge
        """
        try:
            collection = self.db.collection(edge_collection)
            edge = {
                "_from": from_id,
                "_to": to_id
            }
            
            if data:
                edge.update(data)
            
            result = collection.insert(edge, return_new=True)
            return result
        except Exception as e:
            logger.error(f"Error creating edge in {edge_collection} from {from_id} to {to_id}: {e}")
            raise
    
    def query(self, query_string: str, bind_vars: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute an AQL query.
        
        Args:
            query_string: AQL query
            bind_vars: Bind variables for the query
            
        Returns:
            Query results
        """
        try:
            cursor = self.db.aql.execute(query_string, bind_vars=bind_vars or {})
            return list(cursor)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the database.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        try:
            if self.db.has_collection(collection_name):
                self.db.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection does not exist: {collection_name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            raise
    
    def delete_graph(self, graph_name: str, drop_collections: bool = False) -> bool:
        """
        Delete a graph from the database.
        
        Args:
            graph_name: Name of the graph to delete
            drop_collections: Whether to also drop the collections
            
        Returns:
            True if successful
        """
        try:
            if self.db.has_graph(graph_name):
                self.db.delete_graph(graph_name, drop_collections=drop_collections)
                logger.info(f"Deleted graph: {graph_name}")
                return True
            else:
                logger.warning(f"Graph does not exist: {graph_name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting graph {graph_name}: {e}")
            raise
    
    def generate_aql_query(self, 
                          collection_name: str, 
                          query_type: str, 
                          **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an AQL query for common operations.
        
        Args:
            collection_name: Name of the collection to query
            query_type: Type of query (search, filter, join, etc.)
            **kwargs: Query-specific parameters
            
        Returns:
            Tuple of (query_string, bind_vars)
        """
        bind_vars = {}
        
        if query_type == "filter":
            # Filter documents based on field values
            fields = kwargs.get("fields", {})
            bind_vars = {"@collection": collection_name}
            conditions = []
            
            for field, value in fields.items():
                bind_name = f"val_{field.replace('.', '_')}"
                bind_vars[bind_name] = value
                conditions.append(f"doc.{field} == @{bind_name}")
            
            condition_str = " AND ".join(conditions) if conditions else "true"
            
            query = f"""
            FOR doc IN @@collection
                FILTER {condition_str}
                RETURN doc
            """
            
        elif query_type == "search":
            # Full-text search
            search_field = kwargs.get("field", "text")
            search_term = kwargs.get("term", "")
            bind_vars = {
                "@collection": collection_name,
                "searchTerm": search_term
            }
            
            query = f"""
            FOR doc IN FULLTEXT(@@collection, "{search_field}", @searchTerm)
                RETURN doc
            """
            
        elif query_type == "graph_traverse":
            # Graph traversal
            start_vertex = kwargs.get("start_vertex")
            direction = kwargs.get("direction", "outbound")
            graph_name = kwargs.get("graph_name")
            depth = kwargs.get("depth", 1)
            bind_vars = {
                "startVertex": start_vertex,
                "depth": depth
            }
            
            query = f"""
            FOR vertex, edge, path IN 1..@depth {direction} "{start_vertex}"
                GRAPH "{graph_name}"
                RETURN {{
                    "vertex": vertex,
                    "edge": edge,
                    "path": path
                }}
            """
            
        elif query_type == "join":
            # Join two collections
            join_collection = kwargs.get("join_collection")
            local_field = kwargs.get("local_field", "_id")
            foreign_field = kwargs.get("foreign_field", "_id")
            bind_vars = {
                "@collection1": collection_name,
                "@collection2": join_collection
            }
            
            query = f"""
            FOR doc1 IN @@collection1
                FOR doc2 IN @@collection2
                    FILTER doc1.{local_field} == doc2.{foreign_field}
                    RETURN {{
                        "doc1": doc1,
                        "doc2": doc2
                    }}
            """
            
        elif query_type == "aggregate":
            # Aggregation query
            group_field = kwargs.get("group_field")
            aggregate_field = kwargs.get("aggregate_field")
            aggregate_op = kwargs.get("aggregate_op", "SUM")
            bind_vars = {"@collection": collection_name}
            
            query = f"""
            FOR doc IN @@collection
                COLLECT group_val = doc.{group_field} 
                AGGREGATE result = {aggregate_op}(doc.{aggregate_field})
                RETURN {{
                    "{group_field}": group_val,
                    "result": result
                }}
            """
            
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        return query.strip(), bind_vars


# DuaLipa-specific schema for code block extraction storage
DUALIPA_SCHEMA = {
    "collections": [
        {
            "name": "code_blocks",
            "edge": False,
            "schema": {
                "rule": {
                    "type": "object",
                    "properties": {
                        "uuid": {"type": "string"},
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "language": {"type": "string"},
                        "content": {"type": "string"},
                        "file_path": {"type": "string"},
                        "parent_uuid": {"type": ["string", "null"]},
                        "metadata": {"type": "object"}
                    },
                    "required": ["uuid", "id", "type", "content"]
                },
                "level": "moderate",
                "message": "Block validation failed"
            },
            "indexes": [
                {
                    "type": "persistent",
                    "fields": ["uuid"],
                    "unique": True
                },
                {
                    "type": "persistent",
                    "fields": ["type"]
                },
                {
                    "type": "persistent",
                    "fields": ["language"]
                },
                {
                    "type": "persistent",
                    "fields": ["parent_uuid"]
                },
                {
                    "type": "fulltext",
                    "fields": ["content"],
                    "minLength": 3
                }
            ]
        },
        {
            "name": "block_relationships",
            "edge": True,
            "indexes": [
                {
                    "type": "persistent",
                    "fields": ["_from", "_to"],
                    "unique": True
                }
            ]
        },
        {
            "name": "repositories",
            "edge": False,
            "schema": {
                "rule": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                        "branch": {"type": "string"},
                        "extracted_at": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["name", "extracted_at"]
                },
                "level": "moderate"
            },
            "indexes": [
                {
                    "type": "persistent",
                    "fields": ["name"]
                },
                {
                    "type": "persistent",
                    "fields": ["url"]
                }
            ]
        }
    ],
    "graphs": [
        {
            "name": "dualipa_knowledge_graph",
            "edge_definitions": [
                {
                    "collection": "block_relationships",
                    "from": ["code_blocks"],
                    "to": ["code_blocks"]
                }
            ],
            "orphan_collections": ["repositories"]
        }
    ]
}


def create_dualipa_schema(host: str = "http://localhost:8529", 
                          username: str = "root", 
                          password: str = "",
                          database: str = "dualipa") -> ArangoDBManager:
    """
    Create the standard DuaLipa database schema for code extraction storage.
    
    Args:
        host: ArangoDB host URL
        username: Database username
        password: Database password
        database: Database name
        
    Returns:
        ArangoDBManager instance
    """
    try:
        # Create the manager
        manager = ArangoDBManager(host, username, password, database)
        
        # Create the schema
        manager.create_database_schema(DUALIPA_SCHEMA)
        
        logger.info(f"Created DuaLipa database schema in database '{database}'")
        return manager
    except Exception as e:
        logger.error(f"Error creating DuaLipa schema: {e}")
        raise


def store_extraction_blocks(blocks: List[Dict[str, Any]], 
                           arangodb_manager: ArangoDBManager) -> Tuple[int, int]:
    """
    Store extraction blocks in ArangoDB.
    
    Args:
        blocks: Extracted blocks to store
        arangodb_manager: Initialized ArangoDBManager
        
    Returns:
        Tuple of (blocks_stored, relationships_created)
    """
    try:
        blocks_stored = 0
        relationships_created = 0
        
        # Store blocks
        for block in blocks:
            # Insert the block
            result = arangodb_manager.insert_document("code_blocks", block)
            blocks_stored += 1
            
            # Create parent-child relationship if applicable
            if "parent_uuid" in block and block["parent_uuid"]:
                parent_id = f"code_blocks/{block['parent_uuid']}"
                child_id = f"code_blocks/{block['uuid']}"
                
                edge_data = {
                    "relationship_type": "parent_child"
                }
                
                arangodb_manager.create_edge("block_relationships", child_id, parent_id, edge_data)
                relationships_created += 1
        
        logger.info(f"Stored {blocks_stored} blocks and created {relationships_created} relationships")
        return blocks_stored, relationships_created
    
    except Exception as e:
        logger.error(f"Error storing extraction blocks: {e}")
        raise


def extract_and_store_blocks(repo_path: Union[str, Path], 
                            host: str = "http://localhost:8529", 
                            username: str = "root", 
                            password: str = "",
                            database: str = "dualipa") -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Extract blocks from a repository and store them in ArangoDB.
    
    Args:
        repo_path: Path to the repository
        host: ArangoDB host URL
        username: Database username
        password: Database password
        database: Database name
        
    Returns:
        Tuple of (blocks, (blocks_stored, relationships_created))
    """
    try:
        # Import the extraction function
        from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
        
        # Convert to Path if needed
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        
        # Extract blocks
        blocks = extract_all_blocks(repo_path)
        
        # Create the schema and store blocks
        manager = create_dualipa_schema(host, username, password, database)
        stored_stats = store_extraction_blocks(blocks, manager)
        
        return blocks, stored_stats
    
    except Exception as e:
        logger.error(f"Error extracting and storing blocks: {e}")
        raise


def generate_example_schema() -> Dict[str, Any]:
    """
    Generate an example database schema for demonstration purposes.
    
    Returns:
        Example schema definition
    """
    example_schema = {
        "collections": [
            {
                "name": "users",
                "edge": False,
                "schema": {
                    "rule": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "email": {"type": "string"},
                            "password_hash": {"type": "string"},
                            "created_at": {"type": "string"},
                            "active": {"type": "boolean"}
                        },
                        "required": ["username", "email", "password_hash"]
                    },
                    "level": "moderate"
                },
                "indexes": [
                    {
                        "type": "persistent",
                        "fields": ["username"],
                        "unique": True
                    },
                    {
                        "type": "persistent",
                        "fields": ["email"],
                        "unique": True
                    }
                ]
            },
            {
                "name": "documents",
                "edge": False,
                "schema": {
                    "rule": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "created_at": {"type": "string"},
                            "updated_at": {"type": "string"},
                            "user_id": {"type": "string"}
                        },
                        "required": ["title", "content", "user_id"]
                    },
                    "level": "moderate"
                },
                "indexes": [
                    {
                        "type": "persistent",
                        "fields": ["user_id"]
                    },
                    {
                        "type": "fulltext",
                        "fields": ["content"],
                        "minLength": 3
                    }
                ]
            },
            {
                "name": "owns",
                "edge": True,
                "indexes": [
                    {
                        "type": "persistent",
                        "fields": ["_from", "_to"],
                        "unique": True
                    }
                ]
            }
        ],
        "graphs": [
            {
                "name": "document_management",
                "edge_definitions": [
                    {
                        "collection": "owns",
                        "from": ["users"],
                        "to": ["documents"]
                    }
                ]
            }
        ]
    }
    
    return example_schema


def print_aql_examples():
    """Print example AQL queries for common operations."""
    examples = [
        {
            "description": "Basic document retrieval",
            "query": """
            FOR doc IN users
                FILTER doc.username == @username
                RETURN doc
            """,
            "bind_vars": {"username": "john_doe"}
        },
        {
            "description": "Join collections",
            "query": """
            FOR user IN users
                FOR doc IN documents
                    FILTER doc.user_id == user._key
                    RETURN { "user": user.username, "document": doc.title }
            """
        },
        {
            "description": "Graph traversal",
            "query": """
            FOR user, edge, path IN 1..1 OUTBOUND 'users/123456'
                GRAPH 'document_management'
                RETURN { "user": user.username, "document": edge._to }
            """
        },
        {
            "description": "Aggregation",
            "query": """
            FOR doc IN documents
                COLLECT user_id = doc.user_id 
                AGGREGATE count = COUNT(doc)
                RETURN { "user_id": user_id, "document_count": count }
            """
        },
        {
            "description": "Full-text search",
            "query": """
            FOR doc IN FULLTEXT(documents, "content", @searchTerm)
                RETURN { "title": doc.title, "preview": SUBSTRING(doc.content, 0, 150) }
            """,
            "bind_vars": {"searchTerm": "database"}
        },
        {
            "description": "Document creation",
            "query": """
            INSERT { 
                username: @username, 
                email: @email, 
                password_hash: @password_hash, 
                created_at: DATE_ISO8601(DATE_NOW()),
                active: true
            } INTO users
            RETURN NEW
            """,
            "bind_vars": {
                "username": "new_user",
                "email": "user@example.com",
                "password_hash": "hashed_password"
            }
        },
        {
            "description": "Document update",
            "query": """
            UPDATE @key WITH { 
                content: @content, 
                updated_at: DATE_ISO8601(DATE_NOW()) 
            } IN documents
            RETURN NEW
            """,
            "bind_vars": {
                "key": "123456",
                "content": "Updated content"
            }
        },
        {
            "description": "Document deletion",
            "query": """
            REMOVE @key IN documents
            RETURN OLD
            """,
            "bind_vars": {"key": "123456"}
        },
        {
            "description": "Edge creation",
            "query": """
            INSERT { 
                _from: @from, 
                _to: @to,
                created_at: DATE_ISO8601(DATE_NOW())
            } INTO owns
            RETURN NEW
            """,
            "bind_vars": {
                "from": "users/123456",
                "to": "documents/789012"
            }
        }
    ]
    
    print("AQL Query Examples:")
    print("==================")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print("-" * (len(example['description']) + 4))
        print(example["query"].strip())
        
        if "bind_vars" in example:
            print("\nBind variables:")
            for var, value in example["bind_vars"].items():
                print(f"  @{var}: {value}")
    
    print("\n==================")


if __name__ == "__main__":
    import argparse
    
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="ArangoDB Schema Management Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create schema command
    create_parser = subparsers.add_parser("create", help="Create a database schema")
    create_parser.add_argument("--host", default="http://localhost:8529", help="ArangoDB host URL")
    create_parser.add_argument("--username", default="root", help="Database username")
    create_parser.add_argument("--password", default="", help="Database password")
    create_parser.add_argument("--database", default="dualipa", help="Database name")
    create_parser.add_argument("--schema", help="Path to schema definition file")
    
    # Generate example schema command
    example_parser = subparsers.add_parser("example", help="Generate an example schema")
    example_parser.add_argument("--output", help="Output file path")
    
    # Show AQL examples command
    aql_parser = subparsers.add_parser("aql", help="Show AQL query examples")
    
    # Extract and store command
    extract_parser = subparsers.add_parser("extract", help="Extract and store repository blocks")
    extract_parser.add_argument("repo_path", help="Path to the repository")
    extract_parser.add_argument("--host", default="http://localhost:8529", help="ArangoDB host URL")
    extract_parser.add_argument("--username", default="root", help="Database username")
    extract_parser.add_argument("--password", default="", help="Database password")
    extract_parser.add_argument("--database", default="dualipa", help="Database name")
    
    args = parser.parse_args()
    
    if args.command == "create":
        if args.schema:
            try:
                manager = ArangoDBManager(args.host, args.username, args.password, args.database)
                schema = manager.load_schema_from_file(args.schema)
                created = manager.create_database_schema(schema)
                print(f"Created {len(created['collections'])} collections and {len(created['graphs'])} graphs")
            except Exception as e:
                print(f"Error creating schema: {e}")
        else:
            try:
                manager = create_dualipa_schema(args.host, args.username, args.password, args.database)
                print(f"Created DuaLipa schema in database '{args.database}'")
            except Exception as e:
                print(f"Error creating schema: {e}")
    
    elif args.command == "example":
        example = generate_example_schema()
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(example, f, indent=2)
                print(f"Saved example schema to {args.output}")
            except Exception as e:
                print(f"Error saving example schema: {e}")
        else:
            print(json.dumps(example, indent=2))
    
    elif args.command == "aql":
        print_aql_examples()
    
    elif args.command == "extract":
        try:
            blocks, (stored, relations) = extract_and_store_blocks(
                args.repo_path, 
                args.host, 
                args.username, 
                args.password, 
                args.database
            )
            print(f"Extracted {len(blocks)} blocks and stored {stored} blocks with {relations} relationships")
        except Exception as e:
            print(f"Error extracting and storing blocks: {e}")
    
    else:
        parser.print_help()