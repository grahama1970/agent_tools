"""
AI Knowledge Database module for Cursor Rules.

This module implements the enhanced schema for AI agent knowledge retrieval,
optimized for context efficiency and precision queries. It provides functions
to set up the database, create collections, and store schema documentation.
"""

import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import CollectionCreateError, GraphCreateError, ViewCreateError
from loguru import logger

from agent_tools.cursor_rules.core.db import get_db, create_database
from agent_tools.cursor_rules.schemas import load_schema as schemas_load_schema, AI_KNOWLEDGE_SCHEMA


def load_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the AI knowledge schema from a JSON file.
    
    Args:
        schema_path: Path to the schema file. If None, uses the default schema.
        
    Returns:
        Dict containing the schema
    """
    if schema_path is None:
        # Use the pre-loaded schema from the schemas package
        return AI_KNOWLEDGE_SCHEMA
    
    # If a custom path is provided, load from that path
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    return schema


def create_document_collections(db: StandardDatabase, schema: Dict[str, Any]) -> List[str]:
    """
    Create document collections defined in the schema.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        List of created collection names (including existing ones)
    """
    collections = []
    
    for collection_name, collection_def in schema["collections"].items():
        try:
            if not db.has_collection(collection_name):
                logger.info(f"Creating collection {collection_name}")
                db.create_collection(collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists")
            # Add all collections to the result list, whether newly created or existing
            collections.append(collection_name)
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
    
    return collections


def create_edge_collections(db: StandardDatabase, schema: Dict[str, Any]) -> List[str]:
    """
    Create edge collections defined in the schema.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        List of created edge collection names (including existing ones)
    """
    edge_collections = []
    
    for edge_name, edge_def in schema.get("edge_collections", {}).items():
        try:
            if not db.has_collection(edge_name):
                logger.info(f"Creating edge collection {edge_name}")
                db.create_collection(edge_name, edge=True)
            else:
                logger.info(f"Edge collection {edge_name} already exists")
            # Add all edge collections to the result list, whether newly created or existing
            edge_collections.append(edge_name)
        except Exception as e:
            logger.error(f"Error creating edge collection {edge_name}: {e}")
    
    return edge_collections


def create_named_graphs(db: StandardDatabase, schema: Dict[str, Any]) -> List[str]:
    """
    Create named graphs defined in the schema.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        List of created graph names (including existing ones)
    """
    graphs = []
    
    # Look for graphs in both "graphs" and "named_graphs" keys for compatibility
    graph_definitions = schema.get("graphs", {})
    if not graph_definitions:  # If empty, try the named_graphs key
        graph_definitions = schema.get("named_graphs", {})
    
    for graph_name, graph_def in graph_definitions.items():
        try:
            if not db.has_graph(graph_name):
                logger.info(f"Creating graph {graph_name}")
                
                edge_definitions = []
                for edge_def in graph_def.get("edge_definitions", []):
                    # Extract required properties with proper name conversions
                    collection_name = edge_def.get("collection") or edge_def.get("edge_collection")
                    from_collections = edge_def.get("from") or edge_def.get("from_collections", [])
                    to_collections = edge_def.get("to") or edge_def.get("to_collections", [])
                    
                    # Use the proper format expected by ArangoDB
                    edge_definitions.append({
                        "edge_collection": collection_name,
                        "from_vertex_collections": from_collections,
                        "to_vertex_collections": to_collections
                    })
                
                graph = db.create_graph(graph_name, edge_definitions)
                logger.info(f"Created graph {graph_name} with {len(edge_definitions)} edge definitions")
            else:
                logger.info(f"Graph {graph_name} already exists")
                graph = db.graph(graph_name)
            
            # Add all graphs to the result list, whether newly created or existing
            graphs.append(graph_name)
        except Exception as e:
            logger.error(f"Error creating graph {graph_name}: {str(e)}")
            # Re-raise to ensure test failures
            raise
    
    return graphs


def create_views(db: StandardDatabase, schema: Dict[str, Any]) -> List[str]:
    """
    Create views defined in the schema.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        List of created view names (including existing ones)
    """
    views = []
    
    # Get existing views to check if they already exist
    existing_views = [view["name"] for view in db.views()]
    
    for view_name, view_def in schema.get("views", {}).items():
        try:
            if view_name not in existing_views:
                logger.info(f"Creating view {view_name}")
                
                # Prepare view definition based on schema
                # Make sure to convert any Pydantic models to dictionaries
                properties = _ensure_dict(view_def.get("properties", {}))
                
                # Ensure all links are also converted to dictionaries
                links = _ensure_dict(view_def.get("links", {}))
                
                # For nested objects in links, ensure they're all dictionaries too
                processed_links = {}
                for collection_name, collection_links in links.items():
                    if isinstance(collection_links, dict):
                        processed_collection = {}
                        for field_name, field_config in collection_links.items():
                            processed_collection[field_name] = _ensure_dict(field_config)
                        processed_links[collection_name] = processed_collection
                    else:
                        processed_links[collection_name] = collection_links
                
                view_properties = {
                    **properties,
                    "links": processed_links
                }
                
                # Convert any remaining nested objects to dictionaries
                view_properties = _convert_nested_objects(view_properties)
                
                # Create the view with the prepared properties
                db.create_arangosearch_view(
                    name=view_name,
                    properties=view_properties
                )
            else:
                logger.info(f"View {view_name} already exists")
            # Add all views to the result list, whether newly created or existing
            views.append(view_name)
        except Exception as e:
            logger.error(f"Error creating view {view_name}: {e}")
    
    return views


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    """
    Ensure an object is a dictionary, converting from Pydantic model if needed.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary version of the object
    """
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        # If it's a Pydantic model, convert to dict
        return obj.dict()
    elif hasattr(obj, "__dict__"):
        # For other custom objects with __dict__
        return obj.__dict__
    elif isinstance(obj, dict):
        # Already a dict
        return obj
    else:
        # For other types, return empty dict to avoid errors
        return {}


def _convert_nested_objects(obj: Any) -> Any:
    """
    Recursively convert nested objects to dictionaries.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object with all nested objects as dictionaries
    """
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        # If it's a Pydantic model, convert to dict
        return _convert_nested_objects(obj.dict())
    elif isinstance(obj, dict):
        # Process dictionary values recursively
        return {k: _convert_nested_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Process list items recursively
        return [_convert_nested_objects(item) for item in obj]
    else:
        # For primitive types, return as is
        return obj


def create_analyzers(db: StandardDatabase, schema: Dict[str, Any]) -> List[str]:
    """
    Create analyzers defined in the schema.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        List of created analyzer names (including existing ones)
    """
    analyzers = []
    
    # Get existing analyzers to check if they already exist
    existing_analyzers = [analyzer["name"] for analyzer in db.analyzers()]
    
    for analyzer_name, analyzer_def in schema.get("analyzers", {}).items():
        try:
            if analyzer_name not in existing_analyzers:
                logger.info(f"Creating analyzer {analyzer_name}")
                
                # Prepare analyzer definition based on schema
                analyzer_type = analyzer_def.get("type", "identity")
                properties = analyzer_def.get("properties", {})
                features = analyzer_def.get("features", [])
                
                db.create_analyzer(
                    name=analyzer_name,
                    analyzer_type=analyzer_type,
                    properties=properties,
                    features=features
                )
            else:
                logger.info(f"Analyzer {analyzer_name} already exists")
            # Add all analyzers to the result list, whether newly created or existing
            analyzers.append(analyzer_name)
        except Exception as e:
            logger.error(f"Error creating analyzer {analyzer_name}: {e}")
    
    return analyzers


def store_schema_doc(db: StandardDatabase, schema: Dict[str, Any]) -> str:
    """
    Store the schema document in the schema_docs collection with a timestamp.
    
    Args:
        db: ArangoDB database
        schema: Schema definition
        
    Returns:
        The key of the stored schema document
    """
    # Ensure the schema_docs collection exists
    if not db.has_collection("schema_docs"):
        db.create_collection("schema_docs")
    
    # Create a new schema doc with timestamp
    timestamp = datetime.datetime.now().isoformat()
    schema_key = f"{timestamp.replace(':', '-').replace('.', '_')}_schema"
    
    schema_doc = {
        "_key": schema_key,
        "timestamp": timestamp,
        "description": schema.get("description", "Enhanced AI knowledge schema"),
        "schema": schema,
        "version": "1.0.0",
        "changelog": "Initial schema creation"
    }
    
    # Store the schema document
    schema_docs = db.collection("schema_docs")
    result = schema_docs.insert(schema_doc)
    
    # Use the key from the result if available (for test compatibility)
    if isinstance(result, dict) and "_key" in result:
        schema_key = result["_key"]
    
    logger.info(f"Stored schema document with key {schema_key}")
    return schema_key


async def setup_ai_knowledge_db(
    host: str = "http://localhost:8529",
    username: str = "root",
    password: str = "openSesame",
    db_name: str = "cursor_rules",
    schema_path: Optional[str] = None,
    reset: bool = False
) -> StandardDatabase:
    """
    Set up the AI knowledge database with the enhanced schema.
    
    Args:
        host: ArangoDB host
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name
        schema_path: Path to the schema file
        reset: Whether to reset the database
        
    Returns:
        ArangoDB database object
    """
    # Get the database
    db = await asyncio.to_thread(
        get_db, host, username, password, db_name
    )
    
    # Try to access the database - if it fails, create it
    try:
        # Test if we can access the database
        _ = [c for c in db.collections()]
    except Exception as e:
        logger.info(f"Database {db_name} does not exist, creating it: {e}")
        # Create the database
        sys_db = await asyncio.to_thread(get_db, host, username, password, "_system")
        if await asyncio.to_thread(sys_db.has_database, db_name):
            logger.info(f"Database {db_name} exists but is not accessible, trying to connect after ensuring it exists")
        else:
            await asyncio.to_thread(sys_db.create_database, db_name)
        # Connect to the newly created database
        db = await asyncio.to_thread(
            get_db, host, username, password, db_name
        )
    
    # Load the schema
    schema = load_schema(schema_path)
    
    # Reset the database if requested
    if reset:
        logger.warning(f"Resetting database {db_name}")
        collection_names = [c["name"] for c in db.collections()]
        
        # Drop all non-system collections
        for name in collection_names:
            if not name.startswith("_"):
                try:
                    db.delete_collection(name)
                    logger.info(f"Dropped collection {name}")
                except Exception as e:
                    logger.error(f"Error dropping collection {name}: {e}")
        
        # Drop all views
        for view in db.views():
            try:
                db.delete_view(view["name"])
                logger.info(f"Dropped view {view['name']}")
            except Exception as e:
                logger.error(f"Error dropping view {view['name']}: {e}")
        
        # Drop all graphs
        for graph in db.graphs():
            try:
                db.delete_graph(graph["name"])
                logger.info(f"Dropped graph {graph['name']}")
            except Exception as e:
                logger.error(f"Error dropping graph {graph['name']}: {e}")
    
    # Create collections and other components
    logger.info("Creating document collections")
    create_document_collections(db, schema)
    
    logger.info("Creating edge collections")
    create_edge_collections(db, schema)
    
    logger.info("Creating analyzers")
    create_analyzers(db, schema)
    
    logger.info("Creating graphs")
    create_named_graphs(db, schema)
    
    logger.info("Creating views")
    create_views(db, schema)
    
    # Store the schema in the database
    schema_key = store_schema_doc(db, schema)
    logger.info(f"Set up AI knowledge database with schema {schema_key}")
    
    return db


async def get_schema_doc(
    db: StandardDatabase, 
    version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a schema document from the database.
    
    Args:
        db: ArangoDB database
        version: Schema version to get. If None, gets the latest schema.
        
    Returns:
        Schema document
    """
    if not db.has_collection("schema_docs"):
        raise ValueError("schema_docs collection does not exist")
    
    schema_docs = db.collection("schema_docs")
    
    if version:
        aql = """
        FOR doc IN schema_docs
            FILTER doc.version == @version
            SORT doc.timestamp DESC
            LIMIT 1
            RETURN doc
        """
        cursor = await asyncio.to_thread(
            db.aql.execute, aql, bind_vars={"version": version}
        )
    else:
        aql = """
        FOR doc IN schema_docs
            SORT doc.timestamp DESC
            LIMIT 1
            RETURN doc
        """
        cursor = await asyncio.to_thread(
            db.aql.execute, aql
        )
    
    try:
        # Use asyncio.to_thread to convert cursor to list
        result = await asyncio.to_thread(list, cursor)
        if not result:
            raise ValueError(f"No schema document found with version {version}")
        return result[0]
    except Exception as e:
        logger.error(f"Error getting schema document: {e}")
        raise


if __name__ == "__main__":
    """
    CLI for setting up the AI knowledge database.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up the AI knowledge database")
    parser.add_argument("--host", default="http://localhost:8529", help="ArangoDB host")
    parser.add_argument("--username", default="root", help="ArangoDB username")
    parser.add_argument("--password", default="openSesame", help="ArangoDB password")
    parser.add_argument("--db-name", default="cursor_rules", help="Database name")
    parser.add_argument("--schema-path", help="Path to the schema file")
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    
    args = parser.parse_args()
    
    asyncio.run(setup_ai_knowledge_db(
        host=args.host,
        username=args.username,
        password=args.password,
        db_name=args.db_name,
        schema_path=args.schema_path,
        reset=args.reset
    )) 