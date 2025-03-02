import subprocess
import pyperclip
import csv
import os
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid
from arango.client import ArangoClient
import litellm
from litellm import acompletion
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Tuple
from deepmerge import always_merger
from dotenv import load_dotenv
import regex as re


from search_tool.arangodb.generate_schema_for_llm.utils.config_utils import (
    initialize_environment,
    validate_config,
)

from search_tool.arangodb.generate_schema_for_llm.utils.model_utils import (
    FieldDescription, 
    CollectionDescription, 
    SchemaDescription, 
)
from search_tool.arangodb.generate_schema_for_llm.utils.llm_utils import (
    describe_fields_and_collections, 
    describe_view, 
    describe_analyzer
)

from search_tool.arangodb.generate_schema_for_llm.utils.arango_utils import (
    discover_relationships, 
    get_sample_rows, 
    initialize_database, 
    insert_schema_into_collection, 
    load_aql_queries, 
    truncate_sample_data
)

from search_tool.arangodb.generate_schema_for_llm.utils.schema_utils import generate_sparse_schema



"""
Main function for generating the ArangoDBschema for the LLM to understand the database
and make AQL queries to the database.
"""
async def generate_schema_for_llm(config: Dict) -> None:

    arango_config = config.get("arango_config", {})
    llm_config = config.get("llm_config", {})
    directories = config.get("directories", {})
    allowed_analyzers = arango_config.get("allowed_analyzers", [])  
    excluded_collections = config.get("excluded_collections", [])


    # get the db connection
    db = await asyncio.to_thread(initialize_database, arango_config)

    # Get excluded collections from config, skip if None or empty
    if excluded_collections:
        logger.info(f"Excluding collections: {excluded_collections}")

    # Discover relationships before building the schema
    relationships = discover_relationships(db, config)
    logger.info(f"Discovered {len(relationships)} relationships in the database")

    # Setup the Schema structure
    schema = SchemaDescription(
        collections=[],
        views=[],
        analyzers=[],
        example_queries=[],
        relationships=relationships,  # Use the discovered relationships
    )

    # Load AQL queries from the specified directory
    aql_directory = directories.get("aql_directory", None)
    if aql_directory and aql_directory.exists():
        schema.example_queries = load_aql_queries(aql_directory, config.get("include_queries"))
    else:
        logger.warning(f"AQL queries directory not found: {aql_directory}")

    # Get all collections
    tasks = []
    collections = await asyncio.to_thread(db.collections)
    for collection in collections:
        collection_name = collection["name"]
        collection_type = collection["type"]
        
        # Skip private collections (starting with _) and excluded collections
        if(
            collection_name.startswith('_') or 
            collection_name in excluded_collections
        ):
            logger.info(f"Skipping collection: {collection_name}")
            continue
            
        # Verify collection exists
        # If so Ask the LLM to 'describe_fields_and_collections'
        try:    
            sample_rows = await asyncio.to_thread(get_sample_rows, db, collection_name)
            sample_rows = truncate_sample_data(sample_rows)
            request_id = f"collection_{collection_name}_{uuid.uuid4()}"  # Add UUID4"
            tasks.append(
                describe_fields_and_collections(
                    db, collection_name, collection_type, sample_rows, request_id, config))
            
        except Exception as e:
            logger.error(f"Error processing collection {collection_name}: {e}")
            continue

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error processing collection: {result}")
            continue

        # Create FieldDescription objects
        field_descriptions = [
            FieldDescription(
                name=field, 
                description=desc.description,
                type=desc.type
            )
            for field, desc in result["field_descriptions"].items()
        ]

        # Create CollectionDescription object
        collection_details = CollectionDescription(
            name=result["collection_description"].name,
            type=result["collection_description"].type,  # Get type from the collection description
            fields=field_descriptions,
            description=result["collection_description"].description,
        )
        schema.collections.append(collection_details)

    # Get all views
    tasks = []
    views = await asyncio.to_thread(db.views)  # Fetch the list of views

    for view in views:
        view_name = view["name"]
        view_dict = await asyncio.to_thread(db.view, view_name)  # strange that view_dict != view
        try:
            # Directly process the view without checking existence
            request_id = f"view_{view_name}"
            tasks.append(describe_view(view_dict, request_id, config))
        except Exception as e:
            logger.error(f"Error processing view {view_name}: {e}")
            continue

    # Run all tasks concurrently
    view_results = []
    with tqdm(total=len(tasks), desc="Processing views") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if isinstance(result, Exception):
                logger.error(f"Error processing view: {result}")
                continue
            schema.views.append(result)
            pbar.update(1)

    # Get all analyzers
    tasks = []
    analyzers = await asyncio.to_thread(db.analyzers)
    for analyzer in analyzers:
        analyzer_name = analyzer["name"]

        # Skip analyzers that don't match our allowed patterns
        has_allowed_pattern = any(pattern in analyzer_name for pattern in allowed_analyzers)
        if not has_allowed_pattern:
            logger.info(f"Skipping analyzer: {analyzer_name}")
            continue
        
        try:    
            request_id = f"analyzer_{analyzer_name}"
            tasks.append(describe_analyzer(analyzer, request_id, config))
        except Exception as e:
            logger.error(f"Error processing analyzer {analyzer_name}: {e}")
            continue

    # Run all tasks concurrently
    with tqdm(total=len(tasks), desc="Processing analyzers") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if isinstance(result, Exception):
                logger.error(f"Error processing analyzer: {result}")
                continue
            schema.analyzers.append(result)
            pbar.update(1)

    # Print the schema
    logger.info("Generated schema:")
    logger.debug(schema.model_dump_json(indent=4))

    # Generate sparse schema along with complete schema (for low context LLMs)
    sparse_schema = generate_sparse_schema(schema.model_dump())
    logger.info("Generated sparse schema:")
    logger.debug(sparse_schema)

    # Insert the schema into the collection
    success = await asyncio.to_thread(
        insert_schema_into_collection,
        db, 
        schema.model_dump(), 
        sparse_schema
    )
    if success:
        logger.info("Schema and Sparse Schema successfully stored in the collection.")
    else:
        logger.error("Failed to store the schema and sparse schema in the collection.")

# Centralized configuration
async def main() -> None:
    import pyperclip
    try:
        # Load base configuration from file
        from search_tool.settings.config import config

        # get redis up and running for litellm caching
        # get environmental variables and return the project root
        # Return the project root 
        project_dir = initialize_environment() 

        # Define the updates to the config
        updates = {
            "llm_config": {
                "model": "openai/gpt-4o-mini",  # Example model selection
                "caching": True,
            },
            "excluded_collections": 
                [
                    'microsoft_support', 'message_log', 'glossary', 'database_schema'
                ],
            # AQL queries to include to help the LLM understand the database
            "include_queries": ["bm25_search", "cosine_similarity_search"],  
        }

        # Merge the updates into the config
        config = always_merger.merge(config, updates)
        
        result = await generate_schema_for_llm(config)
        logger.info(f"Schema generation result: {result}")
        # return result
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    asyncio.run(main())