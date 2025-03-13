#!/usr/bin/env python3
"""
Management of common query scenarios for AI agents.

This module provides functionality to store, retrieve, and search through
common query scenarios that AI agents frequently need to handle.

It builds on Phase 2 functionality (cursor_rules) to store these scenarios
in the ArangoDB database and make them searchable.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import os
import json

from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    bm25_keyword_search,
    semantic_search,
    hybrid_search
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Collection name for query scenarios
COLLECTION_NAME = "query_scenarios"

# Required fields for query scenarios
REQUIRED_FIELDS = [
    "title",
    "description",
    "query_example",
    "expected_result_format",
    "category"
]

# Optional fields
OPTIONAL_FIELDS = [
    "priority",
    "tags",
    "documentation_links",
    "related_scenarios"
]

async def ensure_scenarios_collection(db):
    """
    Ensure the query_scenarios collection exists in the database.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        collection: The query_scenarios collection
    """
    collections = await asyncio.to_thread(db.collections)
    collection_names = [c['name'] for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        logger.info(f"Creating {COLLECTION_NAME} collection")
        await asyncio.to_thread(db.create_collection, COLLECTION_NAME)
    
    return await asyncio.to_thread(db.collection, COLLECTION_NAME)


async def validate_scenario(scenario: Dict[str, Any]) -> bool:
    """
    Validate that a scenario has all required fields.
    
    Args:
        scenario: Dictionary containing the scenario
        
    Returns:
        bool: True if valid, False otherwise
    """
    for field in REQUIRED_FIELDS:
        if field not in scenario:
            logger.error(f"Scenario is missing required field: {field}")
            return False
    
    # Title should be unique
    if not scenario.get("title"):
        logger.error("Scenario title cannot be empty")
        return False
        
    return True


async def store_scenario(db, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store a query scenario in the database.
    
    Args:
        db: ArangoDB database connection
        scenario: Dictionary containing the scenario
        
    Returns:
        Dict: The stored scenario with _key
        
    Raises:
        ValueError: If the scenario is invalid
    """
    # Validate scenario
    if not await validate_scenario(scenario):
        raise ValueError("Invalid scenario: missing required fields")
    
    # Ensure collection exists
    collection = await ensure_scenarios_collection(db)
    
    # Check if scenario with same title already exists
    cursor = await asyncio.to_thread(
        lambda: collection.find({"title": scenario["title"]}, limit=1)
    )
    existing_scenarios = [doc for doc in cursor]
    
    if existing_scenarios:
        # Update existing scenario
        existing_key = existing_scenarios[0]["_key"]
        logger.info(f"Updating existing scenario with key {existing_key}")
        await asyncio.to_thread(
            collection.update, existing_key, scenario
        )
        return await asyncio.to_thread(collection.get, existing_key)
    else:
        # Insert new scenario
        logger.info(f"Inserting new scenario: {scenario['title']}")
        result = await asyncio.to_thread(collection.insert, scenario, return_new=True)
        return result["new"]


async def get_scenario_by_title(db, title: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a scenario by its title.
    
    Args:
        db: ArangoDB database connection
        title: The title of the scenario
        
    Returns:
        Dict or None: The scenario or None if not found
    """
    collection = await ensure_scenarios_collection(db)
    
    cursor = await asyncio.to_thread(
        lambda: collection.find({"title": title}, limit=1)
    )
    scenarios = [doc for doc in cursor]
    
    return scenarios[0] if scenarios else None


async def list_all_scenarios(db) -> List[Dict[str, Any]]:
    """
    List all query scenarios.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        List: All scenarios in the database
    """
    collection = await ensure_scenarios_collection(db)
    
    cursor = await asyncio.to_thread(
        lambda: collection.all()
    )
    
    return [doc for doc in cursor]


async def search_scenarios(
    db, 
    query: str, 
    search_type: str = "hybrid",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for scenarios matching a query.
    
    Args:
        db: ArangoDB database connection
        query: The search query
        search_type: Type of search ("keyword", "semantic", or "hybrid")
        limit: Maximum number of results
        
    Returns:
        List: Matching scenarios
    """
    await ensure_scenarios_collection(db)
    
    if search_type == "keyword":
        results = await bm25_keyword_search(
            db, 
            query, 
            collection_name=COLLECTION_NAME,
            limit=limit
        )
    elif search_type == "semantic":
        results = await semantic_search(
            db, 
            query, 
            collection_name=COLLECTION_NAME,
            limit=limit
        )
    else:  # hybrid (default)
        results = await hybrid_search(
            db, 
            query, 
            collection_name=COLLECTION_NAME,
            limit=limit
        )
    
    # Process results to return just the scenario documents
    processed_results = []
    for result in results:
        # Extract the document based on result format
        if isinstance(result, tuple) and len(result) >= 2:
            # Hybrid search returns (doc, score) tuple
            doc = result[0].get("rule", {})
        else:
            # Other search methods might have different formats
            doc = result.get("rule", {})
            
        if doc:
            processed_results.append(doc)
    
    return processed_results


async def import_scenarios_from_file(db, file_path: str) -> List[Dict[str, Any]]:
    """
    Import scenarios from a JSON file.
    
    Args:
        db: ArangoDB database connection
        file_path: Path to JSON file containing scenarios
        
    Returns:
        List: The imported scenarios
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scenario file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    
    imported = []
    for scenario in scenarios:
        if await validate_scenario(scenario):
            stored = await store_scenario(db, scenario)
            imported.append(stored)
        else:
            logger.warning(f"Skipping invalid scenario: {scenario.get('title', 'unknown')}")
    
    return imported 