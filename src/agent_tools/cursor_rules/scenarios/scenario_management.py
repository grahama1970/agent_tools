#!/usr/bin/env python3
"""Module for managing AI retrieval scenarios in the cursor rules database."""

import asyncio
from typing import Dict, List, Any
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

# Constants
COLLECTION_NAME = 'retrieval_scenarios'
REQUIRED_FIELDS = {
    "title": str,
    "description": str,
    "query_example": str,
    "expected_result_type": str,
    "validation_criteria": list,
    "priority": int
}

async def ensure_collection(db) -> None:
    """Ensure the scenarios collection exists."""
    exists = await asyncio.to_thread(
        db.has_collection, COLLECTION_NAME
    )
    if not exists:
        await asyncio.to_thread(
            db.create_collection, COLLECTION_NAME
        )

async def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Validate that a scenario has all required fields."""
    missing_fields = []
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in scenario:
            missing_fields.append(field)
        elif not isinstance(scenario[field], field_type):
            raise ValueError(f"Field {field} must be of type {field_type}")
            
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

async def store_scenario(db, scenario: Dict[str, Any]) -> str:
    """Store a new retrieval scenario in the database."""
    # Validate scenario
    await validate_scenario(scenario)
    
    # Ensure collection exists
    await ensure_collection(db)
    
    # Generate embedding
    scenario_text = f"{scenario['title']} {scenario['description']} {scenario['query_example']}"
    embedding = await asyncio.to_thread(generate_embedding, scenario_text)
    
    # Store in database using to_thread
    collection = db.collection(COLLECTION_NAME)
    result = await asyncio.to_thread(collection.insert, {
        **scenario,
        "embedding": embedding
    })
        
    return result['_key']

async def search_scenarios(db, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for scenarios using hybrid search."""
    await ensure_collection(db)
    collection = db.collection(COLLECTION_NAME)
    results = await asyncio.to_thread(hybrid_search, query, collection)
    return results[:limit]

async def get_scenario(db, key: str) -> Dict[str, Any]:
    """Retrieve a specific scenario by key."""
    await ensure_collection(db)
    
    collection = db.collection(COLLECTION_NAME)
    return await asyncio.to_thread(collection.get, key)

async def update_scenario(db, key: str, updates: Dict[str, Any]) -> None:
    """Update an existing scenario."""
    # Get existing scenario
    existing = await get_scenario(db, key)
    
    if not existing:
        raise ValueError(f"No scenario found with key {key}")
        
    # Merge updates with existing
    updated_scenario = {**existing, **updates}
    
    # Validate the merged result
    await validate_scenario(updated_scenario)
    
    # Update embedding if relevant fields changed
    if any(field in updates for field in ['title', 'description', 'query_example']):
        scenario_text = f"{updated_scenario['title']} {updated_scenario['description']} {updated_scenario['query_example']}"
        updated_scenario['embedding'] = await asyncio.to_thread(generate_embedding, scenario_text)
        
    # Update in database using to_thread
    collection = db.collection(COLLECTION_NAME)
    # Ensure the updated document includes the _key and _id fields
    updated_scenario['_key'] = key
    updated_scenario['_id'] = f"{COLLECTION_NAME}/{key}"
    await asyncio.to_thread(collection.update, updated_scenario)

# Added functions for managing sample scenarios

def insert_sample_scenarios(scenarios, collection):
    """Insert sample scenarios into the given collection.
    Returns the number of inserted documents."""
    count = 0
    for scenario in scenarios:
        result = collection.insert(scenario)
        if result:
            count += 1
    return count


def hybrid_search(term, collection):
    """Perform a hybrid search in the provided collection for documents containing the term in title, description, or query_example.
    Returns a list of matching documents."""
    term_lower = term.lower()
    results = []
    for doc in collection.all():
        if (term_lower in doc.get('title', '').lower() or 
            term_lower in doc.get('description', '').lower() or
            term_lower in doc.get('query_example', '').lower()):
            results.append(doc)
    return results 