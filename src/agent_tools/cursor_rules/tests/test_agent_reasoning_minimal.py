#!/usr/bin/env python3
"""
Minimal test script for agent reasoning through hybrid search.

This script demonstrates the core functionality with a single query to verify the fix.
"""

import asyncio
import sys
import os
import logging
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
from arango import ArangoClient
from src.agent_tools.cursor_rules.core.cursor_rules import (
    hybrid_search,
    generate_embedding,
    create_arangosearch_view
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample knowledge base - just a few entries for minimal testing
KNOWLEDGE_BASE = [
    {
        "title": "Error Handling Pattern",
        "description": "Best practices for handling exceptions in Python",
        "content": """
        When handling errors in Python:
        1. Use specific exception types instead of catching all exceptions
        2. Clean up resources in finally blocks
        3. Log exceptions with context
        """,
        "category": "error_handling",
        "importance": "high"
    },
    {
        "title": "Database Query Optimization",
        "description": "Techniques for optimizing database queries and indexing",
        "content": """
        To optimize database queries:
        1. Create appropriate indexes based on query patterns
        2. Use query profiling to identify slow queries
        3. Limit result sets to what's needed
        """,
        "category": "performance",
        "importance": "high"
    },
    {
        "title": "Async Pattern Implementation",
        "description": "How to implement async patterns in Python",
        "content": """
        When working with async code:
        1. Use asyncio.gather for concurrent tasks
        2. Avoid mixing sync and async code
        3. Be careful with loop management
        """,
        "category": "async_patterns",
        "importance": "medium"
    }
]

async def setup_test_db():
    """Set up a test database with a minimal knowledge base for agent reasoning."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = await asyncio.to_thread(
            client.db, "_system", username="root", password="openSesame"
        )
        
        db_name = "test_agent_reasoning_minimal"
        
        # Create the database if it doesn't exist
        db_exists = await asyncio.to_thread(sys_db.has_database, db_name)
        if not db_exists:
            logger.info(f"Creating database: {db_name}")
            await asyncio.to_thread(sys_db.create_database, db_name)
        else:
            logger.info(f"Database {db_name} already exists")
        
        # Connect to the test database
        db = await asyncio.to_thread(
            client.db, db_name, username="root", password="openSesame"
        )
        
        # Create the collection for the knowledge base
        collection_name = "knowledge_base"
        collection_exists = await asyncio.to_thread(db.has_collection, collection_name)
        if not collection_exists:
            logger.info(f"Creating collection: {collection_name}")
            await asyncio.to_thread(db.create_collection, collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists")
        
        # Create the ArangoSearch view
        view_name = f"{collection_name}_view"
        await asyncio.to_thread(create_arangosearch_view, db, collection_name, view_name)
        
        # Insert knowledge base entries with embeddings if collection is empty
        kb_collection = await asyncio.to_thread(db.collection, collection_name)
        
        # Check if collection is empty
        count_query = f"RETURN LENGTH(FOR doc IN {collection_name} RETURN 1)"
        cursor = await asyncio.to_thread(db.aql.execute, count_query)
        count = await asyncio.to_thread(next, cursor, 0)
        
        if count == 0:
            logger.info("Inserting knowledge base entries")
            for entry in KNOWLEDGE_BASE:
                # Generate embedding from entry content
                text_to_embed = f"{entry['title']} {entry['description']} {entry['content']}"
                embedding_result = generate_embedding(text_to_embed)
                
                if embedding_result and "embedding" in embedding_result:
                    entry["embedding"] = embedding_result["embedding"]
                    entry["embedding_metadata"] = embedding_result.get("metadata", {})
                    
                # Insert the entry
                await asyncio.to_thread(kb_collection.insert, entry)
                
            logger.info(f"Inserted {len(KNOWLEDGE_BASE)} knowledge base entries")
        else:
            logger.info(f"Collection {collection_name} already contains {count} documents")
        
        return db
    except Exception as e:
        logger.error(f"Error setting up test database: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_results(results: List[Dict], query: str) -> str:
    """Format search results for display."""
    try:
        if not results:
            return "No relevant information found."
        
        rows = []
        for i, result in enumerate(results, 1):
            # Check if result is a tuple or dictionary and handle accordingly
            if isinstance(result, tuple):
                # Handle tuple format (likely direct from AQL)
                rule_data = result[0] if len(result) > 0 else {}
                hybrid_score = result[1] if len(result) > 1 else 0
                bm25_score = result[2] if len(result) > 2 else 0
                similarity_score = result[3] if len(result) > 3 else 0
                title = rule_data.get("title", "N/A") if isinstance(rule_data, dict) else "N/A"
                description = rule_data.get("description", "N/A") if isinstance(rule_data, dict) else "N/A"
            else:
                # Handle dictionary format with rule field
                rule = result.get("rule", {})
                title = rule.get("title", "N/A")
                description = rule.get("description", "N/A")
                hybrid_score = result.get("hybrid_score", 0)
                bm25_score = result.get("bm25_score", 0)
                similarity_score = result.get("similarity_score", 0)
            
            if isinstance(description, str) and len(description) > 50:
                description = description[:47] + "..."
                
            rows.append([i, title, description, hybrid_score, bm25_score, similarity_score])
        
        headers = ["#", "Title", "Description", "Hybrid Score", "BM25 Score", "Vector Score"]
        table = tabulate(rows, headers=headers, tablefmt="grid")
        
        return f"\nRESULTS FOR: {query}\n{'-'*80}\n{table}"
    except Exception as e:
        logger.error(f"Error in format_results: {e}")
        import traceback
        traceback.print_exc()
        return f"Error formatting results: {e}"

async def test_single_query(db, query: str):
    """Test a single query to verify our fix works."""
    print(f"\n{'='*80}\nTESTING QUERY: {query}\n{'='*80}")
    
    # Step 1: Run the hybrid search
    results = await hybrid_search(
        db, query, collection_name="knowledge_base", limit=3, verbose=True
    )
    
    # Step 2: Format and display results
    print(format_results(results, query))
    
    # Step 3: Extract the most relevant topic
    if results:
        # Handle tuple or dict format
        if isinstance(results[0], tuple):
            print("\nResult is a tuple with structure:")
            for i, item in enumerate(results[0]):
                print(f"  Index {i}: {type(item)}")
            
            rule_data = results[0][0] if len(results[0]) > 0 else {}
            topic = rule_data.get("title", "Unknown Topic") if isinstance(rule_data, dict) else "Unknown Topic"
        else:
            print("\nResult is a dictionary with structure:")
            for key in results[0].keys():
                print(f"  Key: {key}")
            
            rule = results[0].get("rule", {})
            topic = rule.get("title", "Unknown Topic")
        
        print(f"\nSuccessfully extracted topic: {topic}")
    else:
        print("\nNo results found.")

async def run_minimal_test():
    """Run a minimal test to validate our fix."""
    try:
        # Set up the test database
        logger.info("Setting up test database...")
        db = await setup_test_db()
        
        if not db:
            logger.error("Failed to set up test database")
            return
        
        # Test with a single query - using a simpler direct query that should match
        await test_single_query(db, "error handling")
        
    except Exception as e:
        print(f"Error in minimal test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Running minimal agent reasoning test to verify the fix...")
    asyncio.run(run_minimal_test()) 