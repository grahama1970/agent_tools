#!/usr/bin/env python3
"""
CLI test for hybrid search to demonstrate finding and not finding results.

This script allows testing different search queries and threshold parameters
to show scenarios where:
1. We successfully find relevant results
2. We don't find results despite being confident a rule is relevant
"""

import asyncio
import argparse
import sys
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

# Sample knowledge base for testing
KNOWLEDGE_BASE = [
    {
        "title": "Error Handling Pattern",
        "description": "Best practices for handling exceptions in Python",
        "content": """
        When handling errors in Python:
        1. Use specific exception types instead of catching all exceptions
        2. Clean up resources in finally blocks
        3. Log exceptions with context
        4. Consider using context managers for resource management
        5. For async code, ensure exceptions are properly propagated
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
        4. Use proper filtering in the database, not in application code
        5. For text search, use specialized full-text indexes
        6. Consider denormalization for read-heavy workloads
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
        4. Use proper cancellation and timeout handling
        5. Consider connection pooling for IO-bound operations
        """,
        "category": "async_patterns",
        "importance": "medium"
    },
    {
        "title": "Text Processing Utilities",
        "description": "Utilities for processing and normalizing text",
        "content": """
        For text processing:
        1. Use standardized libraries for text normalization
        2. Consider Unicode normalization for international text
        3. Implement proper tokenization for language-specific needs
        4. Apply stemming or lemmatization as needed
        5. Use regular expressions sparingly and carefully
        """,
        "category": "text_processing",
        "importance": "medium"
    }
]

async def setup_test_db():
    """Set up a test database with the knowledge base."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = await asyncio.to_thread(
            client.db, "_system", username="root", password="openSesame"
        )
        
        db_name = "test_hybrid_search_cli"
        
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
        
        return db, collection_name
    except Exception as e:
        logger.error(f"Error setting up test database: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
        
        return f"\nRESULTS FOR QUERY: {query}\n{'-'*80}\n{table}"
    except Exception as e:
        logger.error(f"Error in format_results: {e}")
        import traceback
        traceback.print_exc()
        return f"Error formatting results: {e}"

async def run_success_scenario(db, collection_name):
    """Run a scenario where the hybrid search successfully finds relevant results."""
    print("\n" + "="*80)
    print(" SUCCESS SCENARIO: Finding relevant results for 'error handling in Python'")
    print("="*80)
    
    query = "error handling in Python"
    print(f"\nQuery: {query}")
    print("Expected: Should find the 'Error Handling Pattern' rule")
    
    results = await hybrid_search(
        db, query, collection_name=collection_name, limit=3, 
        verbose=True, hybrid_score_threshold=0.15
    )
    
    print(format_results(results, query))
    
    if results:
        print("\n✅ SUCCESS: Found relevant results with default thresholds!")
    else:
        print("\n❌ FAILURE: No results found, even though we expected to find some.")

async def run_failure_scenario(db, collection_name):
    """Run a scenario where the hybrid search fails to find results despite semantic relevance."""
    print("\n" + "="*80)
    print(" FAILURE SCENARIO: Failing to find results for 'text normalization utilities'")
    print("="*80)
    
    # First, set a higher threshold to force failure
    query = "text normalization utilities"
    print(f"\nQuery: {query}")
    print("Expected: Should find 'Text Processing Utilities' rule but will fail due to high threshold")
    
    # Using a very high hybrid score threshold to force failure
    results = await hybrid_search(
        db, query, collection_name=collection_name, limit=3, 
        verbose=True, hybrid_score_threshold=0.8
    )
    
    print(format_results(results, query))
    
    if not results:
        print("\n✅ SUCCESS IN DEMONSTRATING FAILURE: No results found with high threshold (0.8),")
        print("   even though 'Text Processing Utilities' is semantically relevant.")
        
        # Now try with a lower threshold to show that the document is actually relevant
        print("\nNow trying with a lower threshold (0.15) to demonstrate the document is relevant:")
        results_with_lower_threshold = await hybrid_search(
            db, query, collection_name=collection_name, limit=3, 
            verbose=True, hybrid_score_threshold=0.15
        )
        
        print(format_results(results_with_lower_threshold, query))
        
        if results_with_lower_threshold:
            print("\nWith a lower threshold, we can find the relevant document,")
            print("demonstrating that threshold configuration is critical for balancing precision vs. recall.")
    else:
        print("\n❓ UNEXPECTED: Found results despite high threshold.")

async def cli_test():
    """Run the CLI test for hybrid search."""
    try:
        # Setup test database
        print("\nSetting up test database...")
        db, collection_name = await setup_test_db()
        
        if not db:
            print("Failed to set up test database. Exiting.")
            return
        
        # Run success scenario
        await run_success_scenario(db, collection_name)
        
        # Run failure scenario
        await run_failure_scenario(db, collection_name)
        
        print("\n" + "="*80)
        print(" SUMMARY")
        print("="*80)
        print("\nThis demonstration shows:")
        print("1. How hybrid search successfully finds relevant results for straightforward queries")
        print("2. How threshold configuration can cause the system to miss semantically relevant results")
        print("3. The importance of proper threshold tuning based on the use case")
        
    except Exception as e:
        print(f"Error in CLI test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Running CLI test for hybrid search...")
    asyncio.run(cli_test()) 