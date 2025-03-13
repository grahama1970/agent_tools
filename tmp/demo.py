#!/usr/bin/env python3
import asyncio
import os
import sys
from tabulate import tabulate
from arango import ArangoClient
from agent_tools.cursor_rules.utils.test_state import _search_test_failures, ensure_test_collections
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

async def demo():
    """Demonstrate hybrid search for test failures."""
    print("\n🔍 HYBRID SEARCH DEMONSTRATION")
    print("============================")
    print("Following the established patterns from cursor_rules.py")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('cursor_rules_test', username='root', password='openSesame')
    
    # Ensure collections exist
    collections = await ensure_test_collections(db)
    print(f"Collections: {', '.join(collections.keys())}")
    
    # Get test failures collection
    test_failures = db.collection('test_failures')
    
    # Truncate the collection to start fresh
    test_failures.truncate()
    print(f"Truncated test_failures collection, starting fresh")
    
    # Create sample test failures with embeddings
    embeddings = {
        "async_operation": generate_embedding("async operation event loop closed failure"),
        "db_connection": generate_embedding("database connection failure credentials"),
        "memory_error": generate_embedding("memory allocation error buffer overflow"),
        "timeout": generate_embedding("operation timeout network delay")
    }
    
    # Insert test data
    print("\n📋 INSERTING TEST DATA:")
    test_data = [
        {
            "_key": "test1",
            "test_name": "test_async_operation",
            "error_message": "RuntimeError: Event loop is closed",
            "analysis": "The async operation failed due to event loop closure. This typically happens when trying to use an event loop after it has been closed.",
            "timestamp": "2025-03-11T08:44:00",
            "embedding": embeddings["async_operation"]["embedding"]
        },
        {
            "_key": "test2",
            "test_name": "test_database_connection",
            "error_message": "ConnectionError: Could not connect to database",
            "analysis": "Database connection failed due to incorrect credentials or server being unavailable.",
            "timestamp": "2025-03-11T08:44:01",
            "embedding": embeddings["db_connection"]["embedding"]
        },
        {
            "_key": "test3",
            "test_name": "test_memory_allocation",
            "error_message": "MemoryError: Failed to allocate buffer",
            "analysis": "Memory allocation error indicates the system is running out of available memory.",
            "timestamp": "2025-03-11T08:44:02",
            "embedding": embeddings["memory_error"]["embedding"]
        },
        {
            "_key": "test4",
            "test_name": "test_api_timeout",
            "error_message": "TimeoutError: Operation timed out after 30 seconds",
            "analysis": "Network operation timed out, possibly due to high latency or server being overloaded.",
            "timestamp": "2025-03-11T08:44:03",
            "embedding": embeddings["timeout"]["embedding"]
        }
    ]
    
    for data in test_data:
        test_failures.insert(data)
        print(f"  ✅ Inserted {data['test_name']}")
    
    # Run hybrid searches
    print("\n🔎 EXECUTING HYBRID SEARCHES:")
    print("----------------------------")
    
    # Test different queries to demonstrate both semantic and keyword matching
    queries = [
        "async operation",            # Direct match to test1
        "connection database",        # Direct match to test2
        "memory buffer problems",     # Semantic match to test3
        "slow network responses",     # Semantic match to test4
        "error in test execution"     # General match that should hit multiple
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)
        
        # Run the hybrid search
        results = await _search_test_failures(db, query)
        
        # Display results
        if results:
            headers = ['Test Name', 'BM25 Score', 'Vector Score', 'Hybrid Score', 'Error Message']
            table_data = []
            
            for result in results:
                table_data.append([
                    result.get('test_name', 'Unknown'),
                    f"{result.get('bm25_score', 0):.2f}",
                    f"{result.get('vector_score', 0):.2f}",
                    f"{result.get('score', 0):.2f}",
                    result.get('error_message', 'Unknown')[:50],
                ])
            
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            print('No matching test failures found.')
    
    print("\n✅ Demonstration complete!")

if __name__ == "__main__":
    asyncio.run(demo()) 