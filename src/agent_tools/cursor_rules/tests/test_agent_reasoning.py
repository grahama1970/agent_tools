#!/usr/bin/env python3
"""
Test script for agent reasoning through hybrid search.

This script demonstrates how an agent uses hybrid search to both:
1. Determine what a user is asking about through embeddings and text search
2. Continuously evaluate its own reasoning process using the same search capabilities
"""

import asyncio
import sys
import os
import logging
from typing import List, Dict, Any, Tuple
from tabulate import tabulate  # Import tabulate directly
from arango import ArangoClient
from src.agent_tools.cursor_rules.core.cursor_rules import (
    hybrid_search,
    generate_embedding,
    create_arangosearch_view
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample knowledge base - more comprehensive for agent reasoning
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
        "title": "Code Review Checklist",
        "description": "Essential items to check during code reviews",
        "content": """
        During code reviews, check for:
        1. Proper error handling
        2. Adequate test coverage
        3. Documentation and comments
        4. Performance considerations
        5. Security vulnerabilities
        6. Adherence to coding standards
        """,
        "category": "development_process",
        "importance": "medium"
    },
    {
        "title": "Security Best Practices",
        "description": "Guidelines for writing secure code",
        "content": """
        Security best practices include:
        1. Input validation and sanitization
        2. Proper authentication and authorization
        3. Use of prepared statements for database queries
        4. Secure handling of secrets and credentials
        5. Regular dependency updates
        6. Code scanning for security vulnerabilities
        """,
        "category": "security",
        "importance": "critical"
    },
    {
        "title": "Testing Best Practices",
        "description": "Guidelines for writing effective tests",
        "content": """
        For effective testing:
        1. Follow the testing pyramid (unit, integration, system)
        2. Use test-driven development where appropriate
        3. Mock external dependencies
        4. Test edge cases and failure scenarios
        5. Ensure test isolation
        6. Use fixtures for test setup and teardown
        """,
        "category": "testing",
        "importance": "high"
    },
    {
        "title": "Performance Optimization",
        "description": "General strategies for improving code performance",
        "content": """
        To improve performance:
        1. Profile before optimizing
        2. Optimize the critical path
        3. Consider caching frequently accessed data
        4. Reduce unnecessary computations
        5. Use appropriate data structures
        6. Consider concurrent or parallel processing
        """,
        "category": "performance",
        "importance": "medium"
    },
    {
        "title": "Python Project Structure",
        "description": "Best practices for organizing Python projects",
        "content": """
        For well-structured Python projects:
        1. Use a clear package hierarchy
        2. Separate concerns (logic, data access, presentation)
        3. Follow standard directory layouts
        4. Use virtual environments
        5. Implement proper dependency management
        6. Include appropriate configuration handling
        """,
        "category": "code_organization",
        "importance": "medium"
    },
    {
        "title": "API Design Principles",
        "description": "Guidelines for designing robust APIs",
        "content": """
        Good API design includes:
        1. Consistency in endpoint naming and behavior
        2. Proper versioning
        3. Clear documentation
        4. Comprehensive error handling
        5. Authentication and authorization
        6. Rate limiting and quotas
        7. Appropriate use of HTTP methods and status codes
        """,
        "category": "api",
        "importance": "high"
    },
    {
        "title": "Logging Best Practices",
        "description": "Effective logging strategies for applications",
        "content": """
        For effective logging:
        1. Use appropriate log levels
        2. Include context in log messages
        3. Implement structured logging
        4. Configure proper log rotation
        5. Consider centralized log aggregation
        6. Don't log sensitive information
        """,
        "category": "observability",
        "importance": "medium"
    }
]

# Sample user queries
USER_QUERIES = [
    "How do I handle errors in my Python application?",
    "My database queries are slow. How can I make them faster?",
    "What's the best way to structure async code in Python?",
    "Security concerns in web applications",
    "How should I organize my Python project?",
    "Best practices for API design"
]

# Agent reasoning examples - these show the agent's thought process
AGENT_REASONING = [
    "I need to consider error handling strategies for this Python application. What are the best practices?",
    "The user is experiencing performance issues with database queries. I should recommend query optimization techniques.",
    "This seems to be an issue with async code structure. I need to recall best practices for async patterns in Python.",
    "For this security concern, I should first identify potential vulnerabilities in the user's code and then suggest mitigation strategies.",
    "The project structure looks disorganized. Let me consider Python project organization patterns to recommend.",
    "When designing this API, I need to ensure consistent naming, versioning, and proper HTTP method usage."
]

async def setup_test_db():
    """Set up a test database with a knowledge base for agent reasoning."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = await asyncio.to_thread(
            client.db, "_system", username="root", password="openSesame"
        )
        
        db_name = "test_agent_reasoning"
        
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

async def simulate_agent_reasoning(db, query: str, max_reasoning_steps: int = 3):
    """Simulate an agent's reasoning process using hybrid search."""
    print(f"\n{'='*80}\nINITIAL USER QUERY: {query}\n{'='*80}")
    
    # Step 1: Understand what the user is asking about
    print("\nSTEP 1: DETERMINING USER INTENT")
    print("-" * 80)
    
    initial_results = await hybrid_search(
        db, query, collection_name="knowledge_base", limit=3, verbose=False
    )
    
    print(format_results(initial_results, query))
    
    if not initial_results:
        print("No relevant information found for the initial query.")
        return
    
    # Extract the most relevant topic from search results
    # Handle tuple or dict format
    if isinstance(initial_results[0], tuple):
        rule_data = initial_results[0][0] if len(initial_results[0]) > 0 else {}
        main_topic = rule_data.get("title", "Unknown Topic") if isinstance(rule_data, dict) else "Unknown Topic"
    else:
        rule = initial_results[0].get("rule", {})
        main_topic = rule.get("title", "Unknown Topic")
        
    print(f"\nMain topic identified: {main_topic}")
    
    # Step 2: Generate agent reasoning based on the identified topic
    reasoning = f"The user is asking about {main_topic}. I need to provide information on best practices and common issues."
    print(f"\nSTEP 2: INITIAL AGENT REASONING\n{'-'*80}\n{reasoning}")
    
    # Step 3: Continuously refine reasoning through additional hybrid searches
    current_reasoning = reasoning
    
    for i in range(max_reasoning_steps):
        print(f"\nSTEP {i+3}: REFINING REASONING - SEARCH ITERATION {i+1}")
        print("-" * 80)
        
        # Use the current reasoning as a search query to find more relevant information
        reasoning_results = await hybrid_search(
            db, current_reasoning, collection_name="knowledge_base", limit=2, verbose=False
        )
        
        print(format_results(reasoning_results, current_reasoning))
        
        if not reasoning_results:
            print("No additional relevant information found.")
            break
        
        # Update reasoning based on new information - handle tuple or dict format
        if isinstance(reasoning_results[0], tuple):
            rule_data = reasoning_results[0][0] if len(reasoning_results[0]) > 0 else {}
            new_info = rule_data.get("title", "Unknown Topic") if isinstance(rule_data, dict) else "Unknown Topic"
        else:
            rule = reasoning_results[0].get("rule", {})
            new_info = rule.get("title", "Unknown Topic")
            
        current_reasoning = f"Based on {main_topic}, I should also consider {new_info}. What specific recommendations can I provide?"
        
        print(f"\nUpdated reasoning: {current_reasoning}")
        
    # Step 4: Final response synthesis
    print(f"\nSTEP {max_reasoning_steps+3}: FINAL RESPONSE SYNTHESIS")
    print("-" * 80)
    
    # Perform one final search with the refined reasoning
    final_results = await hybrid_search(
        db, current_reasoning, collection_name="knowledge_base", limit=5, verbose=False
    )
    
    print(format_results(final_results, current_reasoning))
    
    # Synthesize a response based on all gathered information
    if final_results:
        print("\nFINAL AGENT RESPONSE:")
        print("-" * 80)
        
        # Extract main result - handle tuple or dict format
        if isinstance(final_results[0], tuple):
            main_result = final_results[0][0] if len(final_results[0]) > 0 else {}
        else:
            main_result = final_results[0].get("rule", {})
            
        response = f"Based on your question about {main_topic}, here are some key recommendations:\n\n"
        response += main_result.get("content", "No specific content available.")
        print(response)

async def run_agent_reasoning_tests():
    """Run tests demonstrating agent reasoning using hybrid search."""
    try:
        # Set up the test database
        logger.info("Setting up test database...")
        db = await setup_test_db()
        
        if not db:
            logger.error("Failed to set up test database")
            return
        
        # Test with a few sample user queries
        test_queries = [
            "How do I handle errors in my Python application?",
            "My database queries are slow. How can I make them faster?",
            "What's the best way to structure async code in Python?"
        ]
        
        for query in test_queries:
            await simulate_agent_reasoning(db, query)
            print("\n\n" + "="*100 + "\n\n")
            
        # Also test with actual agent reasoning examples
        print("\n\nTESTING AGENT SELF-EVALUATION OF REASONING\n" + "="*80)
        agent_reasoning = [
            "I need to consider error handling strategies for this Python application.",
            "This seems to be an issue with async code structure in Python."
        ]
        
        for reasoning in agent_reasoning:
            print(f"\nEVALUATING AGENT REASONING: {reasoning}")
            print("-" * 80)
            
            # Search based on the agent's own reasoning
            reasoning_results = await hybrid_search(
                db, reasoning, collection_name="knowledge_base", limit=3, verbose=False
            )
            
            print(format_results(reasoning_results, reasoning))
            
            if reasoning_results:
                # Handle tuple or dict format
                if isinstance(reasoning_results[0], tuple):
                    rule_data = reasoning_results[0][0] if len(reasoning_results[0]) > 0 else {}
                    most_relevant = rule_data.get("title", "Unknown Topic") if isinstance(rule_data, dict) else "Unknown Topic"
                else:
                    rule = reasoning_results[0].get("rule", {})
                    most_relevant = rule.get("title", "Unknown Topic")

                print(f"\nMost relevant knowledge area: {most_relevant}")
                print(f"This confirms the agent's reasoning is on the right track and should focus on {most_relevant}.")
            else:
                print("\nNo relevant knowledge found for this reasoning path. The agent should reconsider its approach.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Running agent reasoning tests with hybrid search...")
    asyncio.run(run_agent_reasoning_tests()) 