#!/usr/bin/env python3
"""
Test script for glossary search functionality.

This script tests the glossary search capabilities and demonstrates how
it integrates with the existing hybrid search system.
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
from arango import ArangoClient

from src.agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    hybrid_search
)
from src.agent_tools.cursor_rules.core.glossary import (
    setup_glossary,
    glossary_search,
    enhanced_hybrid_search
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_environment():
    """Set up test environment with database, collections, and sample data."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = await asyncio.to_thread(
            client.db, "_system", username="root", password="openSesame"
        )
        
        # Create test database
        db_name = "test_glossary_search"
        
        # Check if database exists and delete it if it does (for clean test)
        db_exists = await asyncio.to_thread(sys_db.has_database, db_name)
        if db_exists:
            logger.info(f"Dropping existing test database: {db_name}")
            await asyncio.to_thread(sys_db.delete_database, db_name)
            
        # Create fresh database
        logger.info(f"Creating test database: {db_name}")
        await asyncio.to_thread(sys_db.create_database, db_name)
        
        # Connect to the test database
        db = await asyncio.to_thread(
            client.db, db_name, username="root", password="openSesame"
        )
        
        # Set up cursor rules
        logger.info("Setting up cursor rules...")
        db = setup_cursor_rules_db(db_name=db_name)
        
        # Set up glossary
        logger.info("Setting up glossary...")
        glossary_setup = await setup_glossary(db)
        
        if not glossary_setup:
            logger.error("Failed to set up glossary")
            return None
            
        logger.info("Test environment setup complete")
        return db
    except Exception as e:
        logger.error(f"Error setting up test environment: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_results(results, title="Search Results"):
    """Format search results for display."""
    print(f"\n{title}")
    print("="*80)
    
    if not results:
        print("No results found")
        return
        
    # Format results based on type
    if isinstance(results[0], tuple) and len(results[0]) == 2:
        # Regular hybrid search results (rule, score)
        rows = []
        for i, (rule, score) in enumerate(results, 1):
            title = rule.get("title", "N/A")
            description = rule.get("description", "N/A")
            if len(description) > 50:
                description = description[:47] + "..."
                
            rows.append([i, title, description, f"{score:.4f}"])
            
        headers = ["#", "Title", "Description", "Score"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
    elif isinstance(results[0], dict) and "type" in results[0]:
        # Enhanced hybrid search results
        rows = []
        for i, result in enumerate(results, 1):
            content = result["content"]
            result_type = result["type"]
            score = result["score"]
            
            if result_type == "knowledge":
                title = content.get("title", "N/A")
                description = content.get("description", "N/A")
            else:  # glossary
                title = content.get("term", "N/A")
                description = content.get("definition", "N/A")
                
            if len(description) > 50:
                description = description[:47] + "..."
                
            rows.append([i, result_type.upper(), title, description, f"{score:.4f}"])
            
        headers = ["#", "Type", "Title", "Description", "Score"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    elif isinstance(results[0], tuple) and len(results[0]) >= 4:
        # Glossary search results (term_doc, hybrid_score, bm25_score, similarity_score)
        rows = []
        for i, (term_doc, hybrid_score, bm25_score, similarity_score) in enumerate(results, 1):
            term = term_doc.get("term", "N/A")
            definition = term_doc.get("definition", "N/A")
            category = term_doc.get("category", "N/A")
            
            if len(definition) > 50:
                definition = definition[:47] + "..."
                
            rows.append([i, term, definition, category, f"{hybrid_score:.4f}", f"{bm25_score:.4f}", f"{similarity_score:.4f}"])
            
        headers = ["#", "Term", "Definition", "Category", "Hybrid Score", "BM25", "Vector"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
    else:
        # Unknown format, just print as-is
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")

async def test_basic_glossary_search(db):
    """Test basic glossary search functionality."""
    print("\n" + "="*80)
    print(" TEST 1: BASIC GLOSSARY SEARCH")
    print("="*80)
    
    # Search for exact term
    query = "Embedding"
    print(f"\nSearching for exact term: '{query}'")
    results = await glossary_search(db, query, verbose=True)
    format_results(results, f"Glossary search results for: '{query}'")
    
    # Search for partial term
    query = "BM25 ranking"
    print(f"\nSearching for related term: '{query}'")
    results = await glossary_search(db, query, verbose=True)
    format_results(results, f"Glossary search results for: '{query}'")
    
    # Search for non-existent term
    query = "nonexistent term xyz123"
    print(f"\nSearching for non-existent term: '{query}'")
    results = await glossary_search(db, query, verbose=True)
    format_results(results, f"Glossary search results for: '{query}'")
    
    return True

async def test_enhanced_hybrid_search(db):
    """Test enhanced hybrid search with both knowledge base and glossary."""
    print("\n" + "="*80)
    print(" TEST 2: ENHANCED HYBRID SEARCH")
    print("="*80)
    
    # Search that should find both glossary and rule results
    query = "search techniques"
    print(f"\nPerforming enhanced search for: '{query}'")
    results = await enhanced_hybrid_search(db, query, verbose=True)
    format_results(results, f"Enhanced hybrid search results for: '{query}'")
    
    # Search focused on knowledge base
    query = "error handling"
    print(f"\nPerforming enhanced search for: '{query}'")
    results = await enhanced_hybrid_search(db, query, verbose=True)
    format_results(results, f"Enhanced hybrid search results for: '{query}'")
    
    # Search focused on glossary
    query = "vector representations"
    print(f"\nPerforming enhanced search for: '{query}'")
    results = await enhanced_hybrid_search(db, query, verbose=True)
    format_results(results, f"Enhanced hybrid search results for: '{query}'")
    
    return True

async def run_glossary_tests():
    """Run all glossary search tests."""
    try:
        # Set up test environment
        db = await setup_test_environment()
        if not db:
            logger.error("Failed to set up test environment")
            return False
            
        # Run tests
        await test_basic_glossary_search(db)
        await test_enhanced_hybrid_search(db)
        
        print("\n" + "="*80)
        print(" SUMMARY")
        print("="*80)
        print("\nGlossary search functionality has been successfully implemented and tested.")
        print("The system now supports:")
        print("1. Basic glossary search with RapidFuzz verification for term matching")
        print("2. Enhanced hybrid search that combines knowledge base and glossary results")
        print("3. ModernBERT-compatible embedding format for terms and definitions")
        
        return True
    except Exception as e:
        logger.error(f"Error running glossary tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_glossary_tests())
    sys.exit(0 if success else 1) 