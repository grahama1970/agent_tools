#!/usr/bin/env python3
"""
Test script for glossary term search.

This script demonstrates searching for terms in the glossary,
including both terms that exist and terms that don't exist.
"""

import asyncio
import logging
from arango import ArangoClient

from agent_tools.cursor_rules.core.glossary import (
    setup_glossary,
    SAMPLE_GLOSSARY,
    glossary_search,
    enhanced_hybrid_search
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_test_db():
    """Setup test database with sample glossary entries."""
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    
    # Connect to system database with correct credentials
    sys_db = client.db("_system", username="root", password="openSesame")
    
    # Create test database if it doesn't exist
    test_db_name = "test_glossary_db"
    if not sys_db.has_database(test_db_name):
        sys_db.create_database(test_db_name)
        logger.info(f"Created test database: {test_db_name}")
    
    # Connect to test database
    db = client.db(test_db_name, username="root", password="openSesame")
    
    # Setup glossary collection with sample entries
    await setup_glossary(db)
    
    return db

def format_results(results, search_type="glossary"):
    """Format and display search results."""
    if not results:
        print(f"\nNo {search_type} results found.")
        return
    
    print(f"\n{len(results)} {search_type.upper()} RESULTS:")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        # Different handling based on result type
        if search_type == "glossary" and isinstance(result, tuple):
            # Glossary search returns tuple of (term_doc, hybrid_score, bm25_score, similarity_score)
            term_doc, hybrid_score, bm25_score, similarity_score = result
            print(f"Result {i}:")
            print(f"Term: {term_doc.get('term', 'N/A')}")
            print(f"Definition: {term_doc.get('definition', 'N/A')}")
            print(f"Category: {term_doc.get('category', 'N/A')}")
            print(f"Hybrid Score: {hybrid_score:.4f}")
            print(f"BM25 Score: {bm25_score:.4f}")
            print(f"Vector Similarity: {similarity_score:.4f}")
        elif search_type == "enhanced" and isinstance(result, dict):
            # Enhanced search returns dicts
            content = result.get("content", {})
            print(f"Result {i} ({result.get('type', 'unknown')}):")
            
            if result.get("type") == "glossary":
                print(f"Term: {content.get('term', 'N/A')}")
                print(f"Definition: {content.get('definition', 'N/A')}")
            else:
                print(f"Title: {content.get('title', 'N/A')}")
                print(f"Content: {content.get('content', 'N/A')[:100]}...")
                
            print(f"Score: {result.get('score', 0):.4f}")
            
            if "bm25_score" in result:
                print(f"BM25 Score: {result.get('bm25_score', 0):.4f}")
            if "similarity_score" in result:
                print(f"Vector Similarity: {result.get('similarity_score', 0):.4f}")
        else:
            # Generic handling for other result formats
            print(f"Result {i}: {result}")
            
        print("-"*40)

async def test_glossary_search(db):
    """Test glossary search with existing and non-existing terms."""
    print("\n\nTEST 1: SEARCHING FOR EXISTING TERMS")
    print("="*80)
    
    # 1. Search for a term that exists in the sample glossary
    existing_term = "Hybrid Search"
    logger.info(f"Searching for existing term: {existing_term}")
    results = await glossary_search(db, existing_term, verbose=True)
    format_results(results, "glossary")
    
    # 2. Search for a term that doesn't exist in the glossary
    non_existing_term = "Quantum Computing"
    print("\n\nTEST 2: SEARCHING FOR NON-EXISTING TERM")
    print("="*80)
    logger.info(f"Searching for non-existing term: {non_existing_term}")
    results = await glossary_search(db, non_existing_term, verbose=True)
    format_results(results, "glossary")
    
    # 3. Search for a term that's similar but not exact
    similar_term = "Embedded Vector"  # Similar to "Embedding"
    print("\n\nTEST 3: SEARCHING FOR SIMILAR TERM")
    print("="*80)
    logger.info(f"Searching for similar term: {similar_term}")
    results = await glossary_search(db, similar_term, min_similarity=0.7, verbose=True)
    format_results(results, "glossary")
    
    # 4. Enhanced search with existing term
    print("\n\nTEST 4: ENHANCED SEARCH WITH EXISTING TERM")
    print("="*80)
    logger.info(f"Enhanced search for: {existing_term}")
    enhanced_results = await enhanced_hybrid_search(db, existing_term, verbose=True)
    format_results(enhanced_results, "enhanced")
    
    # 5. Enhanced search with non-existing term
    print("\n\nTEST 5: ENHANCED SEARCH WITH NON-EXISTING TERM")
    print("="*80)
    logger.info(f"Enhanced search for: {non_existing_term}")
    enhanced_results = await enhanced_hybrid_search(db, non_existing_term, verbose=True)
    format_results(enhanced_results, "enhanced")

async def run_tests():
    """Run all tests."""
    try:
        logger.info("Setting up test database...")
        db = await setup_test_db()
        
        logger.info("Running tests...")
        await test_glossary_search(db)
        
        logger.info("All tests completed.")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    asyncio.run(run_tests()) 