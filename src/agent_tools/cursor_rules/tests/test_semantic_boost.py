#!/usr/bin/env python3
"""
Test Semantic Search with Exact Boost

This test demonstrates the improved semantic search functionality that balances
semantic matching with term exactness by boosting terms that appear directly in the query.
"""

import asyncio
import logging
from loguru import logger
from arango import ArangoClient
from agent_tools.cursor_rules.core.unified_glossary_search import unified_glossary_search

# Configure logger to show debug messages
logger.remove()
logger.add(lambda msg: print(msg, flush=True), level="DEBUG")

async def test_semantic_with_exact_boost():
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    
    # Inspect database for debugging
    try:
        collections = db.collections()
        logger.debug(f"Available collections: {[c['name'] for c in collections]}")
        
        # Count documents in the glossary collection
        collection = db.collection(collection_name)
        count = collection.count()
        logger.debug(f"Document count in {collection_name}: {count}")
        
        # Show a sample document to check embeddings
        if count > 0:
            cursor = collection.all()
            sample = next(cursor)
            logger.debug(f"Sample document structure: {list(sample.keys())}")
            logger.debug(f"Sample has embedding: {bool('embedding' in sample)}")
            if 'embedding' in sample:
                logger.debug(f"Embedding vector length: {len(sample['embedding'])}")
    except Exception as e:
        logger.error(f"Error inspecting database: {e}")

    # List of test queries
    test_queries = [
        'How do neural networks work?',
        'What is the difference between a transformer and self-attention?',
        'Explain machine learning concepts related to natural language processing',
        'What is an embedding in the context of vector databases?'
    ]

    print('=== TESTING SEMANTIC SEARCH WITH EXACT BOOST ===')
    for query in test_queries:
        print(f'\nQuery: {query}')
        
        # Run search with standard settings
        results = await unified_glossary_search(
            db, 
            collection_name, 
            query,
            fuzzy_threshold=70,
            semantic_threshold=0.3,  # Lower threshold for testing
            exact_boost=0.2,
            limit=5
        )
        
        # Show the results
        print('Results:')
        for item in results['combined_results']:
            match_type = item['match_type']
            boost = item.get('boost', 0)
            boost_info = f' (boosted: {boost:.2f})' if boost > 0 else ''
            print(f'- {item["term"]} [{match_type}{boost_info}]')

        # Compare with no boosting
        print('\nWithout boost:')
        no_boost_results = await unified_glossary_search(
            db, 
            collection_name, 
            query,
            fuzzy_threshold=70,
            semantic_threshold=0.3,  # Lower threshold for testing
            exact_boost=0.0,  # No boost
            limit=5
        )
        
        # Show the no-boost results
        for item in no_boost_results['combined_results']:
            match_type = item['match_type']
            print(f'- {item["term"]} [{match_type}]')

if __name__ == '__main__':
    asyncio.run(test_semantic_with_exact_boost()) 