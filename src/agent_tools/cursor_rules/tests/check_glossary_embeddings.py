#!/usr/bin/env python3
"""
Check Glossary Embeddings

This script verifies that glossary entries have proper embeddings and
reports any issues with the embedding format.
"""

import asyncio
from arango import ArangoClient
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the embedding utilities
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
from agent_tools.cursor_rules.core.glossary import format_for_embedding

async def check_glossary_embeddings():
    """Check and report on glossary entries and their embeddings."""
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    
    # Query for all glossary entries
    query = f'''
    FOR doc IN {collection_name}
        RETURN {{
            term: doc.term,
            definition: doc.definition,
            category: doc.category,
            has_embedding: HAS(doc, "embedding"),
            embedding_length: HAS(doc, "embedding") ? LENGTH(doc.embedding) : 0,
            _key: doc._key
        }}
    '''
    
    cursor = db.aql.execute(query)
    results = list(cursor)
    
    # Display results in a table
    if results:
        table = []
        for result in results:
            term = result.get('term', 'N/A')
            definition = result.get('definition', 'N/A')
            
            # Truncate definition for display
            short_definition = definition[:50] + '...' if len(definition) > 50 else definition
            
            # Check embedding status
            has_embedding = result.get('has_embedding', False)
            embedding_length = result.get('embedding_length', 0)
            
            # Format for embedding to display how it should be formatted
            formatted = format_for_embedding(term, definition)
            
            table.append([
                term,
                short_definition,
                result.get('category', 'N/A'),
                '✓' if has_embedding else '✗',
                embedding_length,
                formatted[:50] + '...' if len(formatted) > 50 else formatted
            ])
        
        print('\n=== GLOSSARY ENTRIES AND EMBEDDINGS ===')
        print(tabulate(
            table, 
            headers=['Term', 'Definition', 'Category', 'Has Embedding', 'Embedding Length', 'Formatted For Embedding'], 
            tablefmt='grid'
        ))
        print(f'Total entries: {len(results)}')
        
        # Check for entries without embeddings
        missing_embeddings = [r for r in results if not r.get('has_embedding', False)]
        if missing_embeddings:
            print(f'\n⚠️ WARNING: {len(missing_embeddings)} entries missing embeddings:')
            for entry in missing_embeddings:
                print(f"  - {entry.get('term', 'N/A')}")
    else:
        print('No glossary entries found')
        
    # Report on embedding conformance with embedding_utils.py requirements
    print('\n=== EMBEDDING FORMAT VERIFICATION ===')
    # Check if embedding is using the correct format according to embedding_utils.py
    print("The embedding_utils.py expects text to have a prefix like 'search_document:' or 'search_query:'")
    print("Current format_for_embedding returns: [TERM: term] definition")
    print("This format may need to be updated to be consistent with embedding_utils.py requirements.")

async def add_glossary_entries():
    """Add new glossary entries with properly formatted embeddings."""
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    
    # New glossary entries for better test coverage
    new_entries = [
        {
            "term": "Vector Database",
            "definition": "A database designed to store and query high-dimensional vectors, often used for semantic search applications",
            "category": "database",
            "related_terms": ["Semantic Search", "Embedding", "ArangoDB"]
        },
        {
            "term": "Transformer",
            "definition": "A deep learning architecture that uses self-attention mechanisms to process sequential data like text",
            "category": "nlp",
            "related_terms": ["Neural Network", "Embedding", "Self-Attention"]
        },
        {
            "term": "Cosine Similarity",
            "definition": "A measure of similarity between two non-zero vectors by calculating the cosine of the angle between them",
            "category": "math",
            "related_terms": ["Vector", "Embedding", "Semantic Search"]
        },
        {
            "term": "Fuzzy Matching",
            "definition": "An algorithm that finds strings that approximately match a pattern, allowing for variations and misspellings",
            "category": "search",
            "related_terms": ["String Matching", "Levenshtein Distance", "Search"]
        },
        {
            "term": "Self-Attention",
            "definition": "A mechanism in neural networks where different positions of a single sequence attend to each other to compute representations",
            "category": "nlp",
            "related_terms": ["Transformer", "Neural Network", "Attention Mechanism"]
        }
    ]
    
    # Get the collection
    collection = db.collection(collection_name)
    
    # Add entries with properly formatted embeddings
    added_count = 0
    for entry in new_entries:
        # Check if entry already exists
        query = f'''
        FOR doc IN {collection_name}
            FILTER doc.term == @term
            RETURN doc
        '''
        cursor = db.aql.execute(query, bind_vars={"term": entry["term"]})
        existing = list(cursor)
        
        if existing:
            print(f"Entry '{entry['term']}' already exists, skipping")
            continue
        
        # Format for embedding and generate embedding
        term = entry["term"]
        definition = entry["definition"]
        
        # Format according to the current standard
        text_to_embed = format_for_embedding(term, definition)
        print(f"Generating embedding for: {text_to_embed}")
        
        # Generate embedding
        embedding_result = generate_embedding(text_to_embed)
        
        if embedding_result and "embedding" in embedding_result:
            # Add embedding to entry
            entry["embedding"] = embedding_result["embedding"]
            entry["embedding_metadata"] = embedding_result.get("metadata", {})
            
            # Insert into collection
            try:
                collection.insert(entry)
                added_count += 1
                print(f"Added entry: {term}")
            except Exception as e:
                print(f"Error adding entry '{term}': {e}")
        else:
            print(f"Failed to generate embedding for '{term}'")
    
    print(f"\nAdded {added_count} new glossary entries")

if __name__ == "__main__":
    # First check current entries
    asyncio.run(check_glossary_embeddings())
    
    # Ask if user wants to add new entries
    response = input("\nDo you want to add new glossary entries for better test coverage? (y/n): ")
    if response.lower() in ['y', 'yes']:
        asyncio.run(add_glossary_entries())
        # Check again after adding
        asyncio.run(check_glossary_embeddings()) 