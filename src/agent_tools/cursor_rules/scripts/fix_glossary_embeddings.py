#!/usr/bin/env python3
"""
Fix Glossary Embeddings Script

This script updates existing glossary entries to use the correct prefix format required by
Nomic ModernBert embedding models, which is "search_document: <text>" instead of the current
"[TERM: term] definition" format.
"""

import asyncio
from arango import ArangoClient
from loguru import logger

from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

# Configuration
ARANGO_HOST = "http://localhost:8529"
DB_NAME = "test_semantic_term_definition_db"
COLLECTION_NAME = "test_semantic_glossary"
USERNAME = "root"
PASSWORD = "openSesame"

def format_for_embedding_nomic(term: str, definition: str) -> str:
    """
    Format a term and definition for embedding using the correct Nomic format.
    
    Args:
        term: The terminology or concept
        definition: The definition or explanation
        
    Returns:
        Formatted text ready for embedding with the correct Nomic prefix
    """
    # Using the required 'search_document:' prefix for Nomic embeddings
    return f"search_document: {term} - {definition}"

async def main():
    """Main function to update existing glossary entries with correct embeddings."""
    # Connect to ArangoDB
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    # Get the collection
    if not db.has_collection(COLLECTION_NAME):
        logger.error(f"Collection {COLLECTION_NAME} does not exist")
        return
    
    collection = db.collection(COLLECTION_NAME)
    logger.info(f"Connected to collection: {COLLECTION_NAME}")
    
    # Get all entries
    cursor = collection.all()
    entries = list(cursor)
    total_entries = len(entries)
    logger.info(f"Found {total_entries} entries to process")
    
    # Process each entry and update its embedding
    updated_count = 0
    failed_count = 0
    
    for entry in entries:
        doc_key = entry["_key"]
        term = entry.get("term", "")
        definition = entry.get("definition", "")
        
        if not term or not definition:
            logger.warning(f"Entry {doc_key} is missing term or definition, skipping")
            failed_count += 1
            continue
        
        # Format with correct Nomic prefix
        text_to_embed = format_for_embedding_nomic(term, definition)
        logger.debug(f"Reformatted text for embedding: {text_to_embed}")
        
        # Generate new embedding
        embedding_result = generate_embedding(text_to_embed)
        
        if not embedding_result or "embedding" not in embedding_result:
            logger.warning(f"Failed to generate embedding for term: {term}")
            failed_count += 1
            continue
        
        # Update the entry with new embedding
        try:
            collection.update({
                "_key": doc_key,
                "embedding": embedding_result["embedding"],
                "embedding_format": "nomic_search_document_prefix"  # Mark the format used
            })
            logger.info(f"Updated embedding for term: {term}")
            updated_count += 1
        except Exception as e:
            logger.error(f"Error updating entry {doc_key}: {e}")
            failed_count += 1
    
    logger.info(f"Embedding update complete: Updated {updated_count} entries, failed {failed_count} entries")

if __name__ == "__main__":
    asyncio.run(main()) 