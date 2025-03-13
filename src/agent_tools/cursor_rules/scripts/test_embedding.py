#!/usr/bin/env python
"""
Test script for embedding functionality in the cursor_rules package.

This script verifies that the embedding functionality is properly integrated
and working correctly.
"""

import asyncio
import json
import sys
from loguru import logger

from agent_tools.cursor_rules.core.cursor_rules import generate_embedding, EMBEDDING_AVAILABLE
from agent_tools.cursor_rules.embedding import (
    create_embedding,
    create_embedding_sync,
    create_embedding_with_sentence_transformer,
    ensure_text_has_prefix
)


async def main():
    """Test embedding functionality in the cursor_rules package."""
    logger.info("Testing embedding functionality in cursor_rules package")
    
    # Check if embeddings are available
    logger.info(f"Embeddings available: {EMBEDDING_AVAILABLE}")
    
    if not EMBEDDING_AVAILABLE:
        logger.error("Embedding utilities are not available!")
        logger.error("Check your Python environment and dependencies.")
        return 1
    
    # Test text
    text = "This is a test text for generating embeddings."
    logger.info(f"Test text: {text}")
    
    # Test ensure_text_has_prefix
    prefixed_text = ensure_text_has_prefix(text)
    logger.info(f"Prefixed text: {prefixed_text}")
    
    # Test create_embedding_sync
    logger.info("Testing create_embedding_sync...")
    try:
        sync_result = create_embedding_sync(text)
        logger.info(f"Sync embedding dimension: {len(sync_result['embedding'])}")
        logger.info(f"Sync embedding metadata: {json.dumps(sync_result['metadata'], indent=2)}")
    except Exception as e:
        logger.error(f"Error in create_embedding_sync: {e}")
    
    # Test generate_embedding
    logger.info("Testing generate_embedding...")
    try:
        result = generate_embedding(text)
        logger.info(f"Generated embedding dimension: {len(result['embedding'])}")
        logger.info(f"Generated embedding metadata: {json.dumps(result['metadata'], indent=2)}")
    except Exception as e:
        logger.error(f"Error in generate_embedding: {e}")
    
    # Test async create_embedding
    logger.info("Testing async create_embedding...")
    try:
        async_result = await create_embedding(text)
        logger.info(f"Async embedding dimension: {len(async_result['embedding'])}")
        logger.info(f"Async embedding metadata: {json.dumps(async_result['metadata'], indent=2)}")
    except Exception as e:
        logger.error(f"Error in async create_embedding: {e}")
    
    # Try to test create_embedding_with_sentence_transformer if available
    logger.info("Testing create_embedding_with_sentence_transformer...")
    try:
        st_result = create_embedding_with_sentence_transformer(text)
        logger.info(f"SentenceTransformer embedding dimension: {len(st_result['embedding'])}")
        logger.info(f"SentenceTransformer metadata: {json.dumps(st_result['metadata'], indent=2)}")
    except Exception as e:
        logger.error(f"Error in create_embedding_with_sentence_transformer: {e}")
        logger.info("SentenceTransformer may not be installed or properly configured.")
    
    logger.info("Embedding tests completed")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 