"""
Example of using Nomic Embed for text embeddings.

This demonstrates the recommended pattern for generating embeddings using
the Nomic Embed v2 model, including proper async handling and metadata.

Documentation References:
- Nomic Embed: https://docs.nomic.ai/reference/nomic-embed-text-v2
- asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
- Related Rule: See `.cursor/rules/011-embedding-practices.mdc` for comprehensive embedding practices
- Package Usage: See `.cursor/rules/003-package-usage.mdc` section on "Embedding"
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger


async def get_embedding(text: str) -> Dict[str, Any]:
    """
    Get embedding for text using Nomic Embed v2.
    
    This function demonstrates the proper pattern for generating embeddings:
    1. Use asyncio.to_thread for the CPU-bound embedding computation
    2. Include standard metadata with the embedding
    3. Handle errors properly
    
    Args:
        text: The text to embed
        
    Returns:
        Dictionary containing the embedding vector and metadata
    """
    try:
        logger.info(f"Generating embedding for text: {text[:50]}...")
        
        # Use to_thread for CPU-bound embedding computation
        start_time = time.time()
        embedding_vector = await asyncio.to_thread(
            compute_embedding_sync, 
            text
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Generated embedding with {len(embedding_vector)} dimensions in {elapsed:.2f} seconds")
        
        # Format with standard metadata
        return {
            "embedding": embedding_vector,
            "metadata": {
                "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
                "embedding_timestamp": datetime.now().isoformat(),
                "embedding_method": "local",
                "embedding_dim": len(embedding_vector),
                "text_length": len(text),
                "processing_time": elapsed
            }
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def compute_embedding_sync(text: str) -> List[float]:
    """
    Synchronous function to compute embedding.
    
    In a real implementation, this would use the actual Nomic Embed model.
    For this example, we're simulating the embedding computation.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    # In a real implementation, you would use:
    # from snippets.embedding import get_embeddings
    # return get_embeddings([text])[0]
    
    # For this example, we'll simulate the embedding computation
    # with a simple hash-based approach
    import hashlib
    import struct
    
    # Generate a deterministic "embedding" based on the text
    # (This is NOT a real embedding, just a simulation for the example)
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    # Convert hash to a list of floats (simulating a 768-dimensional embedding)
    embedding_dim = 768
    embedding = []
    
    for i in range(0, min(len(hash_bytes) - 4, embedding_dim * 4), 4):
        # Convert 4 bytes to a float
        val = struct.unpack('f', hash_bytes[i:i+4])[0]
        # Normalize to range [-1, 1]
        val = max(min(val, 1.0), -1.0)
        embedding.append(val)
    
    # Pad to full dimension if needed
    while len(embedding) < embedding_dim:
        embedding.append(0.0)
    
    # Simulate computation time
    time.sleep(0.1)
    
    return embedding[:embedding_dim]


async def batch_get_embeddings(texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Get embeddings for multiple texts in batches.
    
    This demonstrates how to efficiently process multiple texts by:
    1. Breaking them into optimal batch sizes
    2. Processing batches concurrently
    3. Handling errors for individual items
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embedding dictionaries
    """
    logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Process each text in the batch concurrently
        batch_tasks = [
            asyncio.create_task(get_embedding(text))
            for text in batch
        ]
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any errors in the batch
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error embedding text at index {i+j}: {result}")
                # Add None for failed embeddings
                results.append(None)
            else:
                results.append(result)
    
    # Count successful embeddings
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Successfully generated {successful}/{len(texts)} embeddings")
    
    return results


async def create_document_with_embedding(content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a document with embedding for storage.
    
    This demonstrates the recommended document structure for storing
    embeddings in a database.
    
    Args:
        content: The text content to embed
        metadata: Additional metadata for the document
        
    Returns:
        Document with embedding ready for storage
    """
    # Generate embedding
    embedding_result = await get_embedding(content)
    
    # Create document with standard structure
    document = {
        "content": content,
        "metadata": metadata,
        "embedding": embedding_result["embedding"],
        "embedding_metadata": embedding_result["metadata"],
        "created_at": datetime.now().isoformat()
    }
    
    return document


# Example usage
async def main():
    # Example texts
    texts = [
        "This is an example of using Nomic Embed for text embeddings.",
        "Embeddings are useful for semantic search and similarity comparisons.",
        "The Nomic Embed v2 model produces high-quality embeddings for text.",
        "Using asyncio.to_thread helps prevent blocking the event loop during embedding computation.",
        "Always include metadata with your embeddings for better tracking and debugging."
    ]
    
    # Generate embeddings for all texts
    embeddings = await batch_get_embeddings(texts)
    
    # Print results
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            print(f"Text {i+1}: Failed to generate embedding")
        else:
            print(f"Text {i+1}: Generated {embedding['metadata']['embedding_dim']}-dimensional embedding "
                  f"in {embedding['metadata']['processing_time']:.2f} seconds")
    
    # Example of creating a document with embedding
    document = await create_document_with_embedding(
        content="This is a document that will be stored with its embedding.",
        metadata={"title": "Example Document", "author": "AI Assistant"}
    )
    
    print("\nDocument structure for storage:")
    print(f"- Content: {document['content']}")
    print(f"- Metadata: {document['metadata']}")
    print(f"- Embedding dimensions: {len(document['embedding'])}")
    print(f"- Embedding model: {document['embedding_metadata']['embedding_model']}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 