"""
Example of storing and retrieving embeddings in ArangoDB.

This demonstrates the recommended patterns for:
1. Storing documents with embeddings in ArangoDB
2. Creating appropriate indexes for vector search
3. Performing semantic search using embeddings
4. Handling database operations asynchronously

Documentation References:
- ArangoDB Python Driver: https://docs.python-arango.com/en/main/
- ArangoDB Vector Search: https://www.arangodb.com/docs/stable/arangosearch-vector-search.html
- Related Rule: See `.cursor/rules/011-embedding-practices.mdc` section on "Embedding Storage"
- Async DB Operations: See `.cursor/rules/005-async-patterns.mdc` section on "Database Operations"
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger


# Simulated database connection
class ArangoDBConnection:
    """Simulated ArangoDB connection for the example."""
    
    def __init__(self, url: str, db_name: str):
        self.url = url
        self.db_name = db_name
        self.collections = {}
        logger.info(f"Connected to ArangoDB at {url}, database: {db_name}")
    
    def collection(self, name: str):
        """Get or create a collection."""
        if name not in self.collections:
            self.collections[name] = Collection(name)
            logger.info(f"Created collection: {name}")
        return self.collections[name]


class Collection:
    """Simulated ArangoDB collection for the example."""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = {}
        self.indexes = []
    
    def insert(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a document into the collection."""
        if "_key" not in document:
            raise ValueError("Document must have a _key field")
        
        key = document["_key"]
        self.documents[key] = document
        logger.debug(f"Inserted document with key: {key}")
        return document
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a document by key."""
        document = self.documents.get(key)
        if document is None:
            logger.warning(f"Document with key {key} not found")
        return document
    
    def update(self, key: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Update a document."""
        if key not in self.documents:
            raise ValueError(f"Document with key {key} not found")
        
        # Update only the fields provided
        for field, value in document.items():
            if field != "_key":  # Don't update the key
                self.documents[key][field] = value
        
        logger.debug(f"Updated document with key: {key}")
        return self.documents[key]
    
    def delete(self, key: str) -> bool:
        """Delete a document."""
        if key in self.documents:
            del self.documents[key]
            logger.debug(f"Deleted document with key: {key}")
            return True
        return False
    
    def add_vector_index(self, fields: List[str], name: str = None) -> Dict[str, Any]:
        """Add a vector index to the collection."""
        index = {
            "type": "vector",
            "fields": fields,
            "name": name or f"vector_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        logger.info(f"Created vector index on {fields} in collection {self.name}")
        return index
    
    def aql_query(self, query: str, bind_vars: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Simulate an AQL query."""
        # This is a very simplified simulation of AQL queries
        # In a real implementation, this would execute the query on the database
        
        # For this example, we'll simulate a vector search if the query contains "VECTOR_SEARCH"
        if "VECTOR_SEARCH" in query and bind_vars and "query_vector" in bind_vars:
            return self._simulate_vector_search(bind_vars["query_vector"], bind_vars.get("limit", 10))
        
        # Otherwise, return all documents
        return list(self.documents.values())
    
    def _simulate_vector_search(self, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Simulate a vector search by returning random documents."""
        # In a real implementation, this would perform actual vector similarity search
        # For this example, we'll just return some random documents
        import random
        
        # Get a random sample of documents (up to the limit)
        sample_size = min(limit, len(self.documents))
        if sample_size == 0:
            return []
        
        sample_keys = random.sample(list(self.documents.keys()), sample_size)
        return [self.documents[key] for key in sample_keys]


def get_db(url: str, db_name: str) -> ArangoDBConnection:
    """
    Get a database connection.
    
    In a real implementation, this would connect to an actual ArangoDB instance.
    For this example, we're using a simulated connection.
    
    Args:
        url: The URL of the ArangoDB server
        db_name: The name of the database
        
    Returns:
        ArangoDB connection
    """
    # Simulate connection time
    time.sleep(0.1)
    return ArangoDBConnection(url, db_name)


def generate_document_key(url: str) -> str:
    """
    Generate a document key from a URL.
    
    Args:
        url: The URL to generate a key for
        
    Returns:
        A key suitable for use in ArangoDB
    """
    import hashlib
    # Create a hash of the URL
    hash_obj = hashlib.sha256(url.encode('utf-8'))
    # Use the first 16 characters of the hex digest as the key
    return hash_obj.hexdigest()[:16]


async def store_document_with_embedding(
    document: Dict[str, Any],
    embedding: List[float],
    db_url: str = "http://localhost:8529",
    db_name: str = "mydb",
    collection_name: str = "documents"
) -> str:
    """
    Store a document with embedding in ArangoDB.
    
    This function demonstrates the proper pattern for storing documents with embeddings:
    1. Use asyncio.to_thread for database operations
    2. Include standard embedding metadata
    3. Handle errors properly
    
    Args:
        document: The document to store
        embedding: The embedding vector
        db_url: The URL of the ArangoDB server
        db_name: The name of the database
        collection_name: The name of the collection
        
    Returns:
        The key of the stored document
    """
    try:
        logger.info(f"Storing document with embedding in {collection_name}")
        
        # Generate a key if not provided
        if "_key" not in document:
            if "url" in document:
                document["_key"] = generate_document_key(document["url"])
            else:
                document["_key"] = generate_document_key(str(datetime.now().timestamp()))
        
        # Add embedding and metadata
        document["embedding"] = {
            "vector": embedding,
            "metadata": {
                "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
                "embedding_timestamp": datetime.now().isoformat(),
                "embedding_method": "local",
                "embedding_dim": len(embedding)
            }
        }
        
        # Use to_thread for database operations
        db = await asyncio.to_thread(get_db, db_url, db_name)
        collection = db.collection(collection_name)
        
        # Ensure vector index exists
        await asyncio.to_thread(
            collection.add_vector_index,
            ["embedding.vector"],
            "embedding_vector_idx"
        )
        
        # Store the document
        result = await asyncio.to_thread(collection.insert, document)
        
        logger.info(f"Stored document with key: {result['_key']}")
        return result["_key"]
    
    except Exception as e:
        logger.error(f"Error storing document with embedding: {e}")
        raise


async def retrieve_document(
    key: str,
    db_url: str = "http://localhost:8529",
    db_name: str = "mydb",
    collection_name: str = "documents"
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document from ArangoDB by key.
    
    Args:
        key: The key of the document to retrieve
        db_url: The URL of the ArangoDB server
        db_name: The name of the database
        collection_name: The name of the collection
        
    Returns:
        The document, or None if not found
    """
    try:
        logger.info(f"Retrieving document with key: {key}")
        
        # Use to_thread for database operations
        db = await asyncio.to_thread(get_db, db_url, db_name)
        collection = db.collection(collection_name)
        
        # Retrieve the document
        document = await asyncio.to_thread(collection.get, key)
        
        if document:
            logger.info(f"Retrieved document with key: {key}")
        else:
            logger.warning(f"Document with key {key} not found")
        
        return document
    
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise


async def semantic_search(
    query_embedding: List[float],
    db_url: str = "http://localhost:8529",
    db_name: str = "mydb",
    collection_name: str = "documents",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using an embedding vector.
    
    Args:
        query_embedding: The embedding vector to search with
        db_url: The URL of the ArangoDB server
        db_name: The name of the database
        collection_name: The name of the collection
        limit: Maximum number of results to return
        
    Returns:
        List of documents sorted by similarity to the query embedding
    """
    try:
        logger.info(f"Performing semantic search in {collection_name}")
        
        # Use to_thread for database operations
        db = await asyncio.to_thread(get_db, db_url, db_name)
        collection = db.collection(collection_name)
        
        # AQL query for vector search
        # In a real implementation, this would use VECTOR_SEARCH or DISTANCE functions
        aql_query = """
        FOR doc IN @@collection
            FILTER doc.embedding != null
            VECTOR_SEARCH(doc.embedding.vector, @query_vector)
            LIMIT @limit
            RETURN doc
        """
        
        bind_vars = {
            "@collection": collection_name,
            "query_vector": query_embedding,
            "limit": limit
        }
        
        # Execute the query
        results = await asyncio.to_thread(
            collection.aql_query,
            aql_query,
            bind_vars
        )
        
        logger.info(f"Found {len(results)} results for semantic search")
        return results
    
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise


# Example usage
async def main():
    # Example document
    document = {
        "url": "https://example.com/page1",
        "title": "Example Page",
        "content": "This is an example page about embeddings and vector search.",
        "metadata": {
            "author": "AI Assistant",
            "created_at": datetime.now().isoformat()
        }
    }
    
    # Example embedding (simplified for the example)
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # 50-dimensional vector
    
    try:
        # Store the document with embedding
        key = await store_document_with_embedding(document, embedding)
        print(f"Stored document with key: {key}")
        
        # Retrieve the document
        retrieved = await retrieve_document(key)
        if retrieved:
            print(f"Retrieved document: {retrieved['title']}")
            print(f"Embedding dimensions: {len(retrieved['embedding']['vector'])}")
        
        # Store a few more documents for search example
        for i in range(5):
            doc = {
                "url": f"https://example.com/page{i+2}",
                "title": f"Example Page {i+2}",
                "content": f"This is example page {i+2} about {'embeddings' if i % 2 == 0 else 'vectors'}.",
                "metadata": {
                    "author": "AI Assistant",
                    "created_at": datetime.now().isoformat()
                }
            }
            # Slightly different embedding for each document
            emb = [0.1 * (i+1), 0.2 * (i+1), 0.3 * (i+1), 0.4 * (i+1), 0.5 * (i+1)] * 10
            await store_document_with_embedding(doc, emb)
        
        # Perform semantic search
        results = await semantic_search(embedding)
        print(f"\nSemantic search results ({len(results)} found):")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} - {result['url']}")
    
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 