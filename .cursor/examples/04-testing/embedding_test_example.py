"""
Example of testing embedding functionality.

This demonstrates the recommended patterns for:
1. Testing embedding generation with minimal mocking
2. Using real database connections for integration tests
3. Properly handling test fixtures and cleanup
4. Testing both success and error cases

Documentation References:
- pytest Documentation: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
- Related Rule: See `.cursor/rules/004-testing-practices.mdc` section on "Embedding Testing"
- Related Rule: See `.cursor/rules/011-embedding-practices.mdc` section on "Testing Practices"
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock


# Sample fixture for embedding vector
@pytest.fixture
def sample_embedding() -> Dict[str, Any]:
    """
    Fixture providing a sample embedding for testing.
    
    This follows the standard format defined in the embedding practices.
    """
    return {
        'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 10,  # 50-dimensional vector
        'metadata': {
            'embedding_model': 'nomic-ai/nomic-embed-text-v2-moe',
            'embedding_timestamp': '2023-01-01T00:00:00Z',
            'embedding_method': 'local',
            'embedding_dim': 50
        }
    }


# Sample fixture for a document
@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Fixture providing a sample document for testing."""
    return {
        "_key": "test_doc_1",
        "url": "https://example.com/test",
        "title": "Test Document",
        "content": "This is a test document for embedding testing.",
        "created_at": datetime.now().isoformat()
    }


# Simulated database for testing
class TestDatabase:
    """Simple in-memory database for testing."""
    
    def __init__(self):
        self.collections = {}
    
    def collection(self, name: str):
        """Get or create a collection."""
        if name not in self.collections:
            self.collections[name] = TestCollection(name)
        return self.collections[name]
    
    def clear(self):
        """Clear all collections."""
        self.collections = {}


class TestCollection:
    """Simple in-memory collection for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = {}
    
    def insert(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a document."""
        if "_key" not in document:
            raise ValueError("Document must have a _key field")
        
        key = document["_key"]
        self.documents[key] = document
        return document
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a document by key."""
        return self.documents.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a document."""
        if key in self.documents:
            del self.documents[key]
            return True
        return False
    
    def clear(self):
        """Clear all documents."""
        self.documents = {}


# Fixture for the test database
@pytest.fixture
def test_db():
    """Fixture providing a test database."""
    db = TestDatabase()
    yield db
    db.clear()


# Functions to test

async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding for text.
    
    In a real implementation, this would use an actual embedding model.
    For this example, we're simulating the embedding generation.
    
    Args:
        text: The text to embed
        
    Returns:
        Embedding vector
    """
    # In a real implementation, this would use:
    # from snippets.embedding import get_embeddings
    # return await get_embeddings([text])[0]
    
    # For this example, we'll simulate the embedding generation
    import hashlib
    import struct
    
    # Generate a deterministic "embedding" based on the text
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    # Convert hash to a list of floats
    embedding_dim = 50
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
    await asyncio.sleep(0.1)
    
    return embedding[:embedding_dim]


async def create_embedding_metadata(embedding: List[float]) -> Dict[str, Any]:
    """
    Create metadata for an embedding.
    
    Args:
        embedding: The embedding vector
        
    Returns:
        Embedding with metadata
    """
    return {
        'embedding': embedding,
        'metadata': {
            'embedding_model': 'nomic-ai/nomic-embed-text-v2-moe',
            'embedding_timestamp': datetime.now().isoformat(),
            'embedding_method': 'local',
            'embedding_dim': len(embedding)
        }
    }


async def store_document_with_embedding(
    document: Dict[str, Any],
    db,
    collection_name: str = "documents"
) -> str:
    """
    Store a document with embedding in the database.
    
    Args:
        document: The document to store
        db: The database connection
        collection_name: The name of the collection
        
    Returns:
        The key of the stored document
    """
    # Generate embedding if not already present
    if "embedding" not in document:
        # Extract text for embedding
        text = document.get("content", "")
        if not text:
            raise ValueError("Document must have content for embedding")
        
        # Generate embedding
        embedding = await generate_embedding(text)
        embedding_with_metadata = await create_embedding_metadata(embedding)
        document["embedding"] = embedding_with_metadata
    
    # Store the document
    collection = db.collection(collection_name)
    result = collection.insert(document)
    
    return result["_key"]


async def retrieve_document(
    key: str,
    db,
    collection_name: str = "documents"
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document from the database.
    
    Args:
        key: The key of the document
        db: The database connection
        collection_name: The name of the collection
        
    Returns:
        The document, or None if not found
    """
    collection = db.collection(collection_name)
    return collection.get(key)


# Tests

@pytest.mark.asyncio
async def test_generate_embedding():
    """Test generating an embedding for text."""
    # Test with a simple text
    text = "This is a test document"
    embedding = await generate_embedding(text)
    
    # Verify the embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 50  # Expected dimension
    assert all(isinstance(val, float) for val in embedding)
    assert all(-1.0 <= val <= 1.0 for val in embedding)  # Values should be normalized


@pytest.mark.asyncio
async def test_create_embedding_metadata():
    """Test creating metadata for an embedding."""
    # Create a sample embedding
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # 50-dimensional vector
    
    # Create metadata
    result = await create_embedding_metadata(embedding)
    
    # Verify the result
    assert "embedding" in result
    assert "metadata" in result
    assert result["embedding"] == embedding
    assert "embedding_model" in result["metadata"]
    assert "embedding_timestamp" in result["metadata"]
    assert "embedding_method" in result["metadata"]
    assert "embedding_dim" in result["metadata"]
    assert result["metadata"]["embedding_dim"] == 50


@pytest.mark.asyncio
async def test_store_and_retrieve_document(test_db, sample_document):
    """Test storing and retrieving a document with embedding."""
    # Store the document
    key = await store_document_with_embedding(sample_document, test_db)
    
    # Verify the key
    assert key == sample_document["_key"]
    
    # Retrieve the document
    retrieved = await retrieve_document(key, test_db)
    
    # Verify the retrieved document
    assert retrieved is not None
    assert retrieved["_key"] == key
    assert retrieved["title"] == sample_document["title"]
    assert retrieved["content"] == sample_document["content"]
    assert "embedding" in retrieved
    assert "embedding" in retrieved["embedding"]
    assert "metadata" in retrieved["embedding"]
    assert len(retrieved["embedding"]["embedding"]) == 50


@pytest.mark.asyncio
async def test_store_document_without_content(test_db):
    """Test storing a document without content (should fail)."""
    # Create a document without content
    document = {
        "_key": "test_doc_no_content",
        "title": "Test Document Without Content"
    }
    
    # Attempt to store the document (should raise ValueError)
    with pytest.raises(ValueError) as excinfo:
        await store_document_with_embedding(document, test_db)
    
    # Verify the error message
    assert "Document must have content for embedding" in str(excinfo.value)


@pytest.mark.asyncio
async def test_store_document_with_mock_embedding(test_db, sample_document, sample_embedding):
    """Test storing a document with a mocked embedding."""
    # Mock the generate_embedding function
    with patch("__main__.generate_embedding") as mock_generate:
        # Set up the mock to return a fixed embedding
        mock_generate.return_value = sample_embedding["embedding"]
        
        # Store the document
        key = await store_document_with_embedding(sample_document, test_db)
        
        # Verify the key
        assert key == sample_document["_key"]
        
        # Verify that generate_embedding was called with the document content
        mock_generate.assert_called_once_with(sample_document["content"])
        
        # Retrieve the document
        retrieved = await retrieve_document(key, test_db)
        
        # Verify the embedding
        assert retrieved is not None
        assert "embedding" in retrieved
        assert retrieved["embedding"]["embedding"] == sample_embedding["embedding"]


@pytest.mark.asyncio
async def test_integration_with_real_db():
    """
    Integration test with a real database connection.
    
    This test is marked with a custom marker to indicate it requires
    a real database connection. It would be skipped in CI environments
    where a database is not available.
    """
    pytest.skip("This test requires a real database connection")
    
    # In a real test, this would use:
    # from fetch_page.db.arangodb_utils import get_db
    # db = get_db("http://localhost:8529", "test_db")
    
    # For this example, we'll use our test database
    db = TestDatabase()
    
    try:
        # Create a test document
        document = {
            "_key": "integration_test_doc",
            "url": "https://example.com/integration_test",
            "title": "Integration Test Document",
            "content": "This is a document for integration testing with a real database."
        }
        
        # Store the document
        key = await store_document_with_embedding(document, db)
        
        # Retrieve the document
        retrieved = await retrieve_document(key, db)
        
        # Verify the document
        assert retrieved is not None
        assert retrieved["_key"] == key
        assert "embedding" in retrieved
    
    finally:
        # Clean up
        db.clear()


# Example of running the tests
if __name__ == "__main__":
    # This would normally be run using pytest
    print("These tests should be run using pytest:")
    print("python -m pytest embedding_test_example.py -v")
    
    # For demonstration, we'll run one test directly
    async def run_test():
        db = TestDatabase()
        document = {
            "_key": "demo_test_doc",
            "title": "Demo Test Document",
            "content": "This is a document for the demo test."
        }
        
        try:
            key = await store_document_with_embedding(document, db)
            print(f"Stored document with key: {key}")
            
            retrieved = await retrieve_document(key, db)
            print(f"Retrieved document: {retrieved['title']}")
            print(f"Embedding dimensions: {len(retrieved['embedding']['embedding'])}")
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            db.clear()
    
    asyncio.run(run_test()) 