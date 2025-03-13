"""
Example of using ArangoDB with Python.

This demonstrates the recommended patterns for:
1. Connecting to ArangoDB
2. Creating collections and indexes
3. Performing CRUD operations
4. Executing AQL queries
5. Handling database operations asynchronously

Documentation References:
- ArangoDB Python Driver: https://docs.python-arango.com/en/main/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
- Related Rule: See `.cursor/rules/005-async-patterns.mdc` section on "Database Operations"
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from loguru import logger


class ArangoDBClient:
    """
    Client for interacting with ArangoDB.
    
    This class demonstrates the recommended patterns for:
    1. Establishing database connections
    2. Managing collections
    3. Performing CRUD operations
    4. Executing AQL queries
    """
    
    def __init__(self, url: str, db_name: str, username: str = "root", password: str = ""):
        """
        Initialize the ArangoDB client.
        
        Args:
            url: The URL of the ArangoDB server
            db_name: The name of the database
            username: The username for authentication
            password: The password for authentication
        """
        self.url = url
        self.db_name = db_name
        self.username = username
        self.password = password
        self._db = None
        logger.info(f"Initialized ArangoDB client for {url}, database: {db_name}")
    
    def connect(self) -> Any:
        """
        Connect to the ArangoDB database.
        
        In a real implementation, this would use the python-arango library.
        For this example, we're simulating the connection.
        
        Returns:
            Database connection
        """
        # In a real implementation, this would use:
        # from arango import ArangoClient
        # client = ArangoClient(hosts=self.url)
        # db = client.db(self.db_name, username=self.username, password=self.password)
        
        # For this example, we'll simulate the connection
        logger.info(f"Connecting to ArangoDB at {self.url}, database: {self.db_name}")
        time.sleep(0.1)  # Simulate connection time
        
        # Simulate the database connection
        self._db = SimulatedArangoDB(self.url, self.db_name)
        
        return self._db
    
    def get_db(self) -> Any:
        """
        Get the database connection.
        
        Returns:
            Database connection
        """
        if self._db is None:
            self._db = self.connect()
        return self._db
    
    def collection(self, name: str, create: bool = True) -> Any:
        """
        Get a collection.
        
        Args:
            name: The name of the collection
            create: Whether to create the collection if it doesn't exist
            
        Returns:
            Collection object
        """
        db = self.get_db()
        
        # Check if the collection exists
        if name not in db.collections:
            if create:
                logger.info(f"Creating collection: {name}")
                db.create_collection(name)
            else:
                raise ValueError(f"Collection {name} does not exist")
        
        return db.collection(name)
    
    def create_document(self, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a document in a collection.
        
        Args:
            collection_name: The name of the collection
            document: The document to create
            
        Returns:
            The created document
        """
        collection = self.collection(collection_name)
        
        # Insert the document
        result = collection.insert(document)
        logger.debug(f"Created document in {collection_name} with key: {result['_key']}")
        
        return result
    
    def get_document(self, collection_name: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by key.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            
        Returns:
            The document, or None if not found
        """
        collection = self.collection(collection_name, create=False)
        
        # Get the document
        document = collection.get(key)
        
        if document is None:
            logger.warning(f"Document with key {key} not found in {collection_name}")
        else:
            logger.debug(f"Retrieved document from {collection_name} with key: {key}")
        
        return document
    
    def update_document(
        self,
        collection_name: str,
        key: str,
        document: Dict[str, Any],
        keep_none: bool = False
    ) -> Dict[str, Any]:
        """
        Update a document.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            document: The document fields to update
            keep_none: Whether to keep None values
            
        Returns:
            The updated document
        """
        collection = self.collection(collection_name, create=False)
        
        # Update the document
        result = collection.update(key, document, keep_none=keep_none)
        logger.debug(f"Updated document in {collection_name} with key: {key}")
        
        return result
    
    def delete_document(self, collection_name: str, key: str) -> bool:
        """
        Delete a document.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            
        Returns:
            True if the document was deleted, False otherwise
        """
        collection = self.collection(collection_name, create=False)
        
        # Delete the document
        result = collection.delete(key)
        
        if result:
            logger.debug(f"Deleted document from {collection_name} with key: {key}")
        else:
            logger.warning(f"Document with key {key} not found in {collection_name} for deletion")
        
        return result
    
    def execute_query(
        self,
        query: str,
        bind_vars: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Execute an AQL query.
        
        Args:
            query: The AQL query
            bind_vars: The bind variables for the query
            batch_size: The batch size for the query
            
        Returns:
            The query results
        """
        db = self.get_db()
        
        # Execute the query
        logger.debug(f"Executing AQL query: {query}")
        cursor = db.aql.execute(query, bind_vars=bind_vars or {}, batch_size=batch_size)
        
        # Collect the results
        results = list(cursor)
        logger.debug(f"Query returned {len(results)} results")
        
        return results
    
    def create_index(
        self,
        collection_name: str,
        index_type: str,
        fields: List[str],
        name: Optional[str] = None,
        unique: bool = False,
        sparse: bool = False
    ) -> Dict[str, Any]:
        """
        Create an index on a collection.
        
        Args:
            collection_name: The name of the collection
            index_type: The type of index (e.g., "hash", "skiplist", "fulltext", "geo", "persistent")
            fields: The fields to index
            name: The name of the index
            unique: Whether the index should enforce uniqueness
            sparse: Whether the index should be sparse
            
        Returns:
            The created index
        """
        collection = self.collection(collection_name)
        
        # Create the index
        if index_type == "hash":
            result = collection.add_hash_index(fields, unique=unique, sparse=sparse, name=name)
        elif index_type == "skiplist":
            result = collection.add_skiplist_index(fields, unique=unique, sparse=sparse, name=name)
        elif index_type == "fulltext":
            result = collection.add_fulltext_index(fields, name=name)
        elif index_type == "geo":
            result = collection.add_geo_index(fields, name=name)
        elif index_type == "persistent":
            result = collection.add_persistent_index(fields, unique=unique, sparse=sparse, name=name)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created {index_type} index on {fields} in collection {collection_name}")
        
        return result


# Simulated ArangoDB for the example
class SimulatedArangoDB:
    """Simulated ArangoDB for the example."""
    
    def __init__(self, url: str, db_name: str):
        self.url = url
        self.db_name = db_name
        self.collections = {}
        self.aql = AQLExecutor(self)
    
    def create_collection(self, name: str) -> Any:
        """Create a collection."""
        if name not in self.collections:
            self.collections[name] = SimulatedCollection(name)
        return self.collections[name]
    
    def collection(self, name: str) -> Any:
        """Get a collection."""
        return self.collections[name]


class SimulatedCollection:
    """Simulated ArangoDB collection for the example."""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = {}
        self.indexes = []
    
    def insert(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a document."""
        if "_key" not in document:
            document["_key"] = f"doc_{len(self.documents) + 1}"
        
        key = document["_key"]
        self.documents[key] = document
        return document
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a document by key."""
        return self.documents.get(key)
    
    def update(self, key: str, document: Dict[str, Any], keep_none: bool = False) -> Dict[str, Any]:
        """Update a document."""
        if key not in self.documents:
            raise ValueError(f"Document with key {key} not found")
        
        # Update only the fields provided
        for field, value in document.items():
            if value is not None or keep_none:
                self.documents[key][field] = value
        
        return self.documents[key]
    
    def delete(self, key: str) -> bool:
        """Delete a document."""
        if key in self.documents:
            del self.documents[key]
            return True
        return False
    
    def add_hash_index(
        self,
        fields: List[str],
        unique: bool = False,
        sparse: bool = False,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a hash index."""
        index = {
            "type": "hash",
            "fields": fields,
            "unique": unique,
            "sparse": sparse,
            "name": name or f"hash_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        return index
    
    def add_skiplist_index(
        self,
        fields: List[str],
        unique: bool = False,
        sparse: bool = False,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a skiplist index."""
        index = {
            "type": "skiplist",
            "fields": fields,
            "unique": unique,
            "sparse": sparse,
            "name": name or f"skiplist_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        return index
    
    def add_fulltext_index(self, fields: List[str], name: Optional[str] = None) -> Dict[str, Any]:
        """Add a fulltext index."""
        index = {
            "type": "fulltext",
            "fields": fields,
            "name": name or f"fulltext_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        return index
    
    def add_geo_index(self, fields: List[str], name: Optional[str] = None) -> Dict[str, Any]:
        """Add a geo index."""
        index = {
            "type": "geo",
            "fields": fields,
            "name": name or f"geo_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        return index
    
    def add_persistent_index(
        self,
        fields: List[str],
        unique: bool = False,
        sparse: bool = False,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a persistent index."""
        index = {
            "type": "persistent",
            "fields": fields,
            "unique": unique,
            "sparse": sparse,
            "name": name or f"persistent_idx_{len(self.indexes)}"
        }
        self.indexes.append(index)
        return index


class AQLExecutor:
    """Simulated AQL executor for the example."""
    
    def __init__(self, db: SimulatedArangoDB):
        self.db = db
    
    def execute(
        self,
        query: str,
        bind_vars: Dict[str, Any] = None,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Execute an AQL query."""
        # This is a very simplified simulation of AQL queries
        # In a real implementation, this would parse and execute the query
        
        # For this example, we'll just return all documents from the first collection
        if not self.db.collections:
            return []
        
        first_collection = next(iter(self.db.collections.values()))
        return list(first_collection.documents.values())


# Async wrapper for ArangoDB operations
class AsyncArangoDBClient:
    """
    Async wrapper for ArangoDB operations.
    
    This class demonstrates the recommended pattern for handling
    database operations asynchronously using asyncio.to_thread.
    """
    
    def __init__(self, url: str, db_name: str, username: str = "root", password: str = ""):
        """
        Initialize the async ArangoDB client.
        
        Args:
            url: The URL of the ArangoDB server
            db_name: The name of the database
            username: The username for authentication
            password: The password for authentication
        """
        self.client = ArangoDBClient(url, db_name, username, password)
        logger.info(f"Initialized async ArangoDB client for {url}, database: {db_name}")
    
    async def connect(self) -> Any:
        """
        Connect to the ArangoDB database asynchronously.
        
        Returns:
            Database connection
        """
        return await asyncio.to_thread(self.client.connect)
    
    async def get_db(self) -> Any:
        """
        Get the database connection asynchronously.
        
        Returns:
            Database connection
        """
        return await asyncio.to_thread(self.client.get_db)
    
    async def collection(self, name: str, create: bool = True) -> Any:
        """
        Get a collection asynchronously.
        
        Args:
            name: The name of the collection
            create: Whether to create the collection if it doesn't exist
            
        Returns:
            Collection object
        """
        return await asyncio.to_thread(self.client.collection, name, create)
    
    async def create_document(self, collection_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a document in a collection asynchronously.
        
        Args:
            collection_name: The name of the collection
            document: The document to create
            
        Returns:
            The created document
        """
        return await asyncio.to_thread(self.client.create_document, collection_name, document)
    
    async def get_document(self, collection_name: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by key asynchronously.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            
        Returns:
            The document, or None if not found
        """
        return await asyncio.to_thread(self.client.get_document, collection_name, key)
    
    async def update_document(
        self,
        collection_name: str,
        key: str,
        document: Dict[str, Any],
        keep_none: bool = False
    ) -> Dict[str, Any]:
        """
        Update a document asynchronously.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            document: The document fields to update
            keep_none: Whether to keep None values
            
        Returns:
            The updated document
        """
        return await asyncio.to_thread(
            self.client.update_document,
            collection_name,
            key,
            document,
            keep_none
        )
    
    async def delete_document(self, collection_name: str, key: str) -> bool:
        """
        Delete a document asynchronously.
        
        Args:
            collection_name: The name of the collection
            key: The key of the document
            
        Returns:
            True if the document was deleted, False otherwise
        """
        return await asyncio.to_thread(self.client.delete_document, collection_name, key)
    
    async def execute_query(
        self,
        query: str,
        bind_vars: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Execute an AQL query asynchronously.
        
        Args:
            query: The AQL query
            bind_vars: The bind variables for the query
            batch_size: The batch size for the query
            
        Returns:
            The query results
        """
        return await asyncio.to_thread(
            self.client.execute_query,
            query,
            bind_vars,
            batch_size
        )
    
    async def create_index(
        self,
        collection_name: str,
        index_type: str,
        fields: List[str],
        name: Optional[str] = None,
        unique: bool = False,
        sparse: bool = False
    ) -> Dict[str, Any]:
        """
        Create an index on a collection asynchronously.
        
        Args:
            collection_name: The name of the collection
            index_type: The type of index
            fields: The fields to index
            name: The name of the index
            unique: Whether the index should enforce uniqueness
            sparse: Whether the index should be sparse
            
        Returns:
            The created index
        """
        return await asyncio.to_thread(
            self.client.create_index,
            collection_name,
            index_type,
            fields,
            name,
            unique,
            sparse
        )


# Example usage
async def main():
    # Create an async ArangoDB client
    client = AsyncArangoDBClient(
        url="http://localhost:8529",
        db_name="mydb",
        username="root",
        password="password"
    )
    
    try:
        # Connect to the database
        await client.connect()
        
        # Create a collection
        collection_name = "users"
        await client.collection(collection_name)
        
        # Create an index
        await client.create_index(
            collection_name=collection_name,
            index_type="hash",
            fields=["email"],
            unique=True
        )
        
        # Create a document
        user = {
            "_key": "user1",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "active": True
        }
        
        created_user = await client.create_document(collection_name, user)
        print(f"Created user: {created_user['name']} with key: {created_user['_key']}")
        
        # Get the document
        retrieved_user = await client.get_document(collection_name, "user1")
        print(f"Retrieved user: {retrieved_user['name']}")
        
        # Update the document
        updated_user = await client.update_document(
            collection_name,
            "user1",
            {"age": 31, "last_login": "2023-01-01T00:00:00Z"}
        )
        print(f"Updated user: {updated_user['name']}, age: {updated_user['age']}")
        
        # Execute a query
        query = "FOR u IN users FILTER u.active == true RETURN u"
        results = await client.execute_query(query)
        print(f"Query returned {len(results)} results")
        
        # Delete the document
        deleted = await client.delete_document(collection_name, "user1")
        print(f"Deleted user: {deleted}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 