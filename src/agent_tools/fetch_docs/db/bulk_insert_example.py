#!/usr/bin/env python3
"""
Example demonstrating the proper way to do bulk document insertion using ArangoDB's import_bulk method
"""

from arango import ArangoClient
from typing import List, Dict, Any
import time
from loguru import logger


def insert_documents(
    documents: List[Dict[str, Any]],
    host: str = "http://localhost:8529",
    database: str = "_system",
    username: str = "root",
    password: str = "",
    collection_name: str = "test_collection",
):
    """
    Insert documents using ArangoDB's built-in import_bulk method
    """
    client = ArangoClient(hosts=host)
    db = client.db(database, username=username, password=password)

    # Create collection if it doesn't exist
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)

    collection = db.collection(collection_name)

    # Use the proper import_bulk method
    logger.info("Starting bulk import...")
    start_time = time.time()

    result = collection.import_bulk(
        documents,
        halt_on_error=True,  # Stop if any document fails
        details=True,  # Get detailed information about the import
        on_duplicate="error",  # Fail if documents already exist
    )

    end_time = time.time()
    logger.info(f"Bulk import completed in {end_time - start_time:.2f} seconds")

    return result


def main():
    # Example usage
    test_docs = [{"name": f"test_{i}", "value": i} for i in range(1000)]

    result = insert_documents(documents=test_docs)

    # Log the results
    if isinstance(result, dict):
        logger.info(f"Created: {result.get('created', 0)} documents")
        logger.info(f"Errors: {result.get('errors', 0)}")
        if result.get("details"):
            logger.info("Details:", result["details"])


if __name__ == "__main__":
    main()
