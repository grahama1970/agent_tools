#!/usr/bin/env python3
"""
Test script to display the content of the glossary collection.

This test demonstrates how to retrieve and display all terms and definitions
from the glossary collection in a tabulated format.

Documentation references:
- ArangoDB: https://www.arangodb.com/docs/stable/
- tabulate: https://pypi.org/project/tabulate/
"""

import asyncio
import pytest
import pytest_asyncio
import logging
from typing import Dict, List, Any
from arango import ArangoClient
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture(scope="module")
async def test_db():
    """Fixture to connect to the test database with the glossary collection."""
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    
    # Named functions for database operations
    def connect_to_db(client, db_name, username, password):
        return client.db(db_name, username=username, password=password)
    
    # Connect to test database
    db = await asyncio.to_thread(
        connect_to_db, client, "test_semantic_term_definition_db", "root", "openSesame"
    )
    
    yield db

@pytest.mark.asyncio
async def test_list_glossary_terms_and_definitions(test_db):
    """Test that retrieves and displays all terms and definitions from the glossary collection."""
    collection_name = "test_semantic_glossary"
    
    # Define a function to get documents from the collection
    def get_documents(db, collection_name):
        collection = db.collection(collection_name)
        return list(collection.all())
    
    # Get all documents from the collection
    documents = await asyncio.to_thread(get_documents, test_db, collection_name)
    
    # Prepare data for tabulate
    table_data = []
    for doc in documents:
        table_data.append([
            doc.get("term", "N/A"),
            doc.get("definition", "N/A"),
            doc.get("category", "N/A"),
            ", ".join(doc.get("related_terms", [])),
            doc.get("source", "N/A")
        ])
    
    # Sort by term name for consistent display
    table_data.sort(key=lambda x: x[0])
    
    # Display the table
    headers = ["Term", "Definition", "Category", "Related Terms", "Source"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info(f"\nGlossary Collection Content:\n{table}")
    
    # Print directly to console if running script directly
    print(f"\nGlossary Collection Content:\n{table}")
    
    # Verify we have glossary entries
    assert len(documents) > 0, "Expected glossary collection to contain entries"
    
    # Verify each document has required fields
    for doc in documents:
        assert "term" in doc, f"Document missing 'term' field: {doc}"
        assert "definition" in doc, f"Document missing 'definition' field: {doc}"
    
    # Return the table string for use in reports or UI
    return table

@pytest.mark.asyncio
async def test_retrieve_terms_by_category(test_db):
    """Test that retrieves and displays terms filtered by category."""
    collection_name = "test_semantic_glossary"
    
    # Define a function to execute AQL query
    def execute_query(db, query, bind_vars):
        return list(db.aql.execute(query, bind_vars=bind_vars))
    
    # Query to get terms by category
    query = f"""
    FOR doc IN {collection_name}
        FILTER doc.category == @category
        SORT doc.term
        RETURN {{
            term: doc.term,
            definition: doc.definition,
            category: doc.category,
            related_terms: doc.related_terms,
            source: doc.source
        }}
    """
    
    # Test with AI category
    ai_terms = await asyncio.to_thread(
        execute_query, 
        test_db, 
        query, 
        {"@collection": collection_name, "category": "AI"}
    )
    
    # Prepare data for tabulate
    table_data = []
    for doc in ai_terms:
        table_data.append([
            doc.get("term", "N/A"),
            doc.get("definition", "N/A"),
            ", ".join(doc.get("related_terms", []))
        ])
    
    # Display the table
    headers = ["AI Term", "Definition", "Related Terms"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info(f"\nAI Category Terms:\n{table}")
    
    # Print directly to console if running script directly
    print(f"\nAI Category Terms:\n{table}")
    
    # Verify we have AI category terms
    assert len(ai_terms) > 0, "Expected to find terms in the AI category"
    
    # Verify each document is in the AI category
    for doc in ai_terms:
        assert doc["category"] == "AI", f"Document not in AI category: {doc}"
    
    return table

# Function to be called directly when running the script
async def display_all_glossary_content():
    """Display all glossary content when running the script directly."""
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    
    def connect_to_db(client, db_name, username, password):
        return client.db(db_name, username=username, password=password)
    
    # Connect to test database
    db = await asyncio.to_thread(
        connect_to_db, client, "test_semantic_term_definition_db", "root", "openSesame"
    )
    
    # Call the test functions directly
    print("\n===== GLOSSARY COLLECTION CONTENT =====")
    await test_list_glossary_terms_and_definitions(db)
    
    print("\n===== TERMS BY CATEGORY =====")
    await test_retrieve_terms_by_category(db)

if __name__ == "__main__":
    # For manual testing - run this file directly
    print("Running glossary collection display script...")
    asyncio.run(display_all_glossary_content())
    print("\nCompleted display of glossary content.") 