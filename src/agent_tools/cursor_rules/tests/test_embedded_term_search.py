#!/usr/bin/env python3
"""
Test script for embedded term search functionality.

This test specifically validates that the system can find glossary terms when
user questions contain embedded terms in the format [Term: the term].

Documentation references:
- ArangoDB Search: https://www.arangodb.com/docs/stable/arangosearch.html
- ArangoDB Vectors: https://www.arangodb.com/docs/stable/arangosearch-vectors.html
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

import asyncio
import os
import pytest
import pytest_asyncio
import logging
from typing import Dict, List, Any
from arango import ArangoClient
from tabulate import tabulate

from agent_tools.cursor_rules.core.glossary_search import (
    semantic_term_definition_search,
    combined_semantic_fuzzy_search,
    analyze_user_query
)

from agent_tools.cursor_rules.core.glossary import (
    embed_term_in_query
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply the pytestmark for module-level event loop
pytestmark = pytest.mark.asyncio

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

async def test_embedded_term_search_found(test_db):
    """Test searching for embedded terms in user questions when the term exists."""
    collection_name = "test_semantic_glossary"
    view_name = "test_semantic_glossary_view"
    
    # Create questions with embedded terms
    questions = [
        "How does [Term: Neural Network] work in image recognition?",
        "Can you explain [Term: Gradient Descent] with a simple example?",
        "Why is [Term: Semantic Search] better than keyword search?",
        "How can I use [Term: Azure] for my application deployment?"
    ]
    
    expected_terms = [
        "Neural Network",
        "Gradient Descent",
        "Semantic Search",
        "Azure"
    ]
    
    # Test each question
    for i, question in enumerate(questions):
        logger.info(f"Testing question with embedded term: {question}")
        
        # Analyze the query to find embedded terms and their definitions
        analysis = await analyze_user_query(
            test_db, collection_name, view_name, question
        )
        
        # Verify that the embedded term was found
        found_terms = [item["term"] for item in analysis["all_results"]]
        expected_term = expected_terms[i]
        
        logger.info(f"Expected term: {expected_term}, Found terms: {found_terms}")
        assert expected_term in found_terms, f"Expected to find '{expected_term}' in {found_terms}"
        
        # Verify that we have the correct definition for the term
        for result in analysis["all_results"]:
            if result["term"] == expected_term:
                assert "definition" in result, f"Missing definition for term {expected_term}"
                logger.info(f"Found definition for {expected_term}: {result['definition']}")
                break
    
    logger.info("All embedded terms were successfully found and matched with definitions")

async def test_embedded_term_search_not_found(test_db):
    """Test searching for embedded terms in user questions when the term doesn't exist."""
    collection_name = "test_semantic_glossary"
    view_name = "test_semantic_glossary_view"
    
    # Create questions with non-existent embedded terms
    questions = [
        "How does [Term: Quantum Computing] relate to machine learning?",
        "Can you explain [Term: Reinforcement Learning] in simple terms?",
        "Is [Term: Natural Language Processing] better than rule-based systems?"
    ]
    
    non_existent_terms = [
        "Quantum Computing",
        "Reinforcement Learning",
        "Natural Language Processing"
    ]
    
    for i, question in enumerate(questions):
        logger.info(f"Testing question with non-existent embedded term: {question}")
        
        # Analyze the query to find embedded terms and their definitions
        analysis = await analyze_user_query(
            test_db, collection_name, view_name, question
        )
        
        # Check if the non-existent term is found in the extracted terms
        # (it should be extracted but not found in the database)
        assert "extracted_terms" in analysis, "Missing extracted_terms in analysis"
        
        # Verify that we don't get a direct match for the non-existent term
        for term in non_existent_terms:
            direct_match = False
            for result in analysis["all_results"]:
                if result["term"] == term:
                    direct_match = True
                    break
            
            assert not direct_match, f"Unexpectedly found non-existent term: {term}"
        
        # Verify that we have semantic results structure, but don't require results
        # Since these are truly non-existent terms, it's OK if there are no semantic matches
        assert "semantic_results" in analysis, "Missing semantic_results in analysis"
        
        logger.info(f"Non-existent term correctly handled: {non_existent_terms[i]}")
    
    logger.info("All non-existent embedded terms were correctly handled")

async def test_embed_term_in_query_function():
    """Test the embed_term_in_query function directly."""
    # Test cases for embedding terms in queries
    test_cases = [
        {
            "query": "How does neural networks work?",
            "term": "Neural Network",
            "definition": "A computing system inspired by biological neurons",
            "include_definition": False,
            "expected": "How does [TERM: Neural Network] work?"  # TERM is uppercase, and fixed pluralization
        },
        {
            "query": "Explain gradient descent algorithm please",
            "term": "Gradient Descent",
            "definition": "An optimization algorithm used to minimize a function",
            "include_definition": True,
            "expected": "Explain [TERM: Gradient Descent | An optimization algorithm used to minimize a function] algorithm please"  # TERM is uppercase
        },
        {
            "query": "What is semantic search?",
            "term": "Semantic Search",
            "definition": "A search technique that considers meaning and context",
            "include_definition": False,
            "expected": "What is [TERM: Semantic Search]?"  # TERM is uppercase
        }
    ]
    
    for i, case in enumerate(test_cases):
        query = case["query"]
        term = case["term"]
        definition = case["definition"]
        include_definition = case["include_definition"]
        expected = case["expected"]
        
        # Call the embed_term_in_query function
        result = embed_term_in_query(query, term, definition, include_definition)
        
        # Verify the result
        assert result == expected, f"Case {i+1}: Expected '{expected}', got '{result}'"
        logger.info(f"Case {i+1}: Successfully embedded term in query")
    
    logger.info("All embed_term_in_query test cases passed")

async def display_embedded_term_search_results():
    """Function to display embedded term search results when running directly."""
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    
    def connect_to_db(client, db_name, username, password):
        return client.db(db_name, username=username, password=password)
    
    # Connect to test database
    db = await asyncio.to_thread(
        connect_to_db, client, "test_semantic_term_definition_db", "root", "openSesame"
    )
    
    collection_name = "test_semantic_glossary"
    view_name = "test_semantic_glossary_view"
    
    # Test with a question containing an embedded term
    question = "How does [Term: Neural Network] work for image recognition?"
    print(f"\nAnalyzing question: {question}")
    
    # Analyze the query
    analysis = await analyze_user_query(db, collection_name, view_name, question)
    
    # Display the results
    print("\n===== EMBEDDED TERM SEARCH RESULTS =====")
    
    # Display extracted terms
    print(f"\nExtracted Terms: {analysis['extracted_terms']}")
    
    # Display semantic results
    semantic_results = analysis["semantic_results"]
    semantic_table = []
    for result in semantic_results:
        semantic_table.append([
            result.get("term", "N/A"),
            result.get("definition", "N/A"),
            result.get("similarity_score", "N/A")
        ])
    
    print("\nSemantic Search Results:")
    print(tabulate(semantic_table, 
                  headers=["Term", "Definition", "Similarity Score"], 
                  tablefmt="grid"))
    
    # Display all results
    all_results = analysis["all_results"]
    all_table = []
    for result in all_results:
        all_table.append([
            result.get("term", "N/A"),
            result.get("definition", "N/A"),
            result.get("category", "N/A"),
            ", ".join(result.get("related_terms", []))
        ])
    
    print("\nCombined Results:")
    print(tabulate(all_table, 
                  headers=["Term", "Definition", "Category", "Related Terms"], 
                  tablefmt="grid"))
    
    # Test with a non-existent term
    question = "What is [Term: Quantum Computing] used for?"
    print(f"\nAnalyzing question with non-existent term: {question}")
    
    # Analyze the query
    analysis = await analyze_user_query(db, collection_name, view_name, question)
    
    # Display the results
    print("\n===== NON-EXISTENT TERM SEARCH RESULTS =====")
    
    # Display extracted terms
    print(f"\nExtracted Terms: {analysis['extracted_terms']}")
    
    # Display semantic results
    semantic_results = analysis["semantic_results"]
    semantic_table = []
    for result in semantic_results:
        semantic_table.append([
            result.get("term", "N/A"),
            result.get("definition", "N/A"),
            result.get("similarity_score", "N/A")
        ])
    
    print("\nSemantic Search Results (Terms similar to the non-existent term):")
    print(tabulate(semantic_table, 
                  headers=["Term", "Definition", "Similarity Score"], 
                  tablefmt="grid"))
    
    # Display all results
    all_results = analysis["all_results"]
    all_table = []
    for result in all_results:
        all_table.append([
            result.get("term", "N/A"),
            result.get("definition", "N/A"),
            result.get("category", "N/A"),
            ", ".join(result.get("related_terms", []))
        ])
    
    print("\nCombined Results:")
    print(tabulate(all_table, 
                  headers=["Term", "Definition", "Category", "Related Terms"], 
                  tablefmt="grid"))

if __name__ == "__main__":
    # For manual testing - run this file directly
    print("Running embedded term search tests...")
    asyncio.run(display_embedded_term_search_results())
    print("\nCompleted embedded term search tests.") 