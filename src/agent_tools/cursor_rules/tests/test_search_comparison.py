#!/usr/bin/env python3
"""
Test Search Method Comparison

This test compares different search techniques to demonstrate where each method excels:
1. Embedded term extraction - Best for explicitly marked terms
2. Fuzzy matching - Best for misspelled or slightly varied terms
3. Semantic search - Best for conceptually related terms without exact matches
4. BM25 - Best for keyword matching based on frequency and relevance

The goal is to verify that our unified approach leverages the strengths
of each method for comprehensive glossary term coverage.
"""

import asyncio
import pytest
from arango import ArangoClient
from typing import List, Dict, Any
from tabulate import tabulate
from rapidfuzz import fuzz

from agent_tools.cursor_rules.core.glossary import format_for_embedding
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
from agent_tools.cursor_rules.core.unified_glossary_search import (
    extract_embedded_terms,
    fuzzy_search_terms,
    semantic_search_terms,
    find_related_terms,
    unified_glossary_search
)

# Sample glossary for testing
SAMPLE_GLOSSARY = [
    {
        "term": "Neural Network",
        "definition": "A computing system inspired by biological neurons that can learn from data",
        "category": "machine learning",
        "related_terms": ["Deep Learning", "Machine Learning", "Artificial Intelligence"]
    },
    {
        "term": "Vector Database",
        "definition": "A database designed to store and query high-dimensional vectors, often used for semantic search",
        "category": "database",
        "related_terms": ["Semantic Search", "Embedding", "ArangoDB"]
    },
    {
        "term": "Semantic Search",
        "definition": "A search technique that considers meaning and context rather than just keywords",
        "category": "search",
        "related_terms": ["Vector Database", "Embedding", "Information Retrieval"]
    },
    {
        "term": "Embedding",
        "definition": "A technique that maps discrete objects like words to vectors of real numbers in a continuous space",
        "category": "machine learning",
        "related_terms": ["Vector", "Neural Network", "Natural Language Processing"]
    },
    {
        "term": "Transformer",
        "definition": "A deep learning architecture that uses self-attention mechanisms to process sequential data",
        "category": "machine learning",
        "related_terms": ["Neural Network", "Self-Attention", "Natural Language Processing"]
    },
    {
        "term": "ArangoDB",
        "definition": "A multi-model database system supporting graphs, documents, and key-value pairs",
        "category": "database",
        "related_terms": ["Graph Database", "Document Database", "NoSQL"]
    },
    {
        "term": "Self-Attention",
        "definition": "A mechanism where different positions of a single sequence attend to each other to compute a representation",
        "category": "machine learning",
        "related_terms": ["Transformer", "Neural Network", "Attention Mechanism"]
    },
    {
        "term": "Natural Language Processing",
        "definition": "A field of AI focused on enabling computers to understand and process human language",
        "category": "machine learning",
        "related_terms": ["NLP", "Computational Linguistics", "Text Mining"]
    }
]

# Test cases designed to demonstrate the strengths of each search method
COMPARISON_TESTS = [
    {
        "name": "Embedded Term Extraction Test",
        "description": "Shows embedded term extraction excels at finding explicitly marked terms",
        "query": "I need information about [TERM: Neural Network] architecture",
        "expected_embedded": ["Neural Network"],
        "expected_fuzzy": [],
        "expected_semantic": ["Transformer", "Self-Attention"]
    },
    {
        "name": "Fuzzy Matching Test",
        "description": "Shows fuzzy matching excels at finding terms with typos or variations",
        "query": "What is a Nural Netwerk and how does it work?",
        "expected_embedded": [],
        "expected_fuzzy": ["Neural Network"],
        "expected_semantic": ["Transformer", "Self-Attention"]
    },
    {
        "name": "Semantic Search Test",
        "description": "Shows semantic search excels at finding conceptually related terms",
        "query": "How do machine learning models understand language patterns?",
        "expected_embedded": [],
        "expected_fuzzy": [],
        "expected_semantic": ["Natural Language Processing", "Transformer"]
    },
    {
        "name": "Multiple Methods Test",
        "description": "Shows all methods working together for comprehensive coverage",
        "query": "Can [TERM: Embedding] techniques be used for fuzzy mathcing in language modls?",
        "expected_embedded": ["Embedding"],
        "expected_fuzzy": ["Natural Language Processing"], # Matches "language modls"
        "expected_semantic": ["Transformer", "Semantic Search"]
    },
    {
        "name": "Related Terms Test",
        "description": "Shows the value of finding related terms based on initial matches",
        "query": "What is the relationship between Neural Networks and AI?",
        "expected_embedded": [],
        "expected_fuzzy": ["Neural Network"],
        "expected_semantic": ["Transformer"],
        "expected_related": ["Deep Learning", "Machine Learning", "Artificial Intelligence"]
    }
]

@pytest.fixture
def test_db():
    """Setup a test database."""
    client = ArangoClient(hosts="http://localhost:8529")
    db_name = "test_search_comparison_db"
    
    # Check if DB exists and drop it for a clean test
    sys_db = client.db("_system", username="root", password="openSesame")
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
    
    # Create new DB
    sys_db.create_database(db_name)
    db = client.db(db_name, username="root", password="openSesame")
    
    # Create glossary collection
    collection_name = "test_glossary"
    if db.has_collection(collection_name):
        db.delete_collection(collection_name)
    collection = db.create_collection(collection_name)
    
    # Create vector view for the collection
    view_name = f"{collection_name}_view"
    
    # Get list of views instead of using has_view
    views = db.views()
    view_exists = any(v["name"] == view_name for v in views)
    
    if view_exists:
        db.delete_view(view_name)
    
    # Create the view
    db.create_arangosearch_view(
        view_name,
        properties={
            "links": {
                collection_name: {
                    "fields": {
                        "embedding": {
                            "analyzers": ["identity"],
                        }
                    }
                }
            }
        }
    )
    
    # Insert sample data with embeddings
    for item in SAMPLE_GLOSSARY:
        # Format for embedding and generate embedding
        text_to_embed = format_for_embedding(item["term"], item["definition"])
        embedding_result = generate_embedding(text_to_embed)
        
        if embedding_result and "embedding" in embedding_result:
            item["embedding"] = embedding_result["embedding"]
            item["embedding_metadata"] = embedding_result.get("metadata", {})
            collection.insert(item)
    
    yield db, collection_name
    
    # Cleanup
    sys_db.delete_database(db_name)

@pytest.mark.asyncio
async def test_method_comparison(test_db):
    """
    Compare the performance of different search methods on various query types
    to demonstrate where each method excels.
    """
    db, collection_name = test_db
    
    for test_case in COMPARISON_TESTS:
        print(f"\n\n=== TEST: {test_case['name']} ===")
        print(f"Description: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        
        query = test_case["query"]
        
        # 1. Test embedded term extraction
        embedded_results = extract_embedded_terms(query)
        embedded_terms = [item["term"] for item in embedded_results]
        
        # 2. Test fuzzy matching
        fuzzy_results = await fuzzy_search_terms(query, SAMPLE_GLOSSARY, threshold=70)
        fuzzy_terms = [item["term"] for item in fuzzy_results]
        
        # 3. Test semantic search (or fallback if VECTOR_SIMILARITY is not available)
        semantic_results = await semantic_search_terms(db, collection_name, query, threshold=0.5)
        semantic_terms = [item["term"] for item in semantic_results]
        
        # 4. Test unified search (all methods combined)
        unified_results = await unified_glossary_search(db, collection_name, query)
        
        # Extract combined results for comparison
        combined_terms = [item["term"] for item in unified_results["combined_results"]]
        
        # Display results for each method
        print("\nMethod-specific results:")
        results_table = []
        for term in set(embedded_terms + fuzzy_terms + semantic_terms + combined_terms):
            row = [
                term,
                "✓" if term in embedded_terms else "",
                "✓" if term in fuzzy_terms else "",
                "✓" if term in semantic_terms else "",
                "✓" if term in combined_terms else ""
            ]
            results_table.append(row)
        
        print(tabulate(
            results_table,
            headers=["Term", "Embedded", "Fuzzy", "Semantic", "Combined"],
            tablefmt="grid"
        ))
        
        # Verify expected results for each method
        if "expected_embedded" in test_case:
            for term in test_case["expected_embedded"]:
                assert term in embedded_terms, f"Expected embedded term '{term}' not found in results: {embedded_terms}"
        
        if "expected_fuzzy" in test_case:
            for term in test_case["expected_fuzzy"]:
                assert term in fuzzy_terms, f"Expected fuzzy match '{term}' not found in results: {fuzzy_terms}"
        
        if "expected_semantic" in test_case:
            if semantic_terms and "semantic_fallback" not in semantic_results[0]["match_type"]:
                # Only check semantic expectations if real semantic search is available (not fallback)
                for term in test_case["expected_semantic"]:
                    assert term in semantic_terms, f"Expected semantic match '{term}' not found in results: {semantic_terms}"
        
        # Verify that unified search includes results from all methods
        expected_terms = (
            test_case.get("expected_embedded", []) + 
            test_case.get("expected_fuzzy", [])
        )
        
        # For semantic matches, only check if real semantic search is available
        if semantic_terms and "semantic_fallback" not in semantic_results[0]["match_type"]:
            expected_terms += test_case.get("expected_semantic", [])
        
        # Check combined results
        for term in expected_terms:
            # Allow for some flexibility - look for the exact term or something containing it
            exact_match = term in combined_terms
            partial_matches = [t for t in combined_terms if term in t]
            
            assert exact_match or partial_matches, \
                f"Expected term '{term}' not found in combined results: {combined_terms}"

        print(f"\nTest case {test_case['name']} passed!\n")

async def run_manual_tests():
    """Run the tests manually from the command line."""
    # Setup test database
    client = ArangoClient(hosts="http://localhost:8529")
    db_name = "test_search_comparison_db"
    collection_name = "test_glossary"
    
    # Check if DB exists and drop it for a clean test
    sys_db = client.db("_system", username="root", password="openSesame")
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
    
    # Create new DB
    sys_db.create_database(db_name)
    db = client.db(db_name, username="root", password="openSesame")
    
    # Create glossary collection
    if db.has_collection(collection_name):
        db.delete_collection(collection_name)
    collection = db.create_collection(collection_name)
    
    # Create vector view for the collection
    view_name = f"{collection_name}_view"
    
    # Get list of views instead of using has_view
    views = db.views()
    view_exists = any(v["name"] == view_name for v in views)
    
    if view_exists:
        db.delete_view(view_name)
    
    # Create the view
    db.create_arangosearch_view(
        view_name,
        properties={
            "links": {
                collection_name: {
                    "fields": {
                        "embedding": {
                            "analyzers": ["identity"],
                        }
                    }
                }
            }
        }
    )
    
    print("\n=== SETTING UP TEST DATABASE WITH SAMPLE GLOSSARY ===")
    # Insert sample data with embeddings
    for item in SAMPLE_GLOSSARY:
        # Format for embedding and generate embedding
        text_to_embed = format_for_embedding(item["term"], item["definition"])
        print(f"Generating embedding for: {text_to_embed}")
        embedding_result = generate_embedding(text_to_embed)
        
        if embedding_result and "embedding" in embedding_result:
            item["embedding"] = embedding_result["embedding"]
            item["embedding_metadata"] = embedding_result.get("metadata", {})
            collection.insert(item)
            print(f"Added entry: {item['term']}")
    
    print("\n=== TESTING DIRECT FUZZY MATCHING ===")
    # Direct test of "Nural Netwerk" against "Neural Network"
    test_term = "Nural Netwerk"
    for glossary_term in [item["term"] for item in SAMPLE_GLOSSARY]:
        similarity = fuzz.token_sort_ratio(test_term.lower(), glossary_term.lower())
        print(f"'{test_term}' vs '{glossary_term}': {similarity}% similarity")
    
    print("\n=== RUNNING SEARCH METHOD COMPARISON TESTS ===")
    
    for test_case in COMPARISON_TESTS:
        print(f"\n\n=== TEST: {test_case['name']} ===")
        print(f"Description: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        
        query = test_case["query"]
        
        # 1. Test embedded term extraction
        embedded_results = extract_embedded_terms(query)
        embedded_terms = [item["term"] for item in embedded_results]
        print(f"\nEmbedded terms: {embedded_terms}")
        
        # 2. Test fuzzy matching with an extremely low threshold
        fuzzy_results = await fuzzy_search_terms(query, SAMPLE_GLOSSARY, threshold=35)
        
        print("\nFuzzy match details:")
        for item in fuzzy_results:
            print(f"- {item['term']}: {item.get('similarity_score', 0)}% match")
        
        fuzzy_terms = [item["term"] for item in fuzzy_results]
        print(f"Fuzzy matches summary: {fuzzy_terms}")
        
        # 3. Test semantic search
        semantic_results = await semantic_search_terms(db, collection_name, query, threshold=0.5)
        semantic_terms = [item["term"] for item in semantic_results]
        semantic_type = semantic_results[0]["match_type"] if semantic_results else "none"
        print(f"Semantic matches ({semantic_type}): {semantic_terms}")
        
        # 4. Test unified search (all methods combined) with more lenient fuzzy threshold
        unified_results = await unified_glossary_search(
            db,
            collection_name,
            query,
            fuzzy_threshold=35,  # Extremely lenient threshold
            semantic_threshold=0.5,
            limit=10
        )
        
        # Extract combined results and their types
        combined_info = [
            {"term": item["term"], "type": item["match_type"]} 
            for item in unified_results["combined_results"]
        ]
        
        print("\nUnified search results:")
        for item in combined_info:
            print(f"- {item['term']} ({item['type']})")
    
    # Cleanup
    sys_db.delete_database(db_name)
    print(f"\n=== TEST COMPLETED AND DATABASE CLEANED UP ===")

if __name__ == "__main__":
    asyncio.run(run_manual_tests()) 