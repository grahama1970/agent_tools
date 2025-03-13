#!/usr/bin/env python3
"""
Test User Question Term Extraction

This test validates that:
1. Terms embedded in user questions are properly extracted
2. The system finds relevant terms and definitions based on user questions
3. Both exact matches and semantic matches are correctly identified
4. All similar terms and definitions are included in results
"""

import asyncio
import pytest
from arango import ArangoClient
from typing import List, Dict, Any
from tabulate import tabulate

from agent_tools.cursor_rules.core.glossary import format_for_embedding
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
from agent_tools.cursor_rules.core.glossary_search import (
    semantic_term_definition_search_sync,
    combined_semantic_fuzzy_search_sync,
    extract_terms_from_query,
    analyze_user_question
)

# Sample glossary for testing
SAMPLE_GLOSSARY = [
    {
        "term": "Neural Network",
        "definition": "A computing system inspired by biological neurons that can learn from data",
        "category": "ml",
        "related_terms": ["Deep Learning", "Artificial Intelligence"],
        "source": "Test"
    },
    {
        "term": "Vector Database",
        "definition": "A database designed to store and query high-dimensional vectors, often used for semantic search",
        "category": "database",
        "related_terms": ["Semantic Search", "Embedding"],
        "source": "Test"
    },
    {
        "term": "Transformer",
        "definition": "A deep learning architecture that uses self-attention mechanisms to process sequential data",
        "category": "ml",
        "related_terms": ["Neural Network", "Self-Attention"],
        "source": "Test"
    },
    {
        "term": "Semantic Search",
        "definition": "A search technique that considers meaning and context of terms rather than just keywords",
        "category": "search",
        "related_terms": ["Vector Database", "Embedding"],
        "source": "Test"
    },
    {
        "term": "Embedding",
        "definition": "A technique that maps discrete objects like words to vectors of real numbers in a continuous space",
        "category": "ml",
        "related_terms": ["Vector", "Semantic Search"],
        "source": "Test"
    }
]

# Test user questions with embedded terms
TEST_USER_QUESTIONS = [
    {
        "question": "How does [TERM: Neural Network] work for image recognition?",
        "explicit_term": "Neural Network",
        "expected_similar_terms": ["Transformer", "Embedding"]
    },
    {
        "question": "What is the benefit of using a [TERM: Vector Database] over traditional databases?",
        "explicit_term": "Vector Database",
        "expected_similar_terms": ["Semantic Search", "Embedding"]
    },
    {
        "question": "Can you explain how [TERM: Semantic Search] is better than keyword search?",
        "explicit_term": "Semantic Search",
        "expected_similar_terms": ["Vector Database", "Embedding"]
    },
    {
        "question": "What is a Transformer architecture and how does it compare to RNNs?",
        "explicit_term": None,  # No explicit term, should find semantically
        "expected_similar_terms": ["Transformer", "Neural Network"]
    }
]

@pytest.fixture
def test_db():
    """Setup a test database."""
    client = ArangoClient(hosts="http://localhost:8529")
    db_name = "test_user_question_term_extraction_db"
    
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
    if db.has_view(view_name):
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

def test_extract_embedded_terms():
    """Test extracting embedded terms from user questions."""
    # Test extraction of embedded terms
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_term = test_case["explicit_term"]
        
        # Extract terms
        extracted = extract_terms_from_query(question)
        
        if expected_term:
            # Should find the embedded term
            assert len(extracted) > 0, f"Failed to extract terms from: {question}"
            assert expected_term in extracted, f"Failed to extract {expected_term} from: {question}"
            print(f"Successfully extracted {expected_term} from: {question}")
        else:
            # There's no embedded term, so extraction should return empty
            assert len(extracted) == 0, f"Incorrectly extracted terms from: {question}"
            print(f"Correctly found no embedded terms in: {question}")

def test_semantic_search_for_embedded_terms(test_db):
    """Test semantic search finds terms embedded in user questions."""
    db, collection_name = test_db
    
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_term = test_case["explicit_term"]
        
        # Perform semantic search
        results = semantic_term_definition_search_sync(
            db, 
            collection_name, 
            question,
            limit=5
        )
        
        if expected_term:
            # Should find the embedded term as a top result
            assert len(results) > 0, f"No results found for: {question}"
            terms = [r.get("term") for r in results]
            assert expected_term in terms, f"Expected term {expected_term} not found in results: {terms}"
            print(f"Semantic search correctly found {expected_term} for: {question}")
        else:
            # Should still find semantically relevant terms
            assert len(results) > 0, f"No semantic results found for: {question}"
            print(f"Semantic search found {len(results)} results for: {question}")

def test_combined_search_includes_all_similar_terms(test_db):
    """Test that combined search includes both exact and similar terms."""
    db, collection_name = test_db
    
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_similar_terms = test_case["expected_similar_terms"]
        
        # Perform combined search
        results = combined_semantic_fuzzy_search_sync(
            db,
            collection_name,
            question,
            limit=10
        )
        
        # Should find both exact and similar terms
        assert len(results) > 0, f"No results found for: {question}"
        
        found_terms = [r.get("term") for r in results]
        for term in expected_similar_terms:
            assert term in found_terms, f"Expected similar term {term} not found in results: {found_terms}"
        
        print(f"Combined search found the following terms for: {question}")
        print(f"  - Found terms: {found_terms}")
        print(f"  - Expected similar terms: {expected_similar_terms}")

def test_user_question_analysis_comprehensive(test_db):
    """Test comprehensive analysis of user questions."""
    db, collection_name = test_db
    
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_term = test_case["explicit_term"]
        expected_similar_terms = test_case["expected_similar_terms"]
        
        # Perform full analysis
        analysis = analyze_user_question(
            db,
            collection_name, 
            question,
            semantic_limit=5,
            fuzzy_limit=5,
            combined_limit=10
        )
        
        # Analyze results
        print(f"\n=== ANALYSIS FOR: {question} ===")
        
        # Check extracted terms
        extracted = analysis.get("extracted_terms", [])
        if expected_term:
            assert expected_term in extracted, f"Expected term {expected_term} not in extracted terms: {extracted}"
            print(f"✅ Extracted terms: {extracted}")
        else:
            print(f"ℹ️ Extracted terms: {extracted}")
        
        # Check semantic results
        semantic_results = analysis.get("semantic_results", [])
        semantic_terms = [r.get("term") for r in semantic_results]
        print(f"ℹ️ Semantic search results: {semantic_terms}")
        
        # Check combined results
        combined_results = analysis.get("combined_results", [])
        combined_terms = [r.get("term") for r in combined_results]
        print(f"ℹ️ Combined search results: {combined_terms}")
        
        # Verify all expected similar terms are found
        for term in expected_similar_terms:
            assert term in combined_terms, f"Expected similar term {term} not found in combined results: {combined_terms}"
        
        print(f"✅ All expected similar terms found: {expected_similar_terms}")
        
        # Display found definitions
        print("\nDefinitions found:")
        table_data = []
        for result in combined_results:
            term = result.get("term")
            definition = result.get("definition", "")
            short_def = definition[:50] + "..." if len(definition) > 50 else definition
            table_data.append([term, short_def])
        
        if table_data:
            print(tabulate(table_data, headers=["Term", "Definition"], tablefmt="grid"))

def run_tests():
    """Run tests manually from command line."""
    # Setup database
    client = ArangoClient(hosts="http://localhost:8529")
    db_name = "test_user_question_term_extraction_db"
    
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
    if db.has_view(view_name):
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
    
    print(f"\n=== SETTING UP TEST DATABASE WITH SAMPLE GLOSSARY ===")
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
    
    print(f"\n=== RUNNING TERM EXTRACTION TESTS ===")
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_term = test_case["explicit_term"]
        
        print(f"\nTesting extraction for: {question}")
        # Extract terms
        extracted = extract_terms_from_query(question)
        
        if expected_term:
            if expected_term in extracted:
                print(f"✅ Successfully extracted: {extracted}")
            else:
                print(f"❌ Failed to extract {expected_term}. Got: {extracted}")
        else:
            if len(extracted) == 0:
                print(f"✅ Correctly found no embedded terms")
            else:
                print(f"❌ Incorrectly extracted: {extracted}")
    
    print(f"\n=== RUNNING COMPREHENSIVE ANALYSIS TESTS ===")
    for test_case in TEST_USER_QUESTIONS:
        question = test_case["question"]
        expected_term = test_case["explicit_term"]
        expected_similar_terms = test_case["expected_similar_terms"]
        
        print(f"\nAnalyzing: {question}")
        # Perform full analysis
        analysis = analyze_user_question(
            db,
            collection_name, 
            question,
            semantic_limit=5,
            fuzzy_limit=5,
            combined_limit=10
        )
        
        # Check extracted terms
        extracted = analysis.get("extracted_terms", [])
        if expected_term:
            if expected_term in extracted:
                print(f"✅ Extracted terms: {extracted}")
            else:
                print(f"❌ Failed to extract {expected_term}. Got: {extracted}")
        else:
            print(f"ℹ️ Extracted terms: {extracted}")
        
        # Check combined results
        combined_results = analysis.get("combined_results", [])
        combined_terms = [r.get("term") for r in combined_results]
        
        # Verify all expected similar terms are found
        missing_terms = []
        for term in expected_similar_terms:
            if term not in combined_terms:
                missing_terms.append(term)
        
        if missing_terms:
            print(f"❌ Missing expected terms: {missing_terms}")
            print(f"  Found terms: {combined_terms}")
        else:
            print(f"✅ All expected similar terms found: {expected_similar_terms}")
        
        # Display found definitions
        print("\nDefinitions found:")
        table_data = []
        for result in combined_results:
            term = result.get("term")
            definition = result.get("definition", "")
            short_def = definition[:50] + "..." if len(definition) > 50 else definition
            table_data.append([term, short_def])
        
        if table_data:
            print(tabulate(table_data, headers=["Term", "Definition"], tablefmt="grid"))
    
    # Cleanup
    sys_db.delete_database(db_name)
    print(f"\n=== TEST COMPLETED AND DATABASE CLEANED UP ===")

if __name__ == "__main__":
    run_tests() 