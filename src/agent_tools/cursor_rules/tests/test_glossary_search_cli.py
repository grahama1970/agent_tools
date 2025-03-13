#!/usr/bin/env python3
"""
CLI test script for demonstrating glossary search effectiveness.

This script shows how the glossary search can find relevant results
in various challenging search scenarios:
- Misspelled terms
- Conceptual queries
- Term variations
- Semantic matching with term+definition combinations
"""

import asyncio
import os
from arango import ArangoClient
from tabulate import tabulate
import re

from agent_tools.cursor_rules.core.glossary import glossary_search, format_for_embedding
from agent_tools.cursor_rules.core.glossary_search import semantic_term_definition_search_sync, analyze_user_query

# Configure min similarity for different test cases
EXACT_MODE = 0.98  # Very strict matching
FUZZY_MODE = 0.65  # More lenient matching for typos/variations
SEMANTIC_MODE = 0.3  # Even more lenient for conceptual matching
VERY_LENIENT_MODE = 0.25  # For testing with significant typos

async def test_semantic_term_definition_matching():
    """Test specifically the semantic matching between queries and term+definition combinations."""
    print('\n===== SEMANTIC TERM+DEFINITION MATCHING TEST =====')
    print('This test focuses on ensuring semantic search works when comparing queries to term+definition combinations')
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    view_name = 'test_semantic_glossary_view'
    
    # Queries that should match term+definition combinations semantically
    # Including typos and conceptual matches
    semantic_test_queries = [
        {
            "query": "thst semantic sesrch technique", # typos that would fail exact matching
            "expected_term": "Semantic Search",
            "description": "Typo example that should still match semantically"
        },
        {
            "query": "systems inspired by brains for computing", # conceptual match
            "expected_term": "Neural Network",
            "description": "Conceptual description that should match semantically"
        },
        {
            "query": "finding text similarity without exact keyword matching",
            "expected_term": "Semantic Search",
            "description": "Functional description that should match semantically"
        },
        {
            "query": "optimization method for minimizing loss functions", 
            "expected_term": "Gradient Descent",
            "description": "Alternative phrasing of definition"
        },
        {
            "query": "databas full-txt srch engine",  # multiple typos
            "expected_term": "ArangoSearch",
            "description": "Multiple typos example"
        }
    ]
    
    print("\nTesting semantic matching between queries and term+definition combinations:")
    
    for test_case in semantic_test_queries:
        query = test_case["query"]
        expected_term = test_case["expected_term"]
        description = test_case["description"]
        
        print(f"\n{'='*80}")
        print(f"QUERY: \"{query}\"")
        print(f"EXPECTED TERM: {expected_term}")
        print(f"DESCRIPTION: {description}")
        print(f"{'-'*80}")
        
        # First perform a direct semantic search for the term+definition
        semantic_results = semantic_term_definition_search_sync(
            db, 
            collection_name,
            query,
            semantic_threshold=0.25  # Low threshold to catch more potential matches
        )
        
        # Then use the more complete analyze_user_query function
        analysis = await analyze_user_query(
            db, 
            collection_name, 
            view_name, 
            query
        )
        
        # Display semantic search results
        print("\nDIRECT SEMANTIC SEARCH RESULTS:")
        if not semantic_results:
            print("  No direct semantic matches found")
        else:
            semantic_table = []
            found_expected = False
            
            for result in semantic_results:
                term = result.get('term', 'N/A')
                if term == expected_term:
                    found_expected = True
                
                # Format the combined term+definition that was used for embedding
                combined = format_for_embedding(
                    result.get('term', ''), 
                    result.get('definition', '')
                )
                
                # Truncate for display
                definition = result.get('definition', '')
                if len(definition) > 50:
                    definition = definition[:47] + '...'
                    
                semantic_table.append([
                    term,
                    definition,
                    f"{result.get('similarity_score', 0)*100:.1f}%",
                    "✓" if term == expected_term else ""
                ])
            
            print(tabulate(
                semantic_table,
                headers=['Term', 'Definition', 'Similarity', 'Expected Match'],
                tablefmt='grid'
            ))
            
            if found_expected:
                print(f"✅ Successfully found the expected term: {expected_term}")
            else:
                print(f"❌ Failed to find the expected term: {expected_term}")
        
        # Display analyze_user_query results
        print("\nCOMPLETE ANALYSIS RESULTS:")
        if not analysis or not analysis.get("semantic_results"):
            print("  No semantic results in complete analysis")
        else:
            found_in_analysis = False
            analysis_table = []
            
            for result in analysis["semantic_results"]:
                term = result.get('term', 'N/A')
                if term == expected_term:
                    found_in_analysis = True
                
                # Truncate for display
                definition = result.get('definition', '')
                if len(definition) > 50:
                    definition = definition[:47] + '...'
                    
                analysis_table.append([
                    term,
                    definition,
                    f"{result.get('similarity_score', 0)*100:.1f}%",
                    "✓" if term == expected_term else ""
                ])
            
            print(tabulate(
                analysis_table,
                headers=['Term', 'Definition', 'Similarity', 'Expected Match'],
                tablefmt='grid'
            ))
            
            if found_in_analysis:
                print(f"✅ Complete analysis found the expected term: {expected_term}")
            else:
                print(f"❌ Complete analysis failed to find the expected term: {expected_term}")
                
    print('\n===== TEST COMPLETE =====')

async def test_glossary_search():
    """Run glossary search tests with different parameters to show where it succeeds."""
    print('\n===== GLOSSARY SEARCH EFFECTIVENESS TEST =====')
    print('This test demonstrates where glossary search succeeds when other approaches might fail')
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    
    # Test queries organized by category
    test_scenarios = [
        {
            "name": "Exact Term Queries",
            "description": "Direct searches using exact term names where BM25 should work well",
            "queries": [
                "Neural Network", 
                "BM25", 
                "ArangoSearch"
            ],
            "min_similarity": EXACT_MODE
        },
        {
            "name": "Misspelled Terms",
            "description": "Searches with typos where strict BM25 would fail but fuzzy search helps",
            "queries": [
                "What is a nueral network?",  # misspelled 'neural'
                "How does embeding work?",    # misspelled 'embedding'
                "Arango serch capabilities"   # misspelled 'search'
            ],
            "min_similarity": FUZZY_MODE
        },
        {
            "name": "Conceptual Queries",
            "description": "Searches that don't directly mention terms but relate conceptually where semantic search helps",
            "queries": [
                "How do computers understand text meaning?",     # related to embeddings
                "Method that combines vector and keyword search", # related to hybrid search
                "Finding documents with similar meanings"         # related to semantic search
            ],
            "min_similarity": SEMANTIC_MODE
        },
        {
            "name": "Typo Semantic Queries",
            "description": "Queries with typos that should still match semantically with term+definition",
            "queries": [
                "thst semantic sesrch technique",  # typos in semantic search
                "databas full-txt srch engine",    # typos in ArangoSearch
                "neural computng systms"           # typos in Neural Network
            ],
            "min_similarity": VERY_LENIENT_MODE
        }
    ]
    
    # Run each test scenario
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"DESCRIPTION: {scenario['description']}")
        print(f"SIMILARITY THRESHOLD: {scenario['min_similarity']}")
        print(f"{'='*80}")
        
        for query in scenario["queries"]:
            print(f"\nQUERY: \"{query}\"")
            print('-'*40)
            
            # Run search with specified similarity
            results = await glossary_search(
                db, 
                query, 
                collection_name=collection_name, 
                min_similarity=scenario["min_similarity"],
                verbose=True
            )
            
            if not results:
                print("NO RESULTS FOUND")
            else:
                # Display results in a table
                table_data = []
                for doc, hybrid_score, bm25_score, similarity_score in results:
                    # Truncate definition for display
                    definition = doc.get('definition', '')
                    if len(definition) > 60:
                        definition = definition[:57] + '...'
                        
                    table_data.append([
                        doc.get('term', 'N/A'),
                        definition,
                        f"{bm25_score*100:.1f}%",
                        f"{similarity_score*100:.1f}%",
                        f"{hybrid_score*100:.1f}%"
                    ])
                
                print(tabulate(
                    table_data,
                    headers=['Term', 'Definition', 'BM25 Score', 'Vector Score', 'Hybrid Score'],
                    tablefmt='grid'
                ))
    
    print('\n===== TEST COMPLETE =====')

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_semantic_term_definition_matching())
    # Uncomment to run the general glossary search test
    # asyncio.run(test_glossary_search()) 