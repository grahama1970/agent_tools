#!/usr/bin/env python3
"""
Comparative Search Testing

This script demonstrates the relative strengths of different search methods:
1. BM25 search - Good at exact keyword matching
2. Semantic search - Good at conceptual/meaning matching
3. Fuzzy search - Good at handling typos/variations
4. Combined approaches - Best overall coverage

For each test case, we show where one method succeeds while others fail,
proving the value of our multi-method approach.
"""

import asyncio
import os
from arango import ArangoClient
from tabulate import tabulate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup search functions
from agent_tools.cursor_rules.core.glossary_search import (
    semantic_term_definition_search_sync,
    fuzzy_glossary_search_sync,
    combined_semantic_fuzzy_search_sync
)

from agent_tools.cursor_rules.core.cursor_rules import (
    generate_embedding,
    hybrid_search
)

async def run_search_comparison():
    """
    Run a comparison of different search methods, highlighting strengths and weaknesses.
    """
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db('test_semantic_term_definition_db', username='root', password='openSesame')
    collection_name = 'test_semantic_glossary'
    view_name = 'test_semantic_glossary_view'

    # Test cases designed to show where specific methods excel
    test_cases = [
        {
            "category": "BM25 Strengths",
            "description": "Cases where exact keyword matching succeeds but semantic search might fail",
            "queries": [
                "ArangoSearch",                           # Direct technical term match
                "BM25 ranking function",                  # Specific technical phrase
                "gradient descent optimization algorithm" # Term with technical context
            ]
        },
        {
            "category": "Semantic Search Strengths",
            "description": "Cases where conceptual matching succeeds but keyword matching might fail",
            "queries": [
                "brain-inspired computing system",        # Conceptual match to Neural Network
                "document relevance scoring",             # Conceptual match to BM25
                "finding meaning in text data"            # Conceptual match to Semantic Search
            ]
        },
        {
            "category": "Fuzzy Search Strengths",
            "description": "Cases where typo handling succeeds where both others might fail",
            "queries": [
                "nueral ntwrk",                           # Severe misspelling
                "semntic srch",                           # Multiple typos
                "arangosrch for textsearch"               # Missing characters
            ]
        }
    ]

    print("\n=== SEARCH METHOD COMPARISON TEST ===")
    print("This test compares the effectiveness of different search approaches")
    print("by highlighting scenarios where each method excels.\n")

    # Run each test category
    for case in test_cases:
        print(f"\n\n{'='*80}")
        print(f"CATEGORY: {case['category']}")
        print(f"DESCRIPTION: {case['description']}")
        print(f"{'='*80}\n")

        for query in case["queries"]:
            print(f"\nQUERY: \"{query}\"")
            print(f"{'-'*40}")

            # Generate embedding for semantic search
            query_embedding = generate_embedding(query)
            query_vector = query_embedding.get("embedding", []) if query_embedding else []

            # 1. Run BM25-focused search (keyword search)
            print("\n1. BM25 SEARCH RESULTS:")
            bm25_query = f"""
            FOR doc IN {view_name}
                SEARCH ANALYZER(
                    doc.term IN TOKENS(@query, "text_en") OR
                    doc.definition IN TOKENS(@query, "text_en"),
                    "text_en"
                )
                SORT BM25(doc) DESC
                LIMIT 3
                RETURN {{
                    term: doc.term,
                    definition: doc.definition,
                    score: BM25(doc)
                }}
            """
            
            bm25_cursor = db.aql.execute(bm25_query, bind_vars={"query": query})
            bm25_results = list(bm25_cursor)
            
            if not bm25_results:
                print("  No results found with BM25 search")
            else:
                bm25_table = []
                for result in bm25_results:
                    # Truncate definition for display
                    definition = result.get('definition', '')
                    if len(definition) > 50:
                        definition = definition[:47] + '...'
                        
                    bm25_table.append([
                        result.get('term', 'N/A'),
                        definition,
                        f"{result.get('score', 0):.3f}"
                    ])
                
                print(tabulate(
                    bm25_table,
                    headers=['Term', 'Definition', 'BM25 Score'],
                    tablefmt='grid'
                ))

            # 2. Run Semantic search
            print("\n2. SEMANTIC SEARCH RESULTS:")
            semantic_results = semantic_term_definition_search_sync(
                db, 
                collection_name,
                query,
                semantic_threshold=0.25  # Lower threshold to catch more matches
            )
            
            if not semantic_results:
                print("  No results found with semantic search")
            else:
                semantic_table = []
                for result in semantic_results:
                    # Truncate definition for display
                    definition = result.get('definition', '')
                    if len(definition) > 50:
                        definition = definition[:47] + '...'
                        
                    semantic_table.append([
                        result.get('term', 'N/A'),
                        definition,
                        f"{result.get('similarity_score', 0):.3f}"
                    ])
                
                print(tabulate(
                    semantic_table,
                    headers=['Term', 'Definition', 'Similarity Score'],
                    tablefmt='grid'
                ))

            # 3. Run Fuzzy search
            print("\n3. FUZZY SEARCH RESULTS:")
            fuzzy_results = fuzzy_glossary_search_sync(
                db,
                view_name,
                query,
                fuzzy_threshold=65.0  # Lower threshold to catch fuzzy matches
            )
            
            if not fuzzy_results:
                print("  No results found with fuzzy search")
            else:
                fuzzy_table = []
                for result in fuzzy_results:
                    # Truncate definition for display
                    definition = result.get('definition', '')
                    if len(definition) > 50:
                        definition = definition[:47] + '...'
                        
                    fuzzy_table.append([
                        result.get('term', 'N/A'),
                        definition,
                        f"{result.get('fuzzy_score', 0):.3f}"
                    ])
                
                print(tabulate(
                    fuzzy_table,
                    headers=['Term', 'Definition', 'Fuzzy Score'],
                    tablefmt='grid'
                ))

            # 4. Run Combined search
            print("\n4. COMBINED SEARCH RESULTS:")
            combined_results = combined_semantic_fuzzy_search_sync(
                db,
                collection_name,
                view_name,
                query,
                query_vector,
                fuzzy_threshold=65.0,
                semantic_threshold=0.25
            )
            
            if not combined_results:
                print("  No results found with combined search")
            else:
                combined_table = []
                for result in combined_results:
                    # Truncate definition for display
                    definition = result.get('definition', '')
                    if len(definition) > 50:
                        definition = definition[:47] + '...'
                        
                    combined_table.append([
                        result.get('term', 'N/A'),
                        definition,
                        f"{result.get('combined_score', 0):.3f}"
                    ])
                
                print(tabulate(
                    combined_table,
                    headers=['Term', 'Definition', 'Combined Score'],
                    tablefmt='grid'
                ))

            # Summary of method effectiveness for this query
            print("\nSUMMARY:")
            methods_with_results = []
            if bm25_results: methods_with_results.append("BM25")
            if semantic_results: methods_with_results.append("Semantic")
            if fuzzy_results: methods_with_results.append("Fuzzy")
            if combined_results: methods_with_results.append("Combined")
            
            if len(methods_with_results) == 0:
                print("❌ No search method found results for this query")
            elif len(methods_with_results) < 3:
                print(f"✅ Found results with: {', '.join(methods_with_results)}")
                print(f"❌ No results with: {', '.join(set(['BM25', 'Semantic', 'Fuzzy']) - set(methods_with_results))}")
            else:
                print(f"✅ Most methods found results: {', '.join(methods_with_results)}")
                
            # Highlight if we have a case where one method works when others fail
            if len(methods_with_results) == 1:
                print(f"🔍 IMPORTANT: Only {methods_with_results[0]} search found results!")
            
            print("\n" + "-"*80)

    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    # Run the comparison test
    asyncio.run(run_search_comparison()) 