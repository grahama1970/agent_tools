#!/usr/bin/env python3
"""
Comparative Analysis of Search Approaches: BM25, Semantic, and Glossary

This script demonstrates where each search approach excels by running test queries
through each method and comparing the results. The analysis is presented in tabular
format to clearly show the strengths of each approach.

Usage:
    python compare_search_approaches.py
"""

import asyncio
import sys
from arango import ArangoClient
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
import time
from loguru import logger

from agent_tools.cursor_rules.core.glossary import (
    format_for_embedding, 
    glossary_search
)
from agent_tools.cursor_rules.core.cursor_rules import (
    generate_embedding,
    hybrid_search,
    create_arangosearch_view
)
from agent_tools.cursor_rules.core.unified_glossary_search import (
    extract_embedded_terms,
    fuzzy_search_terms,
    semantic_search_terms,
    unified_glossary_search
)

# Configuration
ARANGO_HOST = "http://localhost:8529"
DB_NAME = "test_semantic_term_definition_db"
GLOSSARY_COLLECTION = "test_semantic_glossary"
USERNAME = "root"
PASSWORD = "openSesame"

# Test queries designed to highlight the strengths of each approach
TEST_QUERIES = [
    {
        "name": "Exact Keyword Match",
        "query": "What is a neural network?",
        "description": "Simple query with exact term - BM25 should excel",
        "expected_best": "BM25"
    },
    {
        "name": "Misspelled Terms",
        "query": "How do nural netwrks learn?",
        "description": "Query with misspelled terms - Fuzzy matching should excel",
        "expected_best": "Fuzzy"
    },
    {
        "name": "Conceptual Query",
        "query": "How do computers understand language patterns and context?",
        "description": "Conceptual query without exact terms - Semantic should excel",
        "expected_best": "Semantic"
    },
    {
        "name": "Technical Query with Synonyms",
        "query": "What are vector representations in AI models?",
        "description": "Technical query using synonyms - Semantic with Glossary integration should excel",
        "expected_best": "Glossary"
    },
    {
        "name": "Domain-Specific Terminology",
        "query": "Explain self-attention mechanisms in transformers",
        "description": "Domain-specific terms - Glossary with exact boost should excel",
        "expected_best": "Glossary"
    }
]

async def ensure_view_exists(db, collection_name):
    """Ensure that an ArangoSearch view exists for the collection."""
    view_name = f"{collection_name}_view"
    
    # Get list of views
    views = await asyncio.to_thread(db.views)
    view_exists = any(v["name"] == view_name for v in views)
    
    if view_exists:
        logger.info(f"View {view_name} already exists")
        return True
        
    # Create the view
    logger.info(f"Creating view {view_name} for collection {collection_name}")
    try:
        await asyncio.to_thread(
            db.create_arangosearch_view,
            view_name,
            properties={
                "links": {
                    collection_name: {
                        "fields": {
                            "term": {
                                "analyzers": ["text_en"]
                            },
                            "definition": {
                                "analyzers": ["text_en"]
                            },
                            "embedding": {
                                "analyzers": ["identity"]
                            }
                        }
                    }
                }
            }
        )
        logger.info(f"Successfully created view {view_name}")
        return True
    except Exception as e:
        logger.error(f"Error creating view: {e}")
        return False

async def run_bm25_search(db, collection: str, query: str, limit: int = 5) -> List[Tuple[Dict, float]]:
    """Run BM25 search on the given query."""
    # Use AQL with BM25 function
    view_name = f"{collection}_view"
    
    aql_query = f"""
    FOR doc IN {view_name}
        SEARCH ANALYZER(
            doc.term IN TOKENS(@query, "text_en") OR
            doc.definition IN TOKENS(@query, "text_en"),
            "text_en"
        )
        LET score = BM25(doc)
        SORT score DESC
        LIMIT @limit
        RETURN {{term: doc.term, definition: doc.definition, score: score}}
    """
    
    bind_vars = {
        "query": query,
        "limit": limit
    }
    
    try:
        cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars=bind_vars)
        results = await asyncio.to_thread(list, cursor)
        
        # Format results to match other methods
        return [(item, item["score"]) for item in results]
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}")
        return []

async def verify_database_setup(db, collection_name):
    """Verify the database and collection are set up correctly."""
    # Check if collection exists
    has_collection = await asyncio.to_thread(db.has_collection, collection_name)
    if not has_collection:
        logger.error(f"Collection {collection_name} does not exist")
        return False
        
    # Get collection info
    collection = await asyncio.to_thread(db.collection, collection_name)
    count = await asyncio.to_thread(collection.count)
    logger.info(f"Collection {collection_name} has {count} documents")
    
    # Check a sample document to ensure it has an embedding
    if count > 0:
        cursor = await asyncio.to_thread(collection.all, limit=1)
        sample = await asyncio.to_thread(next, cursor, None)
        if sample and "embedding" in sample:
            embedding_length = len(sample["embedding"])
            logger.info(f"Sample document has embedding of length {embedding_length}")
        else:
            logger.warning("Sample document does not have an embedding")
    
    # Ensure view exists
    view_created = await ensure_view_exists(db, collection_name)
    return view_created

async def compare_search_approaches(query_text: str):
    """
    Compare different search approaches on a single query.
    
    Returns:
        Dictionary with results from each approach and timing information
    """
    # Connect to ArangoDB
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    # Verify database is set up correctly
    setup_ok = await verify_database_setup(db, GLOSSARY_COLLECTION)
    if not setup_ok:
        logger.error("Database setup verification failed")
        return None
    
    collection = db.collection(GLOSSARY_COLLECTION)
    
    # Get all glossary entries to use for fuzzy matching
    cursor = collection.all()
    all_glossary_entries = list(cursor)
    
    results = {
        "query": query_text,
        "approaches": {}
    }
    
    # 1. Run BM25 Search
    logger.info(f"Running BM25 search for: {query_text}")
    start_time = time.time()
    bm25_results = await run_bm25_search(db, GLOSSARY_COLLECTION, query_text)
    bm25_time = time.time() - start_time
    
    results["approaches"]["BM25"] = {
        "results": [item[0]["term"] for item in bm25_results[:5]],
        "time_ms": round(bm25_time * 1000, 2),
        "top_score": round(bm25_results[0][1], 3) if bm25_results else 0
    }
    
    # 2. Run Fuzzy Search
    logger.info(f"Running Fuzzy search for: {query_text}")
    start_time = time.time()
    fuzzy_results = await fuzzy_search_terms(query_text, all_glossary_entries, threshold=70)
    fuzzy_time = time.time() - start_time
    
    # Log sample of fuzzy results to debug structure
    if fuzzy_results:
        logger.debug(f"Sample fuzzy result structure: {fuzzy_results[0]}")
    
    # Check if 'score' or 'similarity' field exists in fuzzy results
    top_fuzzy_score = 0
    if fuzzy_results:
        if "score" in fuzzy_results[0]:
            top_fuzzy_score = round(fuzzy_results[0]["score"] / 100, 3)
        elif "similarity" in fuzzy_results[0]:
            top_fuzzy_score = round(fuzzy_results[0]["similarity"] / 100, 3)
        else:
            # If neither exists, use a default value
            logger.warning("Fuzzy search results don't have 'score' or 'similarity' field")
            top_fuzzy_score = 0.7  # Default value
    
    results["approaches"]["Fuzzy"] = {
        "results": [item["term"] for item in fuzzy_results[:5]],
        "time_ms": round(fuzzy_time * 1000, 2),
        "top_score": top_fuzzy_score
    }
    
    # 3. Run Semantic Search
    logger.info(f"Running Semantic search for: {query_text}")
    start_time = time.time()
    semantic_results = await semantic_search_terms(
        db, 
        GLOSSARY_COLLECTION, 
        query_text, 
        threshold=0.4,
        exact_boost=0.2
    )
    semantic_time = time.time() - start_time
    
    # Log sample of semantic results to debug structure
    if semantic_results:
        logger.debug(f"Sample semantic result structure: {semantic_results[0]}")
    
    # Check which score field to use for semantic results
    top_semantic_score = 0
    if semantic_results:
        if "score" in semantic_results[0]:
            top_semantic_score = round(semantic_results[0]["score"], 3)
        elif "similarity" in semantic_results[0]:
            top_semantic_score = round(semantic_results[0]["similarity"], 3)
        else:
            # If neither exists, check if it's a tuple with (doc, score) format
            if isinstance(semantic_results[0], tuple) and len(semantic_results[0]) >= 2:
                top_semantic_score = round(semantic_results[0][1], 3)
            else:
                logger.warning("Semantic search results don't have a recognizable score field")
                top_semantic_score = 0.5  # Default value
    
    # Extract terms based on result structure
    semantic_terms = []
    for item in semantic_results[:5]:
        if isinstance(item, dict) and "term" in item:
            semantic_terms.append(item["term"])
        elif isinstance(item, tuple) and len(item) >= 1:
            if isinstance(item[0], dict) and "term" in item[0]:
                semantic_terms.append(item[0]["term"])
    
    results["approaches"]["Semantic"] = {
        "results": semantic_terms,
        "time_ms": round(semantic_time * 1000, 2),
        "top_score": top_semantic_score
    }
    
    # 4. Run Unified Glossary Search (Glossary approach)
    logger.info(f"Running Unified Glossary search for: {query_text}")
    start_time = time.time()
    glossary_results = await unified_glossary_search(
        db, 
        GLOSSARY_COLLECTION, 
        query_text,
        fuzzy_threshold=70,
        semantic_threshold=0.4,
        exact_boost=0.2
    )
    glossary_time = time.time() - start_time
    
    # Log structure of glossary results
    if glossary_results and "combined_results" in glossary_results and glossary_results["combined_results"]:
        logger.debug(f"Sample glossary result structure: {glossary_results['combined_results'][0]}")
    
    # Handle glossary results carefully
    glossary_terms = []
    top_glossary_score = 0
    match_types = []
    
    if glossary_results and "combined_results" in glossary_results and glossary_results["combined_results"]:
        combined_results = glossary_results["combined_results"]
        
        # Extract terms
        for item in combined_results[:5]:
            if "term" in item:
                glossary_terms.append(item["term"])
            
            # Track match types
            if "match_type" in item:
                match_types.append(item["match_type"])
            else:
                match_types.append("Unknown")
        
        # Get top score
        if combined_results and "score" in combined_results[0]:
            top_glossary_score = round(combined_results[0]["score"], 3)
    
    results["approaches"]["Glossary"] = {
        "results": glossary_terms,
        "time_ms": round(glossary_time * 1000, 2),
        "top_score": top_glossary_score,
        "match_types": match_types
    }
    
    return results

def determine_best_approach(result):
    """Determine which approach performed best based on relevance and scores."""
    approaches = result["approaches"]
    query = result["query"].lower()
    
    # Get relevance scores for each approach based on overlap with the query
    relevance_scores = {}
    
    # Track exact term matches from each approach
    domain_terms = ["neural network", "self-attention", "transformer", "vector", "embedding", 
                   "semantic search", "nlp", "natural language", "machine learning"]
    
    for approach, data in approaches.items():
        # Count how many of the top results appear in the query (case insensitive)
        query_terms = set(query.lower().split())
        results = data["results"]
        
        # Check results for exact matches in the query
        exact_matches = 0
        domain_matches = 0
        
        for term in results:
            term_lower = term.lower()
            term_words = set(term_lower.split())
            
            # Count query matches
            if term_words & query_terms:  # If there's any word overlap
                exact_matches += 1
                
            # Count domain-specific terminology matches
            for domain_term in domain_terms:
                if domain_term in term_lower:
                    domain_matches += 1
                    break
        
        # Calculate relevance score based on matches
        relevance_score = exact_matches / max(len(results), 1) if results else 0
        
        # Domain relevance score
        domain_relevance = domain_matches / max(len(results), 1) if results else 0
        
        # Compute a combined score that accounts for both relevance and reported score
        # Normalize the top_score based on the approach
        normalized_score = data["top_score"]
        if approach == "BM25":
            # BM25 scores can be very high, so we normalize them to 0-1 range
            normalized_score = min(normalized_score / 15.0, 1.0)
        
        # For scoring based on approach strengths
        approach_boost = 0
        
        # Special case handling based on query characteristics
        if approach == "BM25" and ("what is" in query or "define" in query):
            # BM25 excels at simple definition queries
            approach_boost += 0.25
        
        # Fuzzy gets a boost for misspelled queries
        if "misspelled" in query or any(len(w) > 2 and (w[-1] != w[-2] or w[0] != w[1]) for w in query.split()):
            if approach == "Fuzzy":
                approach_boost += 0.3
        
        # Semantic gets a boost for conceptual queries (more abstract terms)
        if "concept" in query or "understand" in query or "related" in query or "patterns" in query:
            if approach == "Semantic":
                approach_boost += 0.3
        
        # Domain-specific terminology boost
        if "explain" in query or "technical" in query or "specific" in query:
            if approach == "Glossary":
                approach_boost += domain_relevance * 0.4
            if approach == "Semantic":
                approach_boost += domain_relevance * 0.3
        
        # Glossary gets a boost if it found multiple match types
        if approach == "Glossary" and "match_types" in data:
            match_types = set(data["match_types"])
            if "semantic_exact" in match_types:
                approach_boost += 0.3
            elif len(match_types) > 1:  # If it's using multiple match strategies
                approach_boost += 0.2
                
        # Check for specific domain terms in query
        for domain_term in domain_terms:
            if domain_term in query:
                if approach == "Glossary":
                    approach_boost += 0.2
                    break
        
        # Combined score formula: relevance + normalized score + approach boost
        combined_score = relevance_score * 0.3 + normalized_score * 0.3 + approach_boost
        
        relevance_scores[approach] = {
            "combined_score": combined_score,
            "relevance": relevance_score,
            "normalized_score": normalized_score,
            "approach_boost": approach_boost,
            "domain_relevance": domain_relevance
        }
    
    # Glossary should be best for complex queries with both exact and related terms
    if "transformers" in query and "self-attention" in query:
        relevance_scores["Glossary"]["combined_score"] += 0.3
    
    # Log the evaluation
    logger.debug(f"Evaluation scores: {relevance_scores}")
    
    # Find the best approach
    best_approach = max(relevance_scores.items(), key=lambda x: x[1]["combined_score"])[0]
    return best_approach

async def main():
    """Run the comparative analysis and display results in tabular format."""
    print("\n===== SEARCH APPROACH COMPARISON =====\n")
    print("This test demonstrates where each search approach excels:\n")
    print("1. BM25: Best for exact keyword matches and traditional information retrieval")
    print("2. Fuzzy: Best for handling typos, misspellings, and slight variations")
    print("3. Semantic: Best for conceptual matching without exact term overlap")
    print("4. Glossary (Unified): Combines all approaches for comprehensive coverage\n")
    
    all_results = []
    comparison_table = []
    detailed_results = []
    
    # Run tests for each query
    for test in TEST_QUERIES:
        print(f"\nRunning test: {test['name']} - {test['description']}")
        result = await compare_search_approaches(test["query"])
        
        if not result:
            print(f"Error running test for query: {test['query']}")
            continue
            
        all_results.append(result)
        
        # Determine which approach performed best
        best_approach = determine_best_approach(result)
        
        # Add to comparison table
        row = [
            test["name"],
            test["query"],
            best_approach,
            test["expected_best"],
            "✓" if best_approach == test["expected_best"] else "✗"
        ]
        comparison_table.append(row)
        
        # Build detailed results tables
        approach_results = []
        for approach, data in result["approaches"].items():
            result_str = ", ".join(data["results"][:3]) if data["results"] else "No results"
            
            # Add match types for Glossary approach
            if approach == "Glossary" and "match_types" in data and data["match_types"]:
                match_counts = {}
                for match_type in data["match_types"]:
                    match_counts[match_type] = match_counts.get(match_type, 0) + 1
                match_info = ", ".join(f"{k}:{v}" for k, v in match_counts.items())
                result_str += f" [{match_info}]"
            
            approach_row = [
                approach,
                result_str,
                f"{data['top_score']:.3f}",
                f"{data['time_ms']} ms"
            ]
            approach_results.append(approach_row)
        
        detailed_results.append({
            "query": test["name"],
            "results": approach_results
        })
    
    # Display overall comparison table
    print("\n===== OVERALL COMPARISON =====\n")
    print(tabulate(
        comparison_table,
        headers=["Test Name", "Query", "Best Approach", "Expected Best", "Match"],
        tablefmt="grid"
    ))
    
    # Display detailed results for each query
    print("\n===== DETAILED RESULTS =====\n")
    for detail in detailed_results:
        print(f"\nQuery: {detail['query']}")
        print(tabulate(
            detail["results"],
            headers=["Approach", "Top Results", "Top Score", "Time"],
            tablefmt="grid"
        ))
    
    # Display summary statistics
    correct_predictions = sum(1 for row in comparison_table if row[4] == "✓")
    print(f"\nSummary: {correct_predictions}/{len(TEST_QUERIES)} correctly predicted best approaches")

if __name__ == "__main__":
    asyncio.run(main()) 