#!/usr/bin/env python3
"""
Unified Glossary Search System

This module combines multiple search techniques for comprehensive glossary term matching:
1. Embedded term extraction - Finding terms explicitly marked with [TERM: term] syntax
2. Semantic search - Finding terms semantically similar to the query
3. Fuzzy matching - Finding terms with slight variations or misspellings

The combined approach provides excellent coverage for identifying relevant glossary
terms in user questions, whether they're explicitly marked, semantically related,
or fuzzy matches.
"""

import asyncio
import re
import string
from typing import List, Dict, Any, Optional, Union, Tuple
from rapidfuzz import fuzz, process
from loguru import logger

from agent_tools.cursor_rules.core.glossary import format_for_embedding
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

# Pattern for extracting embedded terms in [TERM: term] format
TERM_PATTERN = r'\[TERM:\s*([^\]|]+)(?:\s*\|\s*([^\]]+))?\]'

def preprocess_text(text: str) -> str:
    """
    Remove punctuation and convert to lowercase for better matching.
    
    Args:
        text: The input text to preprocess
        
    Returns:
        Preprocessed text with punctuation removed and converted to lowercase
    """
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

def extract_embedded_terms(text: str) -> List[Dict[str, str]]:
    """
    Extract terms embedded in the text in the format [TERM: term] or [TERM: term | definition].
    
    Args:
        text: Input text containing embedded terms
        
    Returns:
        List of dictionaries with extracted terms and optional definitions
    """
    matches = re.finditer(TERM_PATTERN, text, re.IGNORECASE)
    extracted = []
    
    for match in matches:
        term = match.group(1).strip()
        definition = match.group(2).strip() if match.group(2) else None
        
        if term:
            extracted.append({
                "term": term,
                "definition": definition,
                "original_match": match.group(0),
                "match_type": "embedded"
            })
    
    return extracted

async def fuzzy_search_terms(
    query: str, 
    glossary: List[Dict[str, Any]], 
    threshold: int = 80,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform fuzzy matching to find glossary terms that are similar to the query.
    
    Args:
        query: The search query
        glossary: List of glossary entries
        threshold: Minimum similarity score (0-100) to include a match
        limit: Maximum number of results to return
        
    Returns:
        List of matching glossary entries with similarity scores
    """
    processed_query = preprocess_text(query)
    
    # Prepare choices for fuzzy matching
    choices = []
    for i, entry in enumerate(glossary):
        term = entry.get("term", "")
        processed_term = preprocess_text(term)
        choices.append((i, processed_term))
    
    # Use RapidFuzz for efficient fuzzy matching
    results = []
    
    # Using process.extract for more efficient processing
    matches = process.extract(
        processed_query,
        [choice[1] for choice in choices],
        scorer=fuzz.token_sort_ratio,
        limit=limit * 2  # Get more results initially, then filter by threshold
    )
    
    # Filter and format results
    for matched_text, score, idx in matches:
        if score >= threshold:
            entry_idx = choices[idx][0]
            entry = glossary[entry_idx]
            
            results.append({
                "term": entry.get("term", ""),
                "definition": entry.get("definition", ""),
                "category": entry.get("category", ""),
                "related_terms": entry.get("related_terms", []),
                "similarity_score": score,
                "match_type": "fuzzy"
            })
    
    # Sort by similarity score and limit results
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:limit]

# Synchronous helper function for ArangoDB semantic search
def _perform_semantic_search(
    db, 
    collection_name: str, 
    query_embedding: List[float],
    query_text: str,  # Added original query text for term exactness boosting
    limit: int = 5,
    threshold: float = 0.5,
    exact_boost: float = 0.2  # Boost for exact term matches
) -> List[Dict[str, Any]]:
    """
    Perform semantic search in ArangoDB (synchronous version).
    This implementation balances semantic relatedness with term exactness.
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        query_embedding: The embedding vector for the query
        query_text: Original query text for term exactness boosting
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0-1) to include a match
        exact_boost: Boost factor for exact term matches
        
    Returns:
        List of semantically matching glossary entries
    """
    try:
        # Log the query parameters for debugging
        logger.debug(f"Performing semantic search with: collection={collection_name}, threshold={threshold}, limit={limit}")
        logger.debug(f"Embedding vector length: {len(query_embedding)}")
        
        # Use COSINE_SIMILARITY directly as shown in the example
        aql_query = f"""
        FOR doc IN {collection_name}
            FILTER HAS(doc, "embedding")
            LET similarity = COSINE_SIMILARITY(doc.embedding, @query_embedding)
            
            // Boost score for terms that appear in the query directly
            LET term_lower = LOWER(doc.term)
            LET query_lower = LOWER(@query_text)
            LET exact_match_boost = (
                CONTAINS(query_lower, term_lower) ? @exact_boost : 0
            )
            
            LET adjusted_similarity = similarity + exact_match_boost
            FILTER adjusted_similarity >= @threshold
            
            SORT adjusted_similarity DESC
            LIMIT @limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                similarity_score: adjusted_similarity,
                match_type: "semantic",
                original_score: similarity,
                boost: exact_match_boost
            }}
        """
        
        bind_vars = {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "threshold": threshold,
            "limit": limit,
            "exact_boost": exact_boost
        }
        
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = [doc for doc in cursor]
        
        # Check if we got any results
        if not results:
            logger.debug(f"No semantic search results found with threshold {threshold}. Trying with lower threshold.")
            
            # Try with a lower threshold if no results were found
            lower_threshold = max(0.1, threshold - 0.2)  # Lower the threshold but not below 0.1
            
            # Use the same query with a lower threshold
            bind_vars["threshold"] = lower_threshold
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = [doc for doc in cursor]
            
            if results:
                logger.debug(f"Found {len(results)} results with lower threshold {lower_threshold}")
            else:
                # If still no results, try a keyword-based approach as fallback
                logger.debug("No semantic results found even with lower threshold. Using keyword fallback.")
                
                keyword_fallback_query = f"""
                FOR doc IN {collection_name}
                    LET term_lower = LOWER(doc.term)
                    LET query_lower = LOWER(@query_text)
                    
                    // Calculate keyword match score
                    LET words = SPLIT(query_lower, " ")
                    LET term_words = SPLIT(term_lower, " ")
                    
                    LET matches = (
                        FOR word IN words
                            FILTER LENGTH(word) > 3  // Only consider meaningful words
                            FOR term_word IN term_words
                                FILTER term_word == word OR STARTS_WITH(term_word, word) OR STARTS_WITH(word, term_word)
                                RETURN 1
                    )
                    
                    // Calculate score - divide by number of words, avoid division by zero
                    LET word_count = LENGTH(words)
                    LET score = LENGTH(matches) / (word_count > 0 ? word_count : 1)
                    FILTER score > 0.2  // At least some matches
                    
                    SORT score DESC
                    LIMIT @limit
                    
                    RETURN {{
                        term: doc.term, 
                        definition: doc.definition,
                        category: doc.category,
                        related_terms: doc.related_terms,
                        similarity_score: score,
                        match_type: "keyword_fallback",
                        boost: 0
                    }}
                """
                
                try:
                    cursor = db.aql.execute(keyword_fallback_query, bind_vars={"query_text": query_text, "limit": limit})
                    results = [doc for doc in cursor]
                    logger.debug(f"Found {len(results)} results with keyword fallback")
                except Exception as inner_e:
                    logger.error(f"Keyword fallback error: {inner_e}")
        
        return results
            
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        # Provide more detailed error information to help debugging
        logger.error(f"Error details: {str(e)}")
        return []

async def semantic_search_terms(
    db, 
    collection_name: str,
    query: str,
    limit: int = 5,
    threshold: float = 0.5,
    exact_boost: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Perform semantic search to find glossary terms semantically similar to the query.
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        query: The search query
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0-1) to include a match
        exact_boost: Boost factor for exact term matches
        
    Returns:
        List of semantically matching glossary entries
    """
    try:
        # Generate embedding for the query
        query_embedding_result = generate_embedding(query)
        
        if not query_embedding_result or "embedding" not in query_embedding_result:
            logger.warning("Failed to generate embedding for query")
            return []
        
        query_embedding = query_embedding_result["embedding"]
        
        # Use asyncio.to_thread to run the synchronous ArangoDB operation
        results = await asyncio.to_thread(
            _perform_semantic_search,
            db,
            collection_name,
            query_embedding,
            query,  # Pass the original query for term exactness boosting
            limit,
            threshold,
            exact_boost
        )
        
        return results
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []

async def find_related_terms(
    terms: List[Dict[str, Any]], 
    glossary: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Find related terms based on the provided terms.
    
    Args:
        terms: List of identified terms
        glossary: List of glossary entries
        
    Returns:
        List of related glossary entries
    """
    results = []
    matched_terms = set()
    
    # Keep track of already matched terms to avoid duplicates
    for term_info in terms:
        if "term" in term_info:
            matched_terms.add(term_info["term"].lower())
    
    # For each term, find related terms in the glossary
    for term_info in terms:
        related_term_names = term_info.get("related_terms", [])
        
        for related_term_name in related_term_names:
            related_term_lower = related_term_name.lower()
            
            # Skip if already matched
            if related_term_lower in matched_terms:
                continue
                
            # Find this related term in the glossary
            for entry in glossary:
                if entry["term"].lower() == related_term_lower:
                    results.append({
                        "term": entry["term"],
                        "definition": entry["definition"],
                        "category": entry.get("category", ""),
                        "related_terms": entry.get("related_terms", []),
                        "match_type": "related"
                    })
                    matched_terms.add(related_term_lower)
                    break
    
    return results

# Synchronous helper function to get all glossary entries
def _get_glossary_entries(db, collection_name: str) -> List[Dict[str, Any]]:
    """
    Get all glossary entries from ArangoDB (synchronous version).
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        
    Returns:
        List of glossary entries
    """
    aql_query = f"FOR doc IN {collection_name} RETURN doc"
    cursor = db.aql.execute(aql_query)
    return [doc for doc in cursor]

async def unified_glossary_search(
    db,
    collection_name: str,
    query: str,
    fuzzy_threshold: int = 80,
    semantic_threshold: float = 0.5,
    exact_boost: float = 0.2,  # Added parameter for exact term boost
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform a unified glossary search using multiple techniques.
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        query: The search query
        fuzzy_threshold: Threshold for fuzzy matching (0-100)
        semantic_threshold: Threshold for semantic matching (0-1)
        exact_boost: Boost factor for exact term matches in semantic search (0-1)
        limit: Maximum number of results for each search technique
        
    Returns:
        Dictionary with search results from different techniques and combined results
    """
    # Get the complete glossary for fuzzy and related term searches
    glossary = []
    
    try:
        # Use asyncio.to_thread to run the synchronous ArangoDB operation
        glossary = await asyncio.to_thread(_get_glossary_entries, db, collection_name)
    except Exception as e:
        logger.error(f"Error retrieving glossary: {e}")
        return {
            "query": query,
            "extracted_terms": [],
            "fuzzy_matches": [],
            "semantic_matches": [],
            "related_matches": [],
            "combined_results": []
        }
    
    # 1. Extract embedded terms
    extracted_terms = extract_embedded_terms(query)
    
    # 2. Perform fuzzy search
    fuzzy_matches = await fuzzy_search_terms(
        query, 
        glossary, 
        threshold=fuzzy_threshold,
        limit=limit
    )
    
    # 3. Perform semantic search with term exactness boosting
    semantic_matches = await semantic_search_terms(
        db,
        collection_name,
        query,
        limit=limit,
        threshold=semantic_threshold,
        exact_boost=exact_boost  # Pass the exact_boost parameter
    )
    
    # Combine exact matches from extracted terms and fuzzy matches
    exact_matches = []
    for term_info in extracted_terms:
        term = term_info["term"]
        
        # Look for an exact match in the glossary
        for entry in glossary:
            if entry["term"].lower() == term.lower():
                exact_matches.append({
                    "term": entry["term"],
                    "definition": entry["definition"],
                    "category": entry.get("category", ""),
                    "related_terms": entry.get("related_terms", []),
                    "match_type": "embedded_exact"
                })
                break
    
    # 4. Find related terms based on exact matches
    related_matches = await find_related_terms(exact_matches, glossary)
    
    # 5. Combine all results into a unique set
    all_matches = {}
    
    # Add exact matches first (highest priority)
    for match in exact_matches:
        term = match["term"]
        if term not in all_matches:
            all_matches[term] = match
    
    # Add fuzzy matches
    for match in fuzzy_matches:
        term = match["term"]
        if term not in all_matches:
            all_matches[term] = match
    
    # Add semantic matches
    for match in semantic_matches:
        term = match["term"]
        if term not in all_matches:
            # If it's a semantic match with exact match boost, promote it
            if match.get("boost", 0) > 0:
                match["match_type"] = "semantic_exact"
            all_matches[term] = match
    
    # Add related matches
    for match in related_matches:
        term = match["term"]
        if term not in all_matches:
            all_matches[term] = match
    
    combined_results = list(all_matches.values())
    
    return {
        "query": query,
        "extracted_terms": extracted_terms,
        "fuzzy_matches": fuzzy_matches,
        "semantic_matches": semantic_matches,
        "related_matches": related_matches,
        "combined_results": combined_results
    }

def format_glossary_results(results: Dict[str, Any], format_type: str = "text") -> str:
    """
    Format glossary search results into a human-readable format.
    
    Args:
        results: The search results from unified_glossary_search
        format_type: The output format type ("text", "markdown", or "html")
        
    Returns:
        Formatted string with search results
    """
    query = results["query"]
    combined_results = results["combined_results"]
    extracted_terms = results["extracted_terms"]
    
    if format_type == "markdown":
        output = f"## Glossary Results for: {query}\n\n"
        
        if extracted_terms:
            output += "### Extracted Terms\n\n"
            for term in extracted_terms:
                output += f"- **{term['term']}**"
                if term.get("definition"):
                    output += f": {term['definition']}"
                output += "\n"
            output += "\n"
        
        if combined_results:
            output += "### Matching Terms\n\n"
            for result in combined_results:
                match_type = result["match_type"].replace("_", " ").title()
                output += f"#### {result['term']} ({match_type})\n\n"
                output += f"{result['definition']}\n\n"
                
                if result.get("related_terms"):
                    output += "**Related Terms:** "
                    output += ", ".join(result["related_terms"])
                    output += "\n\n"
        else:
            output += "No matching terms found.\n"
    
    elif format_type == "html":
        output = f"<h2>Glossary Results for: {query}</h2>"
        
        if extracted_terms:
            output += "<h3>Extracted Terms</h3><ul>"
            for term in extracted_terms:
                output += f"<li><strong>{term['term']}</strong>"
                if term.get("definition"):
                    output += f": {term['definition']}"
                output += "</li>"
            output += "</ul>"
        
        if combined_results:
            output += "<h3>Matching Terms</h3>"
            for result in combined_results:
                match_type = result["match_type"].replace("_", " ").title()
                output += f"<h4>{result['term']} ({match_type})</h4>"
                output += f"<p>{result['definition']}</p>"
                
                if result.get("related_terms"):
                    output += "<p><strong>Related Terms:</strong> "
                    output += ", ".join(result["related_terms"])
                    output += "</p>"
        else:
            output += "<p>No matching terms found.</p>"
    
    else:  # default to text
        output = f"Glossary Results for: {query}\n\n"
        
        if extracted_terms:
            output += "Extracted Terms:\n"
            for term in extracted_terms:
                output += f"- {term['term']}"
                if term.get("definition"):
                    output += f": {term['definition']}"
                output += "\n"
            output += "\n"
        
        if combined_results:
            output += "Matching Terms:\n\n"
            for result in combined_results:
                match_type = result["match_type"].replace("_", " ").title()
                output += f"{result['term']} ({match_type}):\n"
                output += f"{result['definition']}\n"
                
                if result.get("related_terms"):
                    output += "Related Terms: "
                    output += ", ".join(result["related_terms"])
                    output += "\n"
                output += "\n"
        else:
            output += "No matching terms found.\n"
    
    return output

async def main():
    """Example usage of the unified glossary search."""
    # Create a connection to ArangoDB
    from arango import ArangoClient
    
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('test_db', username='root', password='openSesame')
    collection_name = 'glossary'
    
    # Example queries
    queries = [
        "How does [TERM: Neural Network] architecture work?",
        "What is the difference between a vector database and traditional databases?",
        "Can you explain how semantic search works for finding similar documents?",
        "What is a Transformer and how does it relate to attention mechanisms?"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        
        results = await unified_glossary_search(
            db,
            collection_name,
            query,
            fuzzy_threshold=80,
            semantic_threshold=0.5,
            exact_boost=0.2,
            limit=5
        )
        
        formatted_results = format_glossary_results(results, format_type="text")
        print(formatted_results)

if __name__ == "__main__":
    asyncio.run(main()) 