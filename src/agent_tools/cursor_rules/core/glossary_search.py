#!/usr/bin/env python3
"""
Enhanced glossary search functionality for cursor_rules.

This module provides optimized glossary search functions that follow best practices
for ArangoDB integration with async code.

Documentation references:
- ArangoDB Search: https://www.arangodb.com/docs/stable/arangosearch.html
- ArangoDB AQL PHRASE: https://www.arangodb.com/docs/stable/aql/functions-arangosearch.html#phrase
- asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
"""

import asyncio
import re
from typing import Any, Dict, List, Union, Optional, Tuple, Set
from loguru import logger
import rapidfuzz
from rapidfuzz import fuzz, process

# Import embedding utilities (assuming similar structure as in the cursor_rules)
# Note: Update this import based on your actual embedding generation function
try:
    from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
except ImportError:
    logger.warning("Could not import generate_embedding function, semantic search may not work properly")
    
    # Fallback function if the real one isn't available
    def generate_embedding(text):
        """Fallback function that returns a mock embedding."""
        logger.warning("Using mock embedding generation - semantic search will not work properly")
        return {"embedding": [0.1] * 10, "metadata": {"model": "mock"}}

# Synchronous database functions (no async here)
def search_glossary_terms_sync(
    db, view_name: str, glossary_terms: List[str], limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Synchronous function to perform a glossary search using PHRASE for exact matching.

    Args:
        db: ArangoDB connection instance.
        view_name: The ArangoSearch view name.
        glossary_terms: List of terms to search for.
        limit: The maximum number of results to return.

    Returns:
        List of matched glossary terms with metadata.
    """
    # Input validation
    if not glossary_terms:
        logger.info("No glossary terms provided for search.")
        return []
    
    try:
        # Using PHRASE for exact term matching as per ArangoDB documentation
        query = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(PHRASE(doc.term, @glossary_terms), "text_en")
            SORT BM25(doc) DESC
            LIMIT @limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                metatags: doc.metatags,
                score: BM25(doc)
            }}
        """
        
        # Execute query synchronously
        cursor = db.aql.execute(query, bind_vars={"glossary_terms": glossary_terms, "limit": limit})
        results = list(cursor)
        logger.info(f"Found {len(results)} matches for glossary terms: {', '.join(glossary_terms)}.")
        return results
    except Exception as e:
        logger.error(f"Glossary search failed for terms '{', '.join(glossary_terms)}': {e}")
        return []

def apply_filter_pattern_sync(items: List[Dict[str, Any]], filter_pattern: str) -> List[Dict[str, Any]]:
    """
    Synchronous function to filter items based on a regex pattern.
    
    Args:
        items: List of glossary items.
        filter_pattern: Regex pattern to filter items.
        
    Returns:
        Filtered list of glossary items.
    """
    if not filter_pattern or not items:
        return items

    try:
        pattern = re.compile(filter_pattern)
        filtered_items = [item for item in items if pattern.match(item["term"])]
        logger.info(f"Applied filter pattern: {filter_pattern}. Remaining items: {len(filtered_items)}.")
        return filtered_items
    except re.error as e:
        logger.error(f"Invalid regex pattern '{filter_pattern}': {e}")
        return items

def semantic_glossary_search_sync(
    db, collection_name: str, query_vector: List[float], limit: int = 10, threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Synchronous function to perform semantic search on the glossary using vector similarity.
    
    Args:
        db: ArangoDB connection instance.
        collection_name: The glossary collection name.
        query_vector: The embedding vector for the search query.
        limit: Maximum number of results to return.
        threshold: Minimum similarity threshold.
        
    Returns:
        List of glossary terms with similarity scores.
    """
    try:
        query = f"""
        FOR doc IN {collection_name}
            LET similarity = COSINE_SIMILARITY(doc.embedding, @query_vector)
            FILTER similarity >= @threshold
            SORT similarity DESC
            LIMIT @limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                score: similarity
            }}
        """
        
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "query_vector": query_vector,
                "threshold": threshold,
                "limit": limit
            }
        )
        
        results = list(cursor)
        logger.info(f"Semantic search found {len(results)} glossary terms with similarity >= {threshold}.")
        return results
    except Exception as e:
        logger.error(f"Semantic glossary search failed: {e}")
        return []

def hybrid_glossary_search_sync(
    db, 
    collection_name: str, 
    view_name: str,
    search_text: str,
    query_vector: List[float],
    limit: int = 10, 
    bm25_threshold: float = 0.1,
    semantic_threshold: float = 0.5,
    hybrid_threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Synchronous function to perform hybrid search combining BM25 and vector similarity.
    
    Args:
        db: ArangoDB connection instance.
        collection_name: The glossary collection name.
        view_name: The ArangoSearch view name.
        search_text: The text query for BM25 search.
        query_vector: The embedding vector for semantic search.
        limit: Maximum number of results to return.
        bm25_threshold: Minimum BM25 score threshold.
        semantic_threshold: Minimum vector similarity threshold.
        hybrid_threshold: Minimum combined score threshold.
        
    Returns:
        List of glossary terms with combined scores.
    """
    try:
        # AQL query combining embedding similarity and BM25 as per ArangoDB documentation
        query = f"""
        LET embedding_results = (
            FOR term IN {collection_name}
                LET similarity = COSINE_SIMILARITY(term.embedding, @query_vector)
                FILTER similarity >= @semantic_threshold
                SORT similarity DESC
                LIMIT @limit
                RETURN {{
                    term: term,
                    _key: term._key,
                    similarity_score: similarity,
                    bm25_score: 0
                }}
        )
        
        LET bm25_results = (
            FOR term IN {view_name}
                SEARCH ANALYZER(
                    term.term IN TOKENS(@search_text, "text_en") OR
                    term.definition IN TOKENS(@search_text, "text_en"),
                    "text_en"
                )
                LET bm25_score = BM25(term, 1.2, 0.75)
                FILTER bm25_score > @bm25_threshold
                SORT bm25_score DESC
                LIMIT @limit
                RETURN {
                    term: term,
                    _key: term._key,
                    similarity_score: 0,
                    bm25_score: bm25_score
                }
        )
        
        // Merge and deduplicate results
        LET merged_results = (
            FOR result IN UNION_DISTINCT(embedding_results, bm25_results)
                COLLECT key = result._key INTO group
                LET term_doc = FIRST(group[*].result.term)
                LET similarity_score = MAX(group[*].result.similarity_score)
                LET bm25_score = MAX(group[*].result.bm25_score)
                LET hybrid_score = (similarity_score * 0.5) + (bm25_score * 0.5)
                FILTER hybrid_score >= @hybrid_threshold
                RETURN {
                    "term": term_doc.term,
                    "definition": term_doc.definition,
                    "category": term_doc.category,
                    "related_terms": term_doc.related_terms,
                    "source": term_doc.source,
                    "_key": key,
                    "similarity_score": similarity_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score
                }
        )
        
        // Sort and limit final results
        FOR result IN merged_results
            SORT result.hybrid_score DESC
            LIMIT @limit
            RETURN result
        """
        
        # Execute query synchronously with all parameters
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "search_text": search_text,
                "query_vector": query_vector,
                "semantic_threshold": semantic_threshold,
                "bm25_threshold": bm25_threshold,
                "hybrid_threshold": hybrid_threshold,
                "limit": limit
            }
        )
        
        results = list(cursor)
        logger.info(f"Hybrid search found {len(results)} glossary terms.")
        return results
    except Exception as e:
        logger.error(f"Hybrid glossary search failed: {e}")
        import traceback
        traceback.print_exc()
        return []

# Async wrapper functions
async def search_glossary_terms(
    db, view_name: str, glossary_terms: Union[str, List[str]], limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Async wrapper to perform a glossary search using PHRASE for exact matching.
    
    Always uses asyncio.to_thread as per best practices for ArangoDB integration.

    Args:
        db: ArangoDB connection instance.
        view_name: The ArangoSearch view name.
        glossary_terms: String or list of terms to search for.
        limit: The maximum number of results to return.

    Returns:
        List of matched glossary terms with metadata.
    """
    # Ensure glossary_terms is a list
    if isinstance(glossary_terms, str):
        glossary_terms = [glossary_terms]
    elif not isinstance(glossary_terms, list):
        raise ValueError("glossary_terms must be a string or a list of strings.")
    
    # Delegate to the synchronous function using asyncio.to_thread
    try:
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            search_glossary_terms_sync, 
            db, 
            view_name, 
            glossary_terms, 
            limit
        )
    except Exception as e:
        logger.error(f"Async glossary search failed: {e}")
        return []

async def apply_filter_pattern(items: List[Dict[str, Any]], filter_pattern: str) -> List[Dict[str, Any]]:
    """
    Async wrapper to filter items based on a regex pattern.
    
    Args:
        items: List of glossary items.
        filter_pattern: Regex pattern to filter items.
        
    Returns:
        Filtered list of glossary items.
    """
    # This operation is not database-related, but we'll keep the async interface consistent
    # and use to_thread for any potentially CPU-bound operations
    try:
        return await asyncio.to_thread(apply_filter_pattern_sync, items, filter_pattern)
    except Exception as e:
        logger.error(f"Async filter application failed: {e}")
        return items

async def semantic_glossary_search(
    db, collection_name: str, query_vector: List[float], limit: int = 10, threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Async wrapper to perform semantic search on the glossary using vector similarity.
    
    Args:
        db: ArangoDB connection instance.
        collection_name: The glossary collection name.
        query_vector: The embedding vector for the search query.
        limit: Maximum number of results to return.
        threshold: Minimum similarity threshold.
        
    Returns:
        List of glossary terms with similarity scores.
    """
    try:
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            semantic_glossary_search_sync,
            db,
            collection_name,
            query_vector,
            limit,
            threshold
        )
    except Exception as e:
        logger.error(f"Async semantic glossary search failed: {e}")
        return []

async def hybrid_glossary_search(
    db, 
    collection_name: str, 
    view_name: str,
    search_text: str,
    query_vector: List[float],
    limit: int = 10, 
    bm25_threshold: float = 0.1,
    semantic_threshold: float = 0.5,
    hybrid_threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Async wrapper to perform hybrid search combining BM25 and vector similarity.
    
    Args:
        db: ArangoDB connection instance.
        collection_name: The glossary collection name.
        view_name: The ArangoSearch view name.
        search_text: The text query for BM25 search.
        query_vector: The embedding vector for semantic search.
        limit: Maximum number of results to return.
        bm25_threshold: Minimum BM25 score threshold.
        semantic_threshold: Minimum vector similarity threshold.
        hybrid_threshold: Minimum combined score threshold.
        
    Returns:
        List of glossary terms with combined scores.
    """
    try:
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            hybrid_glossary_search_sync,
            db,
            collection_name,
            view_name,
            search_text,
            query_vector,
            limit,
            bm25_threshold,
            semantic_threshold,
            hybrid_threshold
        )
    except Exception as e:
        logger.error(f"Async hybrid glossary search failed: {e}")
        return []

# Utility functions
def extract_terms_from_text(text: str, min_length: int = 3) -> List[str]:
    """
    Extract potential glossary terms from text.
    
    Args:
        text: The text to extract terms from.
        min_length: Minimum term length to consider.
        
    Returns:
        List of potential glossary terms.
    """
    if not text:
        return []
        
    # Basic term extraction - this could be enhanced with NLP techniques
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-_]*[A-Za-z0-9]\b', text)
    
    # Extract potential multi-word terms (capitalized phrases)
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    
    # Combine and filter by length
    terms = [w for w in words if len(w) >= min_length] + phrases
    
    # Remove duplicates while preserving order
    unique_terms = []
    seen = set()
    for term in terms:
        if term.lower() not in seen:
            unique_terms.append(term)
            seen.add(term.lower())
            
    return unique_terms

async def auto_identify_glossary_terms(
    db, view_name: str, text: str, limit: int = 5, threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Automatically identify potential glossary terms in a text and search for them.
    
    Args:
        db: ArangoDB connection instance.
        view_name: The ArangoSearch view name.
        text: The text to analyze for glossary terms.
        limit: Maximum number of glossary terms to return.
        threshold: Score threshold for term matches.
        
    Returns:
        List of matched glossary terms.
    """
    # Extract potential terms from the text
    potential_terms = extract_terms_from_text(text)
    
    if not potential_terms:
        logger.info("No potential glossary terms found in the text.")
        return []
    
    logger.info(f"Extracted {len(potential_terms)} potential terms from text.")
    
    # Search for the terms in the glossary
    results = await search_glossary_terms(db, view_name, potential_terms, limit=limit)
    
    # Filter by a minimum score threshold if results have a score field
    if results and "score" in results[0]:
        filtered_results = [r for r in results if r.get("score", 0) >= threshold]
        logger.info(f"Found {len(filtered_results)} glossary terms above threshold {threshold}.")
        return filtered_results
    
    return results

# Usage example function
async def usage_example(db, query: str):
    """
    Example demonstrating how to use the glossary search functions.
    
    Args:
        db: ArangoDB connection instance.
        query: The search query.
    """
    # Example 1: Direct term search
    logger.info("Example 1: Direct term search")
    terms = ["Azure", "Teams", "bugs"]
    results = await search_glossary_terms(db, "microsoft_search_view", terms)
    logger.info(f"Found {len(results)} direct matches.")
    
    # Example 2: Filtered search
    logger.info("Example 2: Filtered search")
    filtered_results = await apply_filter_pattern(results, r"^A.*")
    logger.info(f"After filtering, {len(filtered_results)} results remain.")
    
    # Example 3: Auto-identification of terms
    logger.info("Example 3: Auto-identification of terms")
    text = "I want to know about Azure and Teams, but I hate all the bugs in Teams"
    auto_results = await auto_identify_glossary_terms(db, "microsoft_search_view", text)
    logger.info(f"Auto-identified {len(auto_results)} glossary terms.")
    
    return {
        "direct_search": results,
        "filtered_search": filtered_results,
        "auto_identified": auto_results
    } 

def fuzzy_glossary_search_sync(
    db, 
    view_name: str, 
    search_term: str, 
    limit: int = 10, 
    fuzzy_threshold: float = 80.0, 
    initial_limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Synchronous function for two-stage fuzzy search:
    1. Get an initial broad set of results from ArangoDB
    2. Apply RapidFuzz for more precise fuzzy matching and ranking
    
    Args:
        db: ArangoDB connection
        view_name: View to search
        search_term: Term to search for
        limit: Maximum final results
        fuzzy_threshold: RapidFuzz similarity threshold (0-100)
        initial_limit: How many results to pull from ArangoDB initially
        
    Returns:
        List of glossary terms ordered by fuzzy match score
    """
    try:
        # Stage 1: Wider search in ArangoDB using IN TOKENS for broader matches
        query = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(
                doc.term IN TOKENS(@search_term, "text_en") OR
                doc.definition IN TOKENS(@search_term, "text_en"),
                "text_en"
            )
            SORT BM25(doc) DESC
            LIMIT @initial_limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                metatags: doc.metatags,
                score: BM25(doc)
            }}
        """
        
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "search_term": search_term,
                "initial_limit": initial_limit
            }
        )
        
        initial_results = list(cursor)
        logger.info(f"Initial ArangoDB search found {len(initial_results)} potential matches")
        
        if not initial_results:
            return []
        
        # Stage 2: Apply RapidFuzz for better fuzzy matching
        fuzzy_results = []
        for item in initial_results:
            term = item.get("term", "")
            
            # Use different fuzzy matching strategies and take the best score
            token_ratio = fuzz.token_sort_ratio(search_term.lower(), term.lower())
            partial_ratio = fuzz.partial_ratio(search_term.lower(), term.lower())
            fuzz_ratio = fuzz.ratio(search_term.lower(), term.lower())
            
            # Take the best score among different strategies
            best_score = max(token_ratio, partial_ratio, fuzz_ratio)
            
            if best_score >= fuzzy_threshold:
                # Add fuzzy score to result
                item["fuzzy_score"] = best_score
                fuzzy_results.append(item)
        
        # Sort by fuzzy score and limit results
        fuzzy_results.sort(key=lambda x: x.get("fuzzy_score", 0), reverse=True)
        limited_results = fuzzy_results[:limit]
        
        logger.info(f"RapidFuzz filtering identified {len(limited_results)} glossary terms with similarity >= {fuzzy_threshold}")
        return limited_results
        
    except Exception as e:
        logger.error(f"Fuzzy glossary search failed: {e}")
        return []

def combined_semantic_fuzzy_search_sync(
    db,
    collection_name: str,
    view_name: str,
    query_text: str,
    query_vector: List[float],
    limit: int = 10,
    fuzzy_threshold: float = 70.0,
    semantic_threshold: float = 0.5,
    initial_limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Synchronous function that combines semantic search and fuzzy matching for optimal results.
    
    This approach:
    1. Gets semantic search results based on vector similarity
    2. Gets fuzzy search results based on text matching
    3. Combines them with appropriate weighting
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        view_name: Name of the search view
        query_text: The text query for fuzzy/text search
        query_vector: The embedding vector for semantic search
        limit: Maximum number of final results
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        semantic_threshold: Minimum semantic similarity score (0-1)
        initial_limit: How many results to pull initially from each method
        
    Returns:
        Merged and ranked list of glossary terms
    """
    try:
        # Get semantic results
        semantic_query = f"""
        FOR doc IN {collection_name}
            LET similarity = COSINE_SIMILARITY(doc.embedding, @query_vector)
            FILTER similarity >= @semantic_threshold
            SORT similarity DESC
            LIMIT @initial_limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                metatags: doc.metatags,
                _key: doc._key,
                similarity_score: similarity
            }}
        """
        
        semantic_cursor = db.aql.execute(
            semantic_query,
            bind_vars={
                "query_vector": query_vector,
                "semantic_threshold": semantic_threshold,
                "initial_limit": initial_limit
            }
        )
        
        semantic_results = list(semantic_cursor)
        logger.info(f"Semantic search found {len(semantic_results)} matches")
        
        # Get fuzzy text matches
        text_query = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(
                doc.term IN TOKENS(@query_text, "text_en") OR
                doc.definition IN TOKENS(@query_text, "text_en"),
                "text_en"
            )
            SORT BM25(doc) DESC
            LIMIT @initial_limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                metatags: doc.metatags,
                _key: doc._key,
                bm25_score: BM25(doc)
            }}
        """
        
        text_cursor = db.aql.execute(
            text_query,
            bind_vars={
                "query_text": query_text,
                "initial_limit": initial_limit
            }
        )
        
        text_results = list(text_cursor)
        logger.info(f"Text search found {len(text_results)} matches")
        
        # Merge results by _key and calculate fuzzy scores
        results_by_key = {}
        
        # Process semantic results
        for item in semantic_results:
            key = item.get("_key")
            results_by_key[key] = {
                **item,
                "semantic_score": item.get("similarity_score", 0),
                "bm25_score": 0,
                "fuzzy_score": 0
            }
        
        # Process text results and merge with semantic results
        for item in text_results:
            key = item.get("_key")
            term = item.get("term", "")
            
            # Calculate fuzzy match score
            token_ratio = fuzz.token_sort_ratio(query_text.lower(), term.lower())
            partial_ratio = fuzz.partial_ratio(query_text.lower(), term.lower())
            fuzz_ratio = fuzz.ratio(query_text.lower(), term.lower())
            best_fuzzy = max(token_ratio, partial_ratio, fuzz_ratio) / 100.0  # Normalize to 0-1
            
            if key in results_by_key:
                # Update existing entry
                results_by_key[key]["bm25_score"] = item.get("bm25_score", 0)
                results_by_key[key]["fuzzy_score"] = best_fuzzy
            else:
                # Add new entry
                results_by_key[key] = {
                    **item,
                    "semantic_score": 0,
                    "bm25_score": item.get("bm25_score", 0),
                    "fuzzy_score": best_fuzzy
                }
        
        # Filter by fuzzy threshold and calculate combined score
        combined_results = []
        for key, item in results_by_key.items():
            # Skip items below the fuzzy threshold unless they have a good semantic score
            if item["fuzzy_score"] < (fuzzy_threshold / 100.0) and item["semantic_score"] < semantic_threshold:
                continue
                
            # Calculate combined score with weights
            # - Semantic matching is weighted highest (0.5) because it captures meaning best
            # - Fuzzy matching next (0.3) because it handles spelling/variant forms well
            # - BM25 lowest (0.2) as it's the most basic keyword matching
            combined_score = (
                (item["semantic_score"] * 0.5) + 
                (item["fuzzy_score"] * 0.3) + 
                (item["bm25_score"] * 0.2)
            )
            
            # Add combined score
            item["combined_score"] = combined_score
            combined_results.append(item)
        
        # Sort by combined score and limit results
        combined_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        limited_results = combined_results[:limit]
        
        logger.info(f"Combined search returned {len(limited_results)} results")
        return limited_results
        
    except Exception as e:
        logger.error(f"Combined semantic fuzzy search failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def semantic_term_definition_search_sync(
    db,
    collection_name: str,
    user_question: str,
    limit: int = 5,
    semantic_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Search specifically to find when a user's question is semantically similar to term+definition pairs.
    
    This function:
    1. Generates an embedding for the user's question
    2. Compares it against term+definition embeddings in the database
    
    Args:
        db: ArangoDB connection
        collection_name: Glossary collection name
        user_question: The user's question text
        limit: Maximum number of results to return
        semantic_threshold: Minimum semantic similarity threshold
        
    Returns:
        List of glossary terms ordered by semantic similarity to the question
    """
    try:
        # Generate embedding for the user's question
        question_embedding = generate_embedding(user_question)
        
        if not question_embedding or "embedding" not in question_embedding:
            logger.error("Failed to generate embedding for user question")
            return []
            
        query_vector = question_embedding["embedding"]
        
        # First use a very low threshold to get potential matches for further processing
        initial_threshold = max(0.1, semantic_threshold * 0.5)  # Use a lower initial threshold to catch more candidates
        
        # Query for terms where the term+definition embedding is similar to the question
        query = f"""
        FOR doc IN {collection_name}
            // Compare user question embedding against term+definition embedding
            LET similarity = COSINE_SIMILARITY(doc.embedding, @query_vector)
            FILTER similarity >= @initial_threshold
            SORT similarity DESC
            LIMIT @expanded_limit
            RETURN {{
                term: doc.term,
                definition: doc.definition,
                category: doc.category,
                related_terms: doc.related_terms,
                source: doc.source,
                embedding: doc.embedding,
                similarity_score: similarity
            }}
        """
        
        cursor = db.aql.execute(
            query,
            bind_vars={
                "query_vector": query_vector,
                "initial_threshold": initial_threshold,
                "expanded_limit": limit * 3  # Get more candidates initially
            }
        )
        
        candidates = list(cursor)
        
        if not candidates:
            logger.info(f"Semantic term+definition search found 0 matches for user question")
            return []
            
        # Post-process to increase semantic matching relevance
        # This helps with typos and conceptual matches
        enhanced_results = []
        
        for candidate in candidates:
            term = candidate["term"]
            definition = candidate["definition"]
            base_score = candidate["similarity_score"]
            
            # Check if term or definition contains parts of the user question
            # even with typos (using basic fuzzy matching)
            user_words = set(word.lower() for word in re.findall(r'\b\w+\b', user_question))
            term_words = set(word.lower() for word in re.findall(r'\b\w+\b', term))
            def_words = set(word.lower() for word in re.findall(r'\b\w+\b', definition))
            
            # Calculate word overlap ratios
            term_overlap = len(user_words.intersection(term_words)) / max(1, len(term_words))
            def_overlap = len(user_words.intersection(def_words)) / max(1, len(def_words))
            
            # Boost score based on word overlap
            boost_factor = 1.0 + (term_overlap * 0.3) + (def_overlap * 0.2)
            adjusted_score = min(1.0, base_score * boost_factor)
            
            # Check if the adjusted score meets the threshold
            if adjusted_score >= semantic_threshold:
                candidate["similarity_score"] = adjusted_score
                enhanced_results.append(candidate)
        
        # Sort by adjusted score
        enhanced_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Limit to requested number
        results = enhanced_results[:limit]
        
        logger.info(f"Semantic term+definition search found {len(results)} matches for user question")
        return results
        
    except Exception as e:
        logger.error(f"Semantic term+definition search failed: {e}")
        return []

# Add these new async wrapper functions 

async def fuzzy_glossary_search(
    db, 
    view_name: str, 
    search_term: str, 
    limit: int = 10, 
    fuzzy_threshold: float = 80.0, 
    initial_limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Async wrapper for two-stage fuzzy search using RapidFuzz.
    
    Args:
        db: ArangoDB connection
        view_name: View to search
        search_term: Term to search for
        limit: Maximum final results
        fuzzy_threshold: RapidFuzz similarity threshold (0-100)
        initial_limit: How many results to pull from ArangoDB initially
        
    Returns:
        List of glossary terms ordered by fuzzy match score
    """
    try:
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            fuzzy_glossary_search_sync,
            db,
            view_name,
            search_term,
            limit,
            fuzzy_threshold,
            initial_limit
        )
    except Exception as e:
        logger.error(f"Async fuzzy glossary search failed: {e}")
        return []

async def combined_semantic_fuzzy_search(
    db,
    collection_name: str,
    view_name: str,
    query_text: str,
    query_vector: Optional[List[float]] = None,
    limit: int = 10,
    fuzzy_threshold: float = 70.0,
    semantic_threshold: float = 0.5,
    initial_limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Async wrapper for combined semantic and fuzzy search.
    
    This function handles generating the query vector if one is not provided.
    
    Args:
        db: ArangoDB connection
        collection_name: Name of the glossary collection
        view_name: Name of the search view
        query_text: The text query for fuzzy/text search
        query_vector: Optional pre-generated embedding vector
        limit: Maximum number of final results
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        semantic_threshold: Minimum semantic similarity score (0-1)
        initial_limit: How many results to pull initially from each method
        
    Returns:
        Merged and ranked list of glossary terms
    """
    try:
        # Generate embedding if not provided
        if query_vector is None:
            embedding_result = generate_embedding(query_text)
            if embedding_result and "embedding" in embedding_result:
                query_vector = embedding_result["embedding"]
            else:
                logger.warning("Could not generate embedding for query, semantic search will be limited")
                query_vector = [0.1] * 10  # Mock vector as fallback
        
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            combined_semantic_fuzzy_search_sync,
            db,
            collection_name,
            view_name,
            query_text,
            query_vector,
            limit,
            fuzzy_threshold,
            semantic_threshold,
            initial_limit
        )
    except Exception as e:
        logger.error(f"Async combined semantic fuzzy search failed: {e}")
        return []

async def semantic_term_definition_search(
    db,
    collection_name: str,
    user_question: str,
    limit: int = 5,
    semantic_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Async wrapper for searching when a user's question is semantically similar to term+definition pairs.
    
    Args:
        db: ArangoDB connection
        collection_name: Glossary collection name
        user_question: The user's question text
        limit: Maximum number of results to return
        semantic_threshold: Minimum semantic similarity threshold
        
    Returns:
        List of glossary terms ordered by semantic similarity to the question
    """
    try:
        # ALWAYS await asyncio.to_thread for ArangoDB operations
        return await asyncio.to_thread(
            semantic_term_definition_search_sync,
            db,
            collection_name,
            user_question,
            limit,
            semantic_threshold
        )
    except Exception as e:
        logger.error(f"Async semantic term+definition search failed: {e}")
        return []

# Enhanced utility function to understand user queries
async def analyze_user_query(
    db,
    collection_name: str,
    view_name: str,
    user_query: str,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive analysis of a user query that combines all search methods.
    
    This function:
    1. Checks if the query as a whole is semantically similar to term+definition pairs
    2. Extracts potential terms from the query
    3. Performs fuzzy search on those terms
    4. Combines the results for the most comprehensive understanding
    
    Args:
        db: ArangoDB connection
        collection_name: Glossary collection name
        view_name: View name for text search
        user_query: The user's question or search query
        limit: Maximum number of results per method
        
    Returns:
        Dictionary with results from different search methods and combined analysis
    """
    # Execute searches in parallel for efficiency
    semantic_task = asyncio.create_task(
        semantic_term_definition_search(db, collection_name, user_query, limit)
    )
    
    # Extract potential terms
    potential_terms = extract_terms_from_text(user_query)
    
    # Create tasks for fuzzy search on each term
    fuzzy_tasks = []
    for term in potential_terms[:5]:  # Limit to 5 terms for performance
        fuzzy_tasks.append(
            asyncio.create_task(
                fuzzy_glossary_search(db, view_name, term, limit=3, fuzzy_threshold=70.0)
            )
        )
    
    # Create task for combined search on the full query
    embedding_result = generate_embedding(user_query)
    query_vector = embedding_result.get("embedding") if embedding_result else None
    
    combined_task = asyncio.create_task(
        combined_semantic_fuzzy_search(
            db, collection_name, view_name, user_query, query_vector, limit=limit
        )
    )
    
    # Wait for all tasks to complete
    semantic_results = await semantic_task
    fuzzy_results_list = [await task for task in fuzzy_tasks]
    combined_results = await combined_task
    
    # Process fuzzy results by term
    term_results = {}
    for i, term in enumerate(potential_terms[:5]):
        if i < len(fuzzy_results_list):
            term_results[term] = fuzzy_results_list[i]
    
    # Combine all results for final analysis
    all_results = {}
    
    # Track seen keys to avoid duplication
    seen_keys = set()
    
    # Process in order of importance: combined, semantic, then term-specific
    for item in combined_results:
        key = item.get("_key", "")
        if key and key not in seen_keys:
            all_results[key] = item
            seen_keys.add(key)
    
    for item in semantic_results:
        key = item.get("_key", "")
        if key and key not in seen_keys:
            all_results[key] = item
            seen_keys.add(key)
    
    for term, results in term_results.items():
        for item in results:
            key = item.get("_key", "")
            if key and key not in seen_keys:
                all_results[key] = item
                seen_keys.add(key)
    
    # Convert to list and sort by combined score if available
    final_results = list(all_results.values())
    final_results.sort(
        key=lambda x: (
            x.get("combined_score", 0), 
            x.get("similarity_score", 0),
            x.get("fuzzy_score", 0),
            x.get("score", 0)
        ),
        reverse=True
    )
    
    return {
        "semantic_results": semantic_results,
        "term_results": term_results,
        "combined_results": combined_results,
        "all_results": final_results[:limit],
        "extracted_terms": potential_terms
    }

# Example of how to use these functions
async def enhanced_search_example(db, user_query):
    """
    Example demonstrating the enhanced search capabilities.
    
    Args:
        db: ArangoDB connection
        user_query: User's query text
    """
    # Set up collection and view names
    collection_name = "glossary"
    view_name = f"{collection_name}_view"
    
    logger.info(f"Analyzing user query: '{user_query}'")
    
    # Option 1: Full analysis of the query
    analysis = await analyze_user_query(
        db, collection_name, view_name, user_query
    )
    
    logger.info(f"Found {len(analysis['all_results'])} relevant glossary terms")
    logger.info(f"Extracted terms: {analysis['extracted_terms']}")
    
    # Option 2: Direct semantic search for term+definition matches
    semantic_results = await semantic_term_definition_search(
        db, collection_name, user_query
    )
    
    logger.info(f"Semantic search found {len(semantic_results)} relevant terms")
    
    # Option 3: Fuzzy search for a specific term
    term = "Azure"  # Example term
    fuzzy_results = await fuzzy_glossary_search(
        db, view_name, term
    )
    
    logger.info(f"Fuzzy search for '{term}' found {len(fuzzy_results)} matches")
    
    return {
        "analysis": analysis,
        "semantic_results": semantic_results,
        "fuzzy_results": fuzzy_results
    } 