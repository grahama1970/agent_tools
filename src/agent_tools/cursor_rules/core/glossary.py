#!/usr/bin/env python
"""
Glossary implementation for cursor_rules.

This module provides the functionality to create, manage, and search a glossary
of technical terms and definitions related to the codebase. It enhances the 
existing hybrid search by adding domain-specific terminology support.
"""

import asyncio
import textwrap
from typing import Dict, Any, List, Union, Optional, Tuple
import rapidfuzz
from arango import ArangoClient
from arango.exceptions import ArangoError
from pathlib import Path
import json

from agent_tools.cursor_rules.core.cursor_rules import (
    generate_embedding,
    create_arangosearch_view
)

# Schema definition for the glossary
GLOSSARY_SCHEMA = {
    "type": "object",
    "required": ["term", "definition"],
    "properties": {
        "term": {"type": "string", "description": "The terminology or concept name"},
        "definition": {"type": "string", "description": "Detailed explanation of the term"},
        "category": {"type": "string", "description": "Optional category for grouping terms"},
        "related_terms": {"type": "array", "items": {"type": "string"}, "description": "Related terminology"},
        "examples": {"type": "array", "items": {"type": "string"}, "description": "Usage examples"},
        "aliases": {"type": "array", "items": {"type": "string"}, "description": "Alternative names for the term"},
        "importance": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Importance level"},
        "notes": {"type": "string", "description": "Additional notes or context"}
    }
}

# Sample glossary entries for testing
SAMPLE_GLOSSARY = [
    {
        "term": "Hybrid Search",
        "definition": "A search technique that combines vector similarity and keyword-based (BM25) methods for improved relevance",
        "category": "search",
        "related_terms": ["Vector Search", "BM25", "Semantic Search"],
        "examples": ["Combining embeddings with keyword matching to find relevant documents"],
        "importance": "high"
    },
    {
        "term": "Embedding",
        "definition": "A numerical vector representation of text that captures semantic meaning",
        "category": "nlp",
        "related_terms": ["Vector", "Transformer", "Semantic Representation"],
        "examples": ["Converting a paragraph to a 768-dimensional vector"],
        "importance": "high"
    },
    {
        "term": "BM25",
        "definition": "Best Match 25 - a ranking function used to score documents based on keyword matches",
        "category": "search",
        "related_terms": ["TF-IDF", "Information Retrieval", "Keyword Search"],
        "examples": ["Scoring documents based on term frequency and document length"],
        "importance": "medium"
    },
    {
        "term": "ArangoSearch",
        "definition": "Full-text search and ranking engine integrated into ArangoDB",
        "category": "database",
        "related_terms": ["Full-text Search", "ArangoDB", "View"],
        "examples": ["Creating a view for searching document contents"],
        "importance": "medium"
    }
]

def format_for_embedding(term: str, definition: str) -> str:
    """
    Format a term and definition for embedding using ModernBERT conventions.
    
    Args:
        term: The terminology or concept
        definition: The definition or explanation
        
    Returns:
        Formatted text ready for embedding
    """
    # Use the required 'search_document:' prefix for Nomic embeddings
    # This is the correct format as specified in Nomic documentation
    return f"search_document: {term} - {definition}"

async def create_glossary_collection(db, collection_name="glossary"):
    """
    Create a glossary collection in the database.
    
    Args:
        db: Database handle
        collection_name: Name of the collection
        
    Returns:
        True if created or already exists, False on error
    """
    try:
        # Check if collection exists
        if await asyncio.to_thread(db.has_collection, collection_name):
            print(f"Collection '{collection_name}' already exists")
            return True
        
        # Create collection
        await asyncio.to_thread(db.create_collection, collection_name)
        print(f"Created collection '{collection_name}'")
        
        # Create ArangoSearch view for the glossary
        view_name = f"{collection_name}_view"
        await asyncio.to_thread(
            create_arangosearch_view, 
            db, 
            collection_name, 
            view_name
        )
        
        return True
    except Exception as e:
        print(f"Error creating glossary collection: {e}")
        return False

async def insert_sample_glossary(db, collection_name="glossary"):
    """
    Insert sample glossary entries for testing.
    
    Args:
        db: Database handle
        collection_name: Name of the glossary collection
        
    Returns:
        Number of entries inserted
    """
    try:
        # Get the collection
        collection = await asyncio.to_thread(db.collection, collection_name)
        
        # Check if collection is empty
        count_query = f"RETURN LENGTH(FOR doc IN {collection_name} RETURN 1)"
        cursor = await asyncio.to_thread(db.aql.execute, count_query)
        count = await asyncio.to_thread(next, cursor, 0)
        
        if count > 0:
            print(f"Glossary collection already contains {count} entries")
            return 0
        
        # Insert sample entries with embeddings
        inserted = 0
        for entry in SAMPLE_GLOSSARY:
            # Format and generate embedding
            text_to_embed = format_for_embedding(entry["term"], entry["definition"])
            embedding_result = generate_embedding(text_to_embed)
            
            if embedding_result and "embedding" in embedding_result:
                entry["embedding"] = embedding_result["embedding"]
                entry["embedding_metadata"] = embedding_result.get("metadata", {})
                
                # Insert the entry
                await asyncio.to_thread(collection.insert, entry)
                inserted += 1
        
        print(f"Inserted {inserted} sample glossary entries")
        return inserted
    except Exception as e:
        print(f"Error inserting sample glossary: {e}")
        return 0

async def load_glossary_from_json(db, file_path, collection_name="glossary"):
    """
    Load glossary entries from a JSON file.
    
    Args:
        db: Database handle
        file_path: Path to the JSON file
        collection_name: Name of the glossary collection
        
    Returns:
        Number of entries loaded
    """
    try:
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            print(f"Glossary file not found: {file_path}")
            return 0
        
        # Load JSON file
        with open(path, 'r') as f:
            glossary_data = json.load(f)
        
        # Get collection
        collection = await asyncio.to_thread(db.collection, collection_name)
        
        # Insert entries with embeddings
        inserted = 0
        for entry in glossary_data:
            # Validate required fields
            if "term" not in entry or "definition" not in entry:
                print(f"Skipping entry missing required fields: {entry}")
                continue
                
            # Format and generate embedding
            text_to_embed = format_for_embedding(entry["term"], entry["definition"])
            embedding_result = generate_embedding(text_to_embed)
            
            if embedding_result and "embedding" in embedding_result:
                entry["embedding"] = embedding_result["embedding"]
                entry["embedding_metadata"] = embedding_result.get("metadata", {})
                
                # Insert the entry
                await asyncio.to_thread(collection.insert, entry)
                inserted += 1
        
        print(f"Loaded {inserted} glossary entries from {file_path}")
        return inserted
    except Exception as e:
        print(f"Error loading glossary from JSON: {e}")
        return 0

async def glossary_search(db, query_text, collection_name="glossary", limit=5, min_similarity=0.98, verbose=False):
    """
    Perform glossary search using hybrid approach and RapidFuzz verification.
    
    Args:
        db: Database handle
        query_text: The search query
        collection_name: Name of the glossary collection
        limit: Maximum number of results to return
        min_similarity: Minimum similarity threshold for RapidFuzz verification
        verbose: Whether to print detailed information
        
    Returns:
        List of matching glossary entries with scores
    """
    try:
        if not query_text or query_text.strip() == "":
            return []
            
        if verbose:
            print("\n" + "="*80)
            print(f"GLOSSARY SEARCH: {query_text}")
            print("="*80)
        
        # Ensure view exists
        view_name = f"{collection_name}_view"
        view_created = await asyncio.to_thread(create_arangosearch_view, db, collection_name, view_name)
        
        if not view_created:
            print(f"Failed to create view for glossary search")
            return []
        
        # Generate embedding for query
        query_embedding = generate_embedding(f"[TERM: {query_text}]")
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Thresholds
        bm25_threshold = 0.1
        embedding_threshold = 0.5
        hybrid_threshold = 0.15
        
        # AQL query combining embedding similarity and BM25
        aql_query = textwrap.dedent(f"""
        LET embedding_results = (
            FOR term IN {collection_name}
                LET similarity = COSINE_SIMILARITY(term.embedding, @query_vector)
                FILTER similarity >= @embedding_threshold
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
            FOR term IN @@view
                SEARCH ANALYZER(
                    term.term IN TOKENS(@search_text, "text_en") OR
                    term.definition IN TOKENS(@search_text, "text_en"),
                    "text_en"
                )
                LET bm25_score = BM25(term, @k1, @b)
                FILTER bm25_score > @bm25_threshold
                SORT bm25_score DESC
                LIMIT @limit
                RETURN {{
                    term: term,
                    _key: term._key,
                    similarity_score: 0,
                    bm25_score: bm25_score
                }}
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
                RETURN {{
                    "term_doc": term_doc,
                    "_key": key,
                    "similarity_score": similarity_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score
                }}
        )
        
        // Sort and limit final results
        LET final_results = (
            FOR result IN merged_results
                SORT result.hybrid_score DESC
                LIMIT @limit
                RETURN result
        )
        
        RETURN final_results
        """)
        
        # Execute query
        bind_vars = {
            "@view": view_name,
            "@collection": collection_name,
            "search_text": query_text,
            "query_vector": query_embedding["embedding"],
            "embedding_threshold": embedding_threshold,
            "bm25_threshold": bm25_threshold,
            "hybrid_threshold": hybrid_threshold,
            "k1": k1,
            "b": b,
            "limit": limit
        }
        
        cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars=bind_vars)
        results_list = await asyncio.to_thread(list, cursor)
        
        # The query returns a list with a single item, which is the list of results
        if not results_list:
            return []
            
        results = results_list[0]
        
        # Final verification with RapidFuzz for exact term matching
        verified_results = []
        for result in results:
            term_doc = result["term_doc"]
            term = term_doc.get("term", "")
            
            # Apply RapidFuzz for final verification
            ratio = rapidfuzz.fuzz.ratio(query_text.lower(), term.lower())
            
            if verbose:
                print(f"RapidFuzz check: '{query_text}' vs '{term}' = {ratio:.2f}")
                
            # Use original result if it passes the similarity threshold
            if ratio >= min_similarity * 100:  # RapidFuzz uses 0-100 scale
                verified_results.append((term_doc, result["hybrid_score"], result["bm25_score"], result["similarity_score"]))
            elif verbose:
                print(f"Term '{term}' rejected with similarity {ratio:.2f} < {min_similarity * 100}")
        
        if verbose:
            print(f"Found {len(verified_results)} verified glossary terms")
        
        return verified_results
    
    except Exception as e:
        print(f"Error in glossary search: {e}")
        import traceback
        traceback.print_exc()
        return []

async def enhanced_hybrid_search(
    db, 
    query_text, 
    knowledge_collection="rules", 
    glossary_collection="glossary", 
    limit=5, 
    verbose=False,
    **kwargs
):
    """
    Perform enhanced hybrid search that includes glossary entries.
    
    This function combines results from both the regular hybrid search
    on knowledge base and glossary search.
    
    Args:
        db: Database handle
        query_text: The search query
        knowledge_collection: Name of the knowledge collection
        glossary_collection: Name of the glossary collection
        limit: Maximum number of results
        verbose: Whether to display detailed information
        **kwargs: Additional parameters for search customization
        
    Returns:
        Combined search results from knowledge base and glossary
    """
    from agent_tools.cursor_rules.core.cursor_rules import hybrid_search
    
    # Run both searches concurrently
    knowledge_results_future = asyncio.create_task(
        hybrid_search(db, query_text, collection_name=knowledge_collection, limit=limit, verbose=verbose, **kwargs)
    )
    
    glossary_results_future = asyncio.create_task(
        glossary_search(db, query_text, collection_name=glossary_collection, limit=limit, verbose=verbose)
    )
    
    # Wait for both to complete
    knowledge_results = await knowledge_results_future
    glossary_results = await glossary_results_future
    
    if verbose:
        print(f"\nFound {len(knowledge_results)} knowledge base results and {len(glossary_results)} glossary results")
    
    # Process and combine results
    combined_results = []
    
    # Add knowledge base results
    for item, score in knowledge_results:
        combined_results.append({
            "type": "knowledge",
            "content": item,
            "score": score,
            "source": "knowledge_base"
        })
    
    # Add glossary results
    for term_doc, hybrid_score, bm25_score, similarity_score in glossary_results:
        combined_results.append({
            "type": "glossary",
            "content": term_doc,
            "score": hybrid_score,
            "source": "glossary",
            "bm25_score": bm25_score,
            "similarity_score": similarity_score
        })
    
    # Sort all results by score
    combined_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Limit to requested number
    return combined_results[:limit]

async def setup_glossary(db, glossary_file=None):
    """
    Set up the glossary database.
    
    Args:
        db: Database handle
        glossary_file: Optional path to a JSON file with glossary entries
        
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Create glossary collection
        collection_created = await create_glossary_collection(db)
        if not collection_created:
            return False
        
        # Load glossary data
        if glossary_file:
            # Load from provided file
            entries_loaded = await load_glossary_from_json(db, glossary_file)
            if entries_loaded == 0:
                # Fall back to sample data if no entries loaded
                await insert_sample_glossary(db)
        else:
            # Use sample data
            await insert_sample_glossary(db)
        
        return True
    except Exception as e:
        print(f"Error setting up glossary: {e}")
        return False

def glossary_term_to_embedding_text(term, definition):
    """Format a glossary term for embedding in a way that works well with ModernBERT."""
    return f"[TERM: {term}] {definition}"

def embed_term_in_query(query, term, definition=None, include_definition=False):
    """
    Embed a glossary term in a user query by wrapping it in ModernBERT conventions.
    
    This function properly identifies the term within the query text and wraps it
    with the [TERM: term] annotation. It can optionally include the definition
    of the term inline with the annotation.
    
    Args:
        query: The original user query
        term: The term to embed
        definition: The definition of the term (optional)
        include_definition: Whether to include the definition in the annotation
        
    Returns:
        The query with the term embedded using ModernBERT conventions
    """
    # Prepare the term annotation
    if include_definition and definition:
        term_annotation = f"[TERM: {term} | {definition}]"
    else:
        term_annotation = f"[TERM: {term}]"
    
    # Handle case where the term isn't in the query
    if term.lower() not in query.lower():
        # If term isn't found, return original with annotation appended
        return f"{query} {term_annotation}"
    
    # Find the term in the query (case-insensitive) and replace with annotation
    import re
    # Use word boundaries to avoid partial replacements
    # For example, avoid replacing "neural network" in "neural networks" incorrectly
    pattern = re.compile(r'\b' + re.escape(term) + r'(?:s|es)?\b', re.IGNORECASE)
    embedded_query = pattern.sub(term_annotation, query)
    
    return embedded_query 