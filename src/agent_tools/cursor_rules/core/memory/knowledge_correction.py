#!/usr/bin/env python3
"""
Knowledge Correction Module for Agent Memory System

This module implements the knowledge correction functionality for the agent memory system.
It handles detecting contradictions, updating existing knowledge, and maintaining a history
of knowledge changes.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
- ArangoDB Vector Search: https://www.arangodb.com/docs/stable/aql/functions-vector.html
"""

from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import datetime
import pprint

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem, MemoryFact
from agent_tools.cursor_rules.utils.vector_utils import truncate_vector_for_display, get_vector_stats


def update_knowledge(
    memory_system: AgentMemorySystem,
    fact_content: str,
    similarity_threshold: float = 0.85,
    confidence: float = 0.8,
    domains: List[str] = None,
    ttl_days: float = 365.0,
    importance: float = 0.5,
    source: Optional[str] = None
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """
    Update knowledge by checking for similar facts and updating them if found.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_content: The content of the new fact
        similarity_threshold: Threshold for detecting similar facts (0.0-1.0)
        confidence: Confidence level for the new information (0.0-1.0)
        domains: List of domains the fact belongs to
        ttl_days: Time-to-live in days for the fact
        importance: Importance level of the fact (0.0-1.0)
        source: Optional source of the information
        
    Returns:
        Tuple of (fact_id, is_new, previous_version_if_updated)
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    logger.info(f"Updating knowledge with fact: '{fact_content[:50]}...'")
    logger.debug(f"Parameters: similarity_threshold={similarity_threshold}, confidence={confidence}, domains={domains}")
    
    # Ensure domains is at least an empty list
    if domains is None:
        domains = []
    
    # First, check if similar facts exist
    similar_facts = find_similar_facts(
        memory_system, 
        fact_content, 
        similarity_threshold
    )
    
    # If no similar facts found, create a new one
    if not similar_facts:
        logger.info("No similar facts found, creating new fact")
        result = memory_system.remember(
            content=fact_content,
            confidence=confidence,
            domains=domains,
            ttl_days=ttl_days,
            importance=importance,
            source=source
        )
        
        if result and 'new' in result:
            fact_id = result['new']['fact_id']
            logger.info(f"Created new fact with ID: {fact_id}")
            return fact_id, True, None
        else:
            logger.error("Failed to create new fact")
            return None, False, None
    
    # Get the most similar fact
    most_similar = similar_facts[0]
    fact_id = most_similar["fact_id"]
    logger.info(f"Found similar fact with ID: {fact_id}")
    logger.debug(f"Similar fact: '{most_similar['content'][:50]}...'")
    logger.debug(f"Similarity score: {most_similar.get('similarity_score', 0):.3f}")
    
    # Check if confidence in new fact is higher or equal
    if confidence >= most_similar.get("confidence", 0.0):
        logger.info(f"New fact has higher confidence ({confidence} vs {most_similar.get('confidence', 0.0)}), updating")
        
        # Store the previous version
        correction_history = most_similar.get("correction_history", [])
        previous_version = {
            "content": most_similar["content"],
            "confidence": most_similar.get("confidence", 0.0),
            "timestamp": most_similar.get("last_updated", 
                          datetime.datetime.now().isoformat()),
            "replaced_on": datetime.datetime.now().isoformat()
        }
        
        # Add source information if available
        if source:
            previous_version["replaced_by_source"] = source
        
        # Update the fact with new information
        correction_history.append(previous_version)
        
        # Update domains by merging existing with new domains
        merged_domains = list(set(most_similar.get("domains", []) + domains))
        
        # Calculate the new importance - higher of the two with a small boost
        new_importance = importance if importance > most_similar.get("importance", 0.0) else most_similar.get("importance", 0.0) + 0.05
        new_importance = new_importance if new_importance < 1.0 else 1.0  # Cap at 1.0
        
        # Update the fact
        result = memory_system.remember(
            fact_id=fact_id,
            content=fact_content,
            confidence=confidence,
            domains=merged_domains,
            ttl_days=ttl_days if ttl_days > most_similar.get("ttl_days", 0) else most_similar.get("ttl_days", 0),
            importance=new_importance,
            source=source,
            correction_history=correction_history
        )
        
        if result:
            logger.info(f"Successfully updated fact {fact_id}")
        else:
            logger.error(f"Failed to update fact {fact_id}")
        
        return fact_id, False, previous_version
    
    # If our confidence is lower, keep the existing fact but record the alternative
    else:
        logger.info(f"New fact has lower confidence ({confidence} vs {most_similar.get('confidence', 0.0)}), adding as alternative")
        
        # Get the existing alternatives
        alternatives = most_similar.get("alternatives", [])
        
        # Add our version as an alternative
        new_alternative = {
            "content": fact_content,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source
        }
        
        alternatives.append(new_alternative)
        
        # Update the fact with the new alternative
        result = memory_system.remember(
            fact_id=fact_id,
            alternatives=alternatives
        )
        
        if result:
            logger.info(f"Successfully added alternative to fact {fact_id}")
        else:
            logger.error(f"Failed to add alternative to fact {fact_id}")
        
        return fact_id, False, None

def find_similar_facts(
    memory_system: AgentMemorySystem,
    fact_content: str,
    similarity_threshold: float = 0.85,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find facts similar to the given content using vector similarity.
    If vector embedding fails, falls back to text search.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_content: Content to compare against
        similarity_threshold: Similarity threshold (0.0-1.0)
        limit: Maximum number of similar facts to return
        
    Returns:
        List of similar facts with similarity scores
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    # Get embedding for the fact content
    embedding_data = memory_system._get_embedding(fact_content)
    
    # Check if we have a valid embedding (not empty)
    if embedding_data and 'embedding' in embedding_data and embedding_data['embedding'] and len(embedding_data['embedding']) > 0:
        embedding = embedding_data['embedding']
        # Log with truncated vector for debugging
        stats = get_vector_stats(embedding)
        logger.debug(f"Generated embedding for similarity search:")
        logger.debug(f"  Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, norm={stats['norm']:.3f}")
        logger.debug(f"  Vector: {truncate_vector_for_display(embedding, max_items=3)}")
        
        collection_name = memory_system.config["facts_collection"]
        # Fixed AQL query for vector similarity search - ensure we only search documents with embeddings
        aql_query = f"""
        FOR fact IN {collection_name}
            FILTER fact.embedding != null
            LET similarity = COSINE_SIMILARITY(fact.embedding, @embedding)
            FILTER similarity >= @threshold
            SORT similarity DESC
            LIMIT @limit
            RETURN MERGE(fact, {{ 
                similarity_score: similarity,
                domains: fact.domains || []
            }})
        """
        
        bind_vars = {
            "embedding": embedding,
            "threshold": similarity_threshold,
            "limit": limit
        }
        
        try:
            cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
            logger.debug(f"Found {len(results)} similar facts with threshold {similarity_threshold}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            logger.info("Falling back to text search")
            # Fall through to text search fallback
    else:
        logger.warning(f"Failed to generate valid embedding for: '{fact_content[:50]}...', falling back to text search")
    
    # Fallback to text search if embedding fails or is empty
    try:
        collection_name = memory_system.config["facts_collection"]
        view_name = f"{collection_name}_view"
        
        # Check if view exists, if not, fall back to simple collection query
        views = memory_system.db.views()
        view_exists = any(v["name"] == view_name for v in views)
        
        if view_exists:
            # Use ArangoSearch view for better text matching
            aql_query = f"""
            FOR fact IN {view_name}
                SEARCH ANALYZER(fact.content IN TOKENS(@content, "text_en"), "text_en")
                SORT BM25(fact) DESC
                LIMIT @limit
                RETURN MERGE(fact, {{ 
                    similarity_score: 0.5,
                    domains: fact.domains || []
                }})  // Default similarity for text search
            """
        else:
            # Simple fallback if view doesn't exist
            aql_query = f"""
            FOR fact IN {collection_name}
                FILTER CONTAINS(LOWER(fact.content), LOWER(@content))
                SORT fact.importance DESC
                LIMIT @limit
                RETURN MERGE(fact, {{ 
                    similarity_score: 0.5,
                    domains: fact.domains || []
                }})
            """
        
        bind_vars = {
            "content": fact_content,
            "limit": limit
        }
        
        cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        logger.debug(f"Found {len(results)} similar facts using text search fallback")
        return results
    except Exception as e:
        logger.error(f"Error during text search fallback: {e}")
        return []

def get_fact_history(
    memory_system: AgentMemorySystem,
    fact_id: str
) -> Dict[str, Any]:
    """
    Get the full history of a fact, including all corrections and alternatives.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the fact to get history for
        
    Returns:
        Dict with the fact and its history
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    # First try with fact_id field, then _key as fallback
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER fact.fact_id == @fact_id OR fact._key == @fact_id
        RETURN {{
            current: {{
                content: fact.content,
                confidence: fact.confidence || 0.5,
                domains: fact.domains || [],
                importance: fact.importance || 0.5,
                ttl_days: fact.ttl_days || 30,
                created_at: fact.created_at,
                last_updated: fact.last_updated || fact.created_at,
                access_count: fact.access_count || 0,
                last_accessed: fact.last_accessed || fact.created_at
            }},
            correction_history: fact.correction_history || [],
            alternatives: fact.alternatives || [],
            source: fact.source
        }}
    """
    
    bind_vars = {
        "fact_id": fact_id
    }
    
    try:
        cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
        result = next(cursor, None)
        
        if not result:
            # If we didn't find with either approach, try one more approach by handling collection prefixes
            if '/' in fact_id:
                # If it's in the format collection/key, extract just the key
                key = fact_id.split('/')[-1]
                bind_vars = {"fact_id": key}
                cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
                result = next(cursor, None)
        
        if not result:
            raise ValueError(f"Fact with ID {fact_id} not found")
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving fact history: {e}")
        raise

def resolve_contradictions(
    memory_system: AgentMemorySystem,
    fact_id: str,
    resolution_content: str,
    confidence: float = 0.9,
    resolution_notes: str = None
) -> str:
    """
    Resolve contradictions between a fact and its alternatives by creating a new 
    unified fact.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the fact to resolve
        resolution_content: The resolved content
        confidence: Confidence in the resolution
        resolution_notes: Optional notes about the resolution
        
    Returns:
        ID of the new resolved fact
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    # Get the original fact and its history
    history = get_fact_history(memory_system, fact_id)
    
    # Create a new fact with the resolution
    new_fact_id = memory_system.remember(
        content=resolution_content,
        confidence=confidence,
        domains=history["current"].get("domains", []),
        importance=history["current"].get("importance", 0.5) + 0.1 if history["current"].get("importance", 0.5) + 0.1 < 1.0 else 1.0,
        source="contradiction_resolution",
        resolution_notes=resolution_notes,
        resolved_from=fact_id,
        alternatives=history["alternatives"]
    )
    
    # Mark the original fact as resolved
    collection_name = memory_system.config["facts_collection"]
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER fact._key == @fact_id
        UPDATE fact WITH {{
            resolved_by: @new_fact_id,
            resolution_date: DATE_ISO8601(DATE_NOW())
        }} IN {collection_name}
        RETURN NEW
    """
    
    bind_vars = {
        "fact_id": fact_id,
        "new_fact_id": new_fact_id
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    list(cursor)  # Execute but ignore result
    
    return new_fact_id

def analyze_contradictions(memory_system: AgentMemorySystem) -> List[Dict[str, Any]]:
    """
    Analyze the memory system for potential contradictions.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        
    Returns:
        List of facts with contradictions
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER LENGTH(fact.alternatives) > 0
        FILTER fact.resolved_by == null  // Only unresolved contradictions
        
        LET contradiction_score = LENGTH(fact.alternatives) * 
            (1.0 - fact.confidence)  // More alternatives and lower confidence = higher score
        
        SORT contradiction_score DESC
        
        RETURN {{
            fact_id: fact._key,
            content: fact.content,
            confidence: fact.confidence,
            alternatives: fact.alternatives,
            contradiction_score: contradiction_score
        }}
    """
    
    cursor = memory_system.db.aql.execute(aql_query)
    return list(cursor)

def merge_facts(
    memory_system: AgentMemorySystem,
    fact_ids: List[str],
    merged_content: str,
    confidence: float = 0.95,
    merge_notes: str = None
) -> Dict[str, Any]:
    """
    Merge multiple related facts into a single, more comprehensive fact.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_ids: List of fact IDs to merge
        merged_content: The merged content
        confidence: Confidence in the merged fact
        merge_notes: Optional notes about the merge
        
    Returns:
        The newly created merged fact
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    # First, get all facts to be merged
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER fact._key IN @fact_ids
        RETURN fact
    """
    
    bind_vars = {
        "fact_ids": fact_ids
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    facts = list(cursor)
    
    if len(facts) < 2:
        raise ValueError("Need at least two facts to merge")
    
    # Combine domains and calculate new importance
    all_domains = set()
    for fact in facts:
        all_domains.update(fact.get("domains", []))
    
    # Use the highest importance plus a small boost
    max_importance = max(fact.get("importance", 0.5) for fact in facts)
    merged_importance = max_importance + 0.05 if max_importance + 0.05 < 1.0 else 1.0
    
    # Create the merged fact
    merged_fact_id = memory_system.remember(
        content=merged_content,
        confidence=confidence,
        domains=list(all_domains),
        importance=merged_importance,
        source="fact_merge",
        merge_notes=merge_notes,
        merged_from=fact_ids
    )
    
    # Mark original facts as merged
    update_query = f"""
    FOR fact IN {collection_name}
        FILTER fact._key IN @fact_ids
        UPDATE fact WITH {{
            merged_into: @merged_fact_id,
            merge_date: DATE_ISO8601(DATE_NOW())
        }} IN {collection_name}
        RETURN NEW
    """
    
    bind_vars = {
        "fact_ids": fact_ids,
        "merged_fact_id": merged_fact_id
    }
    
    cursor = memory_system.db.aql.execute(update_query, bind_vars=bind_vars)
    list(cursor)  # Execute but ignore result
    
    # Return the merged fact
    return memory_system.get_fact(merged_fact_id)

def debug_knowledge_correction():
    """
    Debug utility for testing knowledge correction functionality.
    
    This function:
    1. Initializes a memory system with test data
    2. Adds initial facts
    3. Tests updating knowledge with similar facts
    4. Demonstrates contradiction detection and resolution
    5. Shows fact merging
    """
    import argparse
    from agent_tools.cursor_rules.utils.vector_utils import truncate_vector_for_display, get_vector_stats
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug knowledge correction system')
    parser.add_argument('--debug_type', type=str, default="all",
                      choices=['update', 'contradiction', 'merge', 'all', 'mars_example'],
                      help='Type of debugging to perform')
    parser.add_argument('--similarity', type=float, default=0.7,
                      help='Similarity threshold for finding similar facts')
    args = parser.parse_args()
    
    # Initialize memory system
    memory = AgentMemorySystem({
        "arango_host": "http://localhost:8529",
        "username": "root",
        "password": "openSesame"
    })
    memory.initialize()
    
    # Clear existing data for clean testing
    print("Clearing existing data...")
    memory.db.aql.execute(f"FOR doc IN {memory.config['facts_collection']} REMOVE doc IN {memory.config['facts_collection']}")
    
    # Add initial facts
    print("\nAdding initial facts...")
    initial_facts = [
        {
            "content": "The Earth orbits the Sun in approximately 365 days",
            "confidence": 0.9,
            "domains": ["astronomy", "science"],
            "importance": 0.8
        },
        {
            "content": "Water freezes at 0 degrees Celsius at standard pressure",
            "confidence": 0.95,
            "domains": ["chemistry", "science"],
            "importance": 0.7
        },
        {
            "content": "The Eiffel Tower is located in Paris, France",
            "confidence": 0.99,
            "domains": ["geography", "landmarks"],
            "importance": 0.6
        }
    ]
    
    fact_ids = []
    for fact in initial_facts:
        result = memory.remember(**fact)
        if result and 'new' in result:
            fact_id = result['new']['fact_id']
            fact_ids.append(fact_id)
            print(f"Added fact: '{fact['content'][:50]}...' with ID: {fact_id}")
    
    # Mars Example - Demonstrates knowledge correction in detail
    if args.debug_type in ['mars_example', 'all']:
        print("\n=== Mars Moons Knowledge Correction Example ===")
        print("\nThis example demonstrates how knowledge correction works with document-embedded history")
        
        # Step 1: Initial fact
        print("\nStep 1: Adding initial fact about Mars moons")
        initial_mars_fact = {
            "content": "Mars has two moons: Phobos and Deimos.",
            "confidence": 0.9,
            "domains": ["astronomy", "planets"],
            "importance": 0.7
        }
        
        mars_result = memory.remember(**initial_mars_fact)
        mars_fact_id = mars_result['new']['fact_id']
        
        print(f"Added fact with ID: {mars_fact_id}")
        print("Document structure (initial):")
        print("  content: \"Mars has two moons: Phobos and Deimos.\"")
        print("  confidence: 0.9")
        print("  previous_content: null")
        print("  domains: [\"astronomy\", \"planets\"]")
        print("  alternatives: []")
        print("  correction_history: []")
        print("  updated_at: null")
        
        # Print actual document for verification
        fact_doc = memory.facts.get(mars_fact_id)
        
        if fact_doc:
            print("\nActual stored document (simplified):")
            for key in ['fact_id', 'content', 'confidence', 'previous_content', 'domains', 'updated_at']:
                if key in fact_doc:
                    print(f"  {key}: {fact_doc[key]}")
        
        # Step 2: Update with higher confidence
        print("\nStep 2: Updating with more detailed information (higher confidence)")
        mars_update = {
            "fact_id": mars_fact_id,
            "content": "Mars has two small moons: Phobos (diameter 22.2 km) and Deimos (diameter 12.6 km).",
            "confidence": 0.95,
            "domains": ["astronomy", "planets", "satellites"],
            "importance": 0.75
        }
        
        update_result = memory.remember(**mars_update)
        
        print("Document structure after update:")
        print("  content: \"Mars has two small moons: Phobos (diameter 22.2 km) and Deimos (diameter 12.6 km).\"")
        print("  confidence: 0.95")
        print("  previous_content: \"Mars has two moons: Phobos and Deimos.\"")  # Key knowledge correction feature
        print("  domains: [\"astronomy\", \"planets\", \"satellites\"]")
        print("  updated_at: (timestamp)")
        
        # Print actual document for verification
        fact_doc = memory.facts.get(mars_fact_id)
        
        if fact_doc:
            print("\nActual stored document after update (simplified):")
            for key in ['fact_id', 'content', 'confidence', 'previous_content', 'domains', 'updated_at']:
                if key in fact_doc:
                    print(f"  {key}: {fact_doc[key]}")
        
        # Step 3: Add alternative with lower confidence
        print("\nStep 3: Adding alternative information with lower confidence")
        alternative_fact = "Mars has two tiny captured asteroids as moons."
        
        alt_id, is_new, previous = update_knowledge(
            memory_system=memory,
            fact_content=alternative_fact,
            similarity_threshold=args.similarity,
            confidence=0.7,  # Lower confidence
            domains=["astronomy"]
        )
        
        print(f"Added alternative to fact {alt_id}, is_new: {is_new}")
        print("Document structure after adding alternative:")
        print("  content: \"Mars has two small moons: Phobos (diameter 22.2 km) and Deimos (diameter 12.6 km).\"")
        print("  confidence: 0.95")
        print("  previous_content: \"Mars has two moons: Phobos and Deimos.\"")
        print("  alternatives: [")
        print("    {")
        print("      \"content\": \"Mars has two tiny captured asteroids as moons.\"")
        print("      \"confidence\": 0.7")
        print("    }")
        print("  ]")
        
        # Print actual document for verification
        fact_doc = memory.facts.get(alt_id)
        
        if fact_doc:
            print("\nActual stored document with alternative (simplified):")
            print(f"  content: {fact_doc['content']}")
            print(f"  confidence: {fact_doc['confidence']}")
            print(f"  previous_content: {fact_doc.get('previous_content')}")
            if 'alternatives' in fact_doc:
                print("  alternatives:")
                for alt in fact_doc['alternatives']:
                    print(f"    - content: {alt['content']}")
                    print(f"      confidence: {alt['confidence']}")
        
        # Step 4: Contradicting information with higher confidence
        print("\nStep 4: Adding contradicting information with higher confidence")
        contradiction_fact = "Mars has three moons: Phobos, Deimos, and a recently discovered small moonlet."
        
        contr_id, is_new, previous = update_knowledge(
            memory_system=memory,
            fact_content=contradiction_fact,
            similarity_threshold=args.similarity,
            confidence=0.98,  # Higher confidence
            domains=["astronomy", "recent_discoveries"],
            source="NASA press release"
        )
        
        print(f"Updated fact {contr_id} with contradiction, is_new: {is_new}")
        if previous:
            print(f"Previous content moved to correction history: \"{previous['content']}\"")
        
        # Print actual document for verification
        fact_doc = memory.facts.get(contr_id)
        
        if fact_doc:
            print("\nActual stored document after contradiction (simplified):")
            print(f"  content: {fact_doc['content']}")
            print(f"  confidence: {fact_doc['confidence']}")
            print(f"  previous_content: {fact_doc.get('previous_content')}")
            print(f"  source: {fact_doc.get('source')}")
            if 'correction_history' in fact_doc:
                print("  correction_history:")
                for correction in fact_doc['correction_history']:
                    print(f"    - content: {correction.get('content')}")
                    print(f"      confidence: {correction.get('confidence')}")
                    print(f"      replaced_on: {correction.get('replaced_on')}")
        
        # Step 5: Resolving the contradiction (creates a new document with relationship)
        print("\nStep 5: Resolving contradiction (creates a new document)")
        resolution = "Mars has two confirmed moons (Phobos and Deimos) and potentially a third moonlet awaiting confirmation."
        
        resolved_id = resolve_contradictions(
            memory_system=memory,
            fact_id=contr_id,
            resolution_content=resolution,
            confidence=0.99,
            resolution_notes="Combined established knowledge with recent observations pending verification"
        )
        
        print(f"Created resolution as new document with ID: {resolved_id}")
        print("Original document now has 'resolved_by' field pointing to new document")
        
        # Print both documents
        orig_doc = memory.facts.get(contr_id)
        resolved_doc = memory.facts.get(resolved_id)
    
    # Test knowledge update
    if args.debug_type in ['update', 'all']:
        print("\n=== Testing Knowledge Update ===")
        
        # Slightly different version of an existing fact
        updated_fact = "The Earth completes one orbit around the Sun in 365.25 days"
        print(f"\nUpdating with new fact: '{updated_fact}'")
        
        fact_id, is_new, previous = update_knowledge(
            memory,
            fact_content=updated_fact,
            similarity_threshold=args.similarity,
            confidence=0.92,
            domains=["astronomy", "physics"]
        )
        
        print(f"Result: {'New fact created' if is_new else 'Existing fact updated'}")
        print(f"Fact ID: {fact_id}")
        if previous:
            print(f"Previous version: '{previous['content']}'")
            print(f"Previous confidence: {previous['confidence']}")
        
        # Get the updated fact
        if not is_new:
            history = get_fact_history(memory, fact_id)
            print("\nUpdated fact history:")
            print(f"Current content: '{history['current']['content']}'")
            print(f"Current confidence: {history['current']['confidence']}")
            print(f"Domains: {history['current']['domains']}")
            print(f"Correction history: {len(history['correction_history'])} entries")
    
    # Test contradiction handling
    if args.debug_type in ['contradiction', 'all']:
        print("\n=== Testing Contradiction Handling ===")
        
        # Add a contradicting fact with lower confidence
        contradicting_fact = "Water freezes at 32 degrees Fahrenheit at standard pressure"
        print(f"\nAdding contradicting fact: '{contradicting_fact}'")
        
        fact_id, is_new, previous = update_knowledge(
            memory,
            fact_content=contradicting_fact,
            similarity_threshold=args.similarity,
            confidence=0.85,  # Lower confidence than original
            domains=["chemistry", "science"]
        )
        
        print(f"Result: {'New fact created' if is_new else 'Alternative added to existing fact'}")
        print(f"Fact ID: {fact_id}")
        
        # Analyze contradictions
        contradictions = analyze_contradictions(memory)
        print(f"\nFound {len(contradictions)} facts with contradictions:")
        for i, contradiction in enumerate(contradictions):
            print(f"\nContradiction {i+1}:")
            print(f"  Main fact: '{contradiction['content']}'")
            print(f"  Confidence: {contradiction['confidence']}")
            print(f"  Alternatives: {len(contradiction['alternatives'])}")
            print(f"  Contradiction score: {contradiction['contradiction_score']:.3f}")
            
            # Show alternatives
            for j, alt in enumerate(contradiction['alternatives']):
                print(f"    Alternative {j+1}: '{alt['content']}'")
                print(f"      Confidence: {alt['confidence']}")
        
        # Resolve a contradiction if any found
        if contradictions:
            contradiction = contradictions[0]
            resolution = "Water freezes at 0°C (32°F) at standard atmospheric pressure"
            print(f"\nResolving contradiction with: '{resolution}'")
            
            resolved_id = resolve_contradictions(
                memory,
                fact_id=contradiction['fact_id'],
                resolution_content=resolution,
                confidence=0.98,
                resolution_notes="Combined Celsius and Fahrenheit measurements"
            )
            
            print(f"Created resolved fact with ID: {resolved_id}")
    
    # Test fact merging
    if args.debug_type in ['merge', 'all'] and len(fact_ids) >= 2:
        print("\n=== Testing Fact Merging ===")
        
        # Merge two facts
        merge_ids = fact_ids[:2]
        print(f"Merging facts with IDs: {merge_ids}")
        
        # Get the facts to be merged
        facts_to_merge = []
        for fact_id in merge_ids:
            fact = memory.facts.get(fact_id)
            facts_to_merge.append(fact)
            print(f"  Fact to merge: '{fact['content']}'")
        
        merged_content = "The Earth orbits the Sun in 365.25 days, while water freezes at 0°C at standard pressure"
        print(f"\nMerged content: '{merged_content}'")
        
        try:
            merged_fact = merge_facts(
                memory,
                fact_ids=merge_ids,
                merged_content=merged_content,
                confidence=0.9,
                merge_notes="Combined astronomy and chemistry facts"
            )
            
            print(f"Created merged fact with ID: {merged_fact['fact_id']}")
            print(f"Merged domains: {merged_fact['domains']}")
            print(f"Merged importance: {merged_fact['importance']}")
        except Exception as e:
            print(f"Error during merge: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    debug_knowledge_correction()