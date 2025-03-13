#!/usr/bin/env python3
"""
Knowledge Correction Module for Agent Memory System

This module implements the knowledge correction functionality for the agent memory system.
It handles detecting contradictions, updating existing knowledge, and maintaining a history
of knowledge changes.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
"""

from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import datetime

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem, MemoryFact

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
        fact_id = memory_system.remember(
            content=fact_content,
            confidence=confidence,
            domains=domains,
            ttl_days=ttl_days,
            importance=importance,
            source=source
        )
        return fact_id, True, None
    
    # Get the most similar fact
    most_similar = similar_facts[0]
    fact_id = most_similar["_key"]
    
    # Check if confidence in new fact is higher or equal
    if confidence >= most_similar.get("confidence", 0.0):
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
        memory_system.remember(
            fact_id=fact_id,
            content=fact_content,
            confidence=confidence,
            domains=merged_domains,
            ttl_days=ttl_days if ttl_days > most_similar.get("ttl_days", 0) else most_similar.get("ttl_days", 0),
            importance=new_importance,
            source=source,
            correction_history=correction_history
        )
        
        return fact_id, False, previous_version
    
    # If our confidence is lower, keep the existing fact but record the alternative
    else:
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
        memory_system.remember(
            fact_id=fact_id,
            alternatives=alternatives
        )
        
        return fact_id, False, None

def find_similar_facts(
    memory_system: AgentMemorySystem,
    fact_content: str,
    similarity_threshold: float = 0.85,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find facts similar to the given content using vector similarity.
    
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
    embedding = memory_system._get_embedding(fact_content)
    
    collection_name = memory_system.config["facts_collection"]
    # Use the vector index to find similar facts
    aql_query = f"""
    FOR fact IN {collection_name}
        LET similarity = VECTOR_DISTANCE(fact.embedding, @embedding)
        FILTER similarity >= @threshold
        SORT similarity DESC
        LIMIT @limit
        RETURN MERGE(fact, {{ similarity_score: similarity }})
    """
    
    bind_vars = {
        "embedding": embedding,
        "threshold": similarity_threshold,
        "limit": limit
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return list(cursor)

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
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER fact._key == @fact_id
        RETURN {{
            current: {{
                content: fact.content,
                confidence: fact.confidence,
                domains: fact.domains,
                importance: fact.importance,
                ttl_days: fact.ttl_days,
                created_at: fact.created_at,
                last_updated: fact.last_updated,
                access_count: fact.access_count,
                last_accessed: fact.last_accessed
            }},
            correction_history: fact.correction_history || [],
            alternatives: fact.alternatives || [],
            source: fact.source
        }}
    """
    
    bind_vars = {
        "fact_id": fact_id
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    result = next(cursor, None)
    if not result:
        raise ValueError(f"Fact with ID {fact_id} not found")
    
    return result

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