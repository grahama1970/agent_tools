#!/usr/bin/env python3
"""
Associative Memory Module for Agent Memory System

This module implements associative memory functionality, allowing the system to create
and manage associations between facts. Associations can be of different types and have
different strengths.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from loguru import logger
import datetime
import asyncio

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem

class AssociationType(str, Enum):
    """Types of associations between facts"""
    SIMILAR = "similar"  # Facts are semantically similar
    RELATED = "related"  # Facts are related but not similar
    CAUSES = "causes"    # One fact causes another
    CAUSED_BY = "caused_by"  # One fact is caused by another
    IMPLIES = "implies"  # One fact implies another
    IMPLIED_BY = "implied_by"  # One fact is implied by another
    PART_OF = "part_of" # One fact is part of another
    HAS_PART = "has_part" # One fact has another as a part
    EXAMPLE = "example" # One fact is an example of another
    EXEMPLIFIED_BY = "exemplified_by" # One fact is exemplified by another
    OPPOSITE = "opposite" # Facts are opposites
    SEQUENCE = "sequence" # Facts are part of a sequence

    @classmethod
    def get_case_insensitive(cls, value: str) -> Optional['AssociationType']:
        """Get enum member by case-insensitive value"""
        try:
            value = value.lower()
            for member in cls.__members__.values():
                if member.value.lower() == value or member.name.lower() == value:
                    return member
            return None
        except (ValueError, AttributeError):
            return None

    @classmethod
    def get_inverse(cls, assoc_type: str) -> Optional[str]:
        """Get the inverse association type if it exists"""
        try:
            member = cls.get_case_insensitive(assoc_type)
            if not member:
                return None
                
            inverse_map = {
                cls.SIMILAR: cls.SIMILAR,
                cls.RELATED: cls.RELATED,
                cls.CAUSES: cls.CAUSED_BY,
                cls.CAUSED_BY: cls.CAUSES,
                cls.IMPLIES: cls.IMPLIED_BY,
                cls.IMPLIED_BY: cls.IMPLIES,
                cls.PART_OF: cls.HAS_PART,
                cls.HAS_PART: cls.PART_OF,
                cls.EXAMPLE: cls.EXEMPLIFIED_BY,
                cls.EXEMPLIFIED_BY: cls.EXAMPLE,
                cls.OPPOSITE: cls.OPPOSITE,
                cls.SEQUENCE: cls.SEQUENCE
            }
            return inverse_map[member].value
        except (ValueError, AttributeError, KeyError):
            return None

    @classmethod
    def validate(cls, assoc_type: str) -> bool:
        """Validate that an association type is valid"""
        return cls.get_case_insensitive(assoc_type) is not None

async def create_association(
    memory_system: AgentMemorySystem,
    fact_id_1: Union[str, Dict[str, Any]],
    fact_id_2: Union[str, Dict[str, Any]],
    assoc_type: str,
    strength: float = 0.5,
    bidirectional: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an association between two facts.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id_1: ID or document of the first fact
        fact_id_2: ID or document of the second fact
        assoc_type: Type of association
        strength: Association strength (0.0 to 1.0)
        bidirectional: Whether to create inverse association
        metadata: Optional metadata for the association
        
    Returns:
        Created association object
    """
    if not memory_system.initialized:
        memory_system.initialize()
        
    # Convert to lowercase and validate
    assoc_type = assoc_type.lower()
    if not AssociationType.validate(assoc_type):
        raise ValueError(f"Invalid association type: {assoc_type}")
    
    # Extract fact IDs from documents if needed
    if isinstance(fact_id_1, dict) and 'new' in fact_id_1:
        fact_id_1 = fact_id_1['new']['_key']
    if isinstance(fact_id_2, dict) and 'new' in fact_id_2:
        fact_id_2 = fact_id_2['new']['_key']
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Check if facts exist
    aql_query = f"""
    FOR fact IN {facts_collection}
        FILTER fact._key IN [@fact_id_1, @fact_id_2]
        COLLECT WITH COUNT INTO count
        RETURN count
    """
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars={
        "fact_id_1": fact_id_1,
        "fact_id_2": fact_id_2
    })
    if next(cursor) != 2:
        raise ValueError("One or both facts do not exist")
    
    # Check for existing association
    aql_query = f"""
    FOR assoc IN {associations_collection}
        FILTER assoc.fact_1 == @fact_id_1
        AND assoc.fact_2 == @fact_id_2
        AND assoc.type == @assoc_type
        RETURN assoc
    """
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars={
        "fact_id_1": fact_id_1,
        "fact_id_2": fact_id_2,
        "assoc_type": assoc_type
    })
    
    existing = next(cursor, None)
    if existing:
        # Update existing association
        aql_query = f"""
        UPDATE @assoc_id WITH {{
            strength: @strength,
            metadata: @metadata,
            last_updated: DATE_ISO8601(DATE_NOW())
        }} IN {associations_collection}
        RETURN NEW
        """
        
        cursor = memory_system.db.aql.execute(aql_query, bind_vars={
            "assoc_id": existing["_key"],
            "strength": strength,
            "metadata": metadata
        })
        association = next(cursor)
    else:
        # Create new association
        aql_query = f"""
        INSERT {{
            fact_1: @fact_id_1,
            fact_2: @fact_id_2,
            type: @assoc_type,
            strength: @strength,
            metadata: @metadata,
            created: DATE_ISO8601(DATE_NOW()),
            last_updated: DATE_ISO8601(DATE_NOW())
        }} INTO {associations_collection}
        RETURN NEW
        """
        
        cursor = memory_system.db.aql.execute(aql_query, bind_vars={
            "fact_id_1": fact_id_1,
            "fact_id_2": fact_id_2,
            "assoc_type": assoc_type,
            "strength": strength,
            "metadata": metadata
        })
        association = next(cursor)
    
    # Create inverse association if needed
    if bidirectional:
        inverse_type = AssociationType.get_inverse(assoc_type)
        if inverse_type:
            # Create inverse but prevent infinite recursion
            await create_association(
                memory_system=memory_system,
                fact_id_1=fact_id_2,
                fact_id_2=fact_id_1,
                assoc_type=inverse_type,
                strength=strength,
                bidirectional=False,  # Important: prevent recursion
                metadata=metadata
            )
    
    return association

def get_associations(
    memory_system: AgentMemorySystem,
    fact_id: str,
    direction: str = "both",
    assoc_type: Optional[str] = None,
    min_strength: float = 0.0,
    include_facts: bool = True
) -> List[Dict[str, Any]]:
    """
    Get associations for a fact.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the fact to get associations for
        direction: Direction of associations ("outbound", "inbound", or "both")
        assoc_type: Optional type of associations to filter by
        min_strength: Minimum association strength to include
        include_facts: Whether to include full fact objects
        
    Returns:
        List of association objects with connected facts
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Build direction filter
    direction_filter = ""
    if direction == "outbound":
        direction_filter = "FILTER assoc.fact_1 == @fact_id"
    elif direction == "inbound":
        direction_filter = "FILTER assoc.fact_2 == @fact_id"
    else:
        direction_filter = "FILTER assoc.fact_1 == @fact_id OR assoc.fact_2 == @fact_id"
    
    # Build type filter if needed
    type_filter = ""
    if assoc_type:
        type_filter = "FILTER assoc.type == @assoc_type"
    
    # Get associations with connected facts
    aql_query = f"""
    FOR assoc IN {associations_collection}
        {direction_filter}
        FILTER assoc.strength >= @min_strength
        {type_filter}
        LET other_fact_id = assoc.fact_1 == @fact_id ? assoc.fact_2 : assoc.fact_1
        LET other_fact = (
            FOR f IN {facts_collection}
            FILTER f._key == other_fact_id
            RETURN f
        )
        LET source_fact = (
            FOR f IN {facts_collection}
            FILTER f._key == @fact_id
            RETURN f
        )
        RETURN {{
            association: assoc,
            source_fact: source_fact[0],
            connected_fact: other_fact[0],
            direction: assoc.fact_1 == @fact_id ? "outbound" : "inbound"
        }}
    """
    
    bind_vars = {
        "fact_id": fact_id,
        "min_strength": min_strength,
        "assoc_type": assoc_type
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return list(cursor)

def delete_association(
    memory_system: AgentMemorySystem,
    assoc_id: str,
    delete_inverse: bool = True
) -> bool:
    """
    Delete an association by ID.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        assoc_id: ID of the association to delete
        delete_inverse: Whether to delete the inverse association if it exists
        
    Returns:
        True if association was deleted, False if not found
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    associations_collection = memory_system.config["associations_collection"]
    
    # Get association details first
    aql_query = f"""
    FOR assoc IN {associations_collection}
        FILTER assoc._key == @assoc_id
        RETURN assoc
    """
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars={"assoc_id": assoc_id})
    association = next(cursor, None)
    
    if not association:
        return False
    
    # Delete the association
    aql_query = f"""
    REMOVE @assoc_id IN {associations_collection}
    RETURN OLD
    """
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars={"assoc_id": assoc_id})
    deleted = next(cursor, None)
    
    if delete_inverse and deleted:
        # Find and delete inverse association
        inverse_type = AssociationType[deleted["type"]].get_inverse()
        if inverse_type:
            aql_query = f"""
            FOR assoc IN {associations_collection}
                FILTER assoc.fact_1 == @fact_2
                AND assoc.fact_2 == @fact_1
                AND assoc.type == @inverse_type
                REMOVE assoc IN {associations_collection}
                RETURN OLD
            """
            
            cursor = memory_system.db.aql.execute(aql_query, bind_vars={
                "fact_1": deleted["fact_1"],
                "fact_2": deleted["fact_2"],
                "inverse_type": inverse_type.name
            })
            next(cursor, None)
    
    return True

def traverse_associations(
    memory_system: AgentMemorySystem,
    fact_id: str,
    max_depth: int = 2,
    min_strength: float = 0.5,
    assoc_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Traverse associations starting from a fact up to a maximum depth.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the starting fact
        max_depth: Maximum traversal depth
        min_strength: Minimum association strength to traverse
        assoc_types: Optional list of association types to traverse
        
    Returns:
        List of traversal paths with associated facts
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Build type filter if needed
    type_filter = ""
    if assoc_types:
        type_filter = "FILTER assoc.type IN @assoc_types"
    
    # Traverse query with depth control
    aql_query = f"""
    FOR v, e, p IN 1..@max_depth OUTBOUND @fact_id
        GRAPH @graph_name
        FILTER e.strength >= @min_strength
        {type_filter}
        RETURN {{
            path: p.vertices[*]._key,
            facts: p.vertices,
            edges: p.edges,
            depth: LENGTH(p.edges)
        }}
    """
    
    bind_vars = {
        "fact_id": fact_id,
        "max_depth": max_depth,
        "min_strength": min_strength,
        "graph_name": "memory_graph",
        "assoc_types": assoc_types
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return list(cursor)

def find_common_associations(
    memory_system: AgentMemorySystem,
    fact_ids: List[str],
    min_strength: float = 0.5,
    assoc_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Find facts that are commonly associated with a set of input facts.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_ids: List of fact IDs to find common associations for
        min_strength: Minimum association strength
        assoc_types: Optional list of association types to consider
        
    Returns:
        List of facts with their association patterns
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Build type filter if needed
    type_filter = ""
    if assoc_types:
        type_filter = "FILTER assoc.type IN @assoc_types"
    
    # Find facts associated with all input facts
    aql_query = f"""
    LET input_facts = @fact_ids
    
    FOR fact IN {facts_collection}
        LET associations = (
            FOR input_id IN input_facts
                FOR assoc IN {associations_collection}
                    FILTER (assoc.fact_1 == input_id AND assoc.fact_2 == fact._key)
                       OR (assoc.fact_2 == input_id AND assoc.fact_1 == fact._key)
                    FILTER assoc.strength >= @min_strength
                    {type_filter}
                    RETURN assoc
        )
        
        FILTER LENGTH(associations) == LENGTH(input_facts)
        
        RETURN {{
            fact: fact,
            associations: associations,
            connection_strength: AVERAGE(associations[*].strength)
        }}
    """
    
    bind_vars = {
        "fact_ids": fact_ids,
        "min_strength": min_strength,
        "assoc_types": assoc_types
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return list(cursor)

async def create_semantic_associations(
    memory_system: AgentMemorySystem,
    fact_id: str,
    min_similarity: float = 0.7,
    max_associations: int = 5,
    assoc_type: str = AssociationType.SIMILAR,
    bidirectional: bool = True
) -> List[Dict[str, Any]]:
    """
    Create associations between a fact and semantically similar facts.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the fact to find associations for
        min_similarity: Minimum semantic similarity threshold
        max_associations: Maximum number of associations to create
        assoc_type: Type of association to create
        bidirectional: Whether to create bidirectional associations
        
    Returns:
        List of created associations
    """
    if not memory_system.initialized:
        await memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    
    # Get the source fact's embedding
    aql_query = f"""
    FOR fact IN {facts_collection}
        FILTER fact._key == @fact_id
        RETURN fact
    """
    
    bind_vars = {
        "fact_id": fact_id
    }
    
    cursor = await asyncio.to_thread(memory_system.db.aql.execute, aql_query, bind_vars=bind_vars)
    source_fact = await asyncio.to_thread(next, cursor, None)
    
    if not source_fact or "embedding" not in source_fact:
        return []
    
    # Find semantically similar facts
    search_query = f"""
    FOR fact IN {facts_collection}
        FILTER fact._key != @fact_id
        FILTER HAS(fact, "embedding")
        
        LET similarity = COSINE_SIMILARITY(fact.embedding, @embedding)
        FILTER similarity >= @min_similarity
        
        SORT similarity DESC
        LIMIT @max_associations
        
        RETURN {{
            fact: fact,
            similarity: similarity
        }}
    """
    
    bind_vars = {
        "fact_id": fact_id,
        "embedding": source_fact["embedding"],
        "min_similarity": min_similarity,
        "max_associations": max_associations
    }
    
    cursor = await asyncio.to_thread(memory_system.db.aql.execute, search_query, bind_vars=bind_vars)
    similar_facts = [result async for result in cursor]
    
    # Create associations
    created_associations = []
    for result in similar_facts:
        target_fact = result["fact"]
        similarity = result["similarity"]
        
        assoc = await create_association(
            memory_system=memory_system,
            fact_id_1=fact_id,
            fact_id_2=target_fact["_key"],
            assoc_type=assoc_type,
            strength=similarity,
            bidirectional=bidirectional,
            metadata={"semantic_similarity": similarity}
        )
        
        created_associations.append(assoc)
    
    return created_associations

def find_path(
    memory_system: AgentMemorySystem,
    start_fact_id: str,
    end_fact_id: str,
    max_depth: int = 3,
    min_strength: float = 0.5,
    assoc_types: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Find a path between two facts through associations.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        start_fact_id: Starting fact ID
        end_fact_id: Target fact ID
        max_depth: Maximum path length
        min_strength: Minimum association strength
        assoc_types: Optional list of association types to traverse
        
    Returns:
        Path details if found, None otherwise
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Build type filter if needed
    type_filter = ""
    if assoc_types:
        type_filter = "FILTER assoc.type IN @assoc_types"
    
    # Find shortest path query
    aql_query = f"""
    FOR v, e, p IN 1..@max_depth ANY @start_id TO @end_id
        GRAPH @graph_name
        FILTER ALL(assoc IN p.edges[*] | assoc.strength >= @min_strength)
        {type_filter}
        SORT LENGTH(p.edges)
        LIMIT 1
        RETURN {{
            path: p.vertices[*]._key,
            facts: p.vertices,
            edges: p.edges,
            length: LENGTH(p.edges),
            total_strength: AVERAGE(p.edges[*].strength)
        }}
    """
    
    bind_vars = {
        "start_id": start_fact_id,
        "end_id": end_fact_id,
        "max_depth": max_depth,
        "min_strength": min_strength,
        "graph_name": "memory_graph",
        "assoc_types": assoc_types
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return next(cursor, None)

def find_clusters(
    memory_system: AgentMemorySystem,
    min_strength: float = 0.7,
    min_cluster_size: int = 3,
    assoc_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Find clusters of strongly associated facts.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        min_strength: Minimum association strength within clusters
        min_cluster_size: Minimum number of facts in a cluster
        assoc_types: Optional list of association types to consider
        
    Returns:
        List of clusters with their facts and associations
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    facts_collection = memory_system.config["facts_collection"]
    associations_collection = memory_system.config["associations_collection"]
    
    # Build type filter if needed
    type_filter = ""
    if assoc_types:
        type_filter = "FILTER assoc.type IN @assoc_types"
    
    # Find clusters using graph traversal
    aql_query = f"""
    FOR fact IN {facts_collection}
        LET cluster = (
            FOR v, e, p IN 1..3 ANY fact._key
                GRAPH @graph_name
                FILTER e.strength >= @min_strength
                {type_filter}
                RETURN DISTINCT v
        )
        
        FILTER LENGTH(cluster) >= @min_size
        
        LET cluster_associations = (
            FOR fact1 IN cluster
                FOR fact2 IN cluster
                    FILTER fact1._key < fact2._key
                    FOR assoc IN {associations_collection}
                        FILTER (assoc.fact_1 == fact1._key AND assoc.fact_2 == fact2._key)
                           OR (assoc.fact_2 == fact1._key AND assoc.fact_1 == fact2._key)
                        FILTER assoc.strength >= @min_strength
                        {type_filter}
                        RETURN assoc
        )
        
        FILTER LENGTH(cluster_associations) >= LENGTH(cluster) - 1
        
        RETURN {{
            facts: cluster,
            associations: cluster_associations,
            size: LENGTH(cluster),
            avg_strength: AVERAGE(cluster_associations[*].strength)
        }}
    """
    
    bind_vars = {
        "min_strength": min_strength,
        "min_size": min_cluster_size,
        "graph_name": "memory_graph",
        "assoc_types": assoc_types
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return list(cursor)

def prune_associations(
    memory_system: AgentMemorySystem,
    min_strength: float = 0.3,
    older_than: Optional[datetime.datetime] = None,
    assoc_types: Optional[List[str]] = None
) -> int:
    """
    Remove weak or outdated associations.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        min_strength: Remove associations weaker than this
        older_than: Optional datetime to remove older associations
        assoc_types: Optional list of association types to consider
        
    Returns:
        Number of associations removed
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    associations_collection = memory_system.config["associations_collection"]
    
    # Build filters
    filters = ["assoc.strength < @min_strength"]
    if older_than:
        filters.append("DATE_TIMESTAMP(assoc.last_updated) < @cutoff_time")
    if assoc_types:
        filters.append("assoc.type IN @assoc_types")
    
    filter_str = " AND ".join(filters)
    
    # Remove weak/old associations
    aql_query = f"""
    FOR assoc IN {associations_collection}
        FILTER {filter_str}
        REMOVE assoc IN {associations_collection}
        COLLECT WITH COUNT INTO removed
        RETURN removed
    """
    
    bind_vars = {
        "min_strength": min_strength,
        "cutoff_time": older_than.timestamp() if older_than else None,
        "assoc_types": assoc_types
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return next(cursor, 0)

def strengthen_association(
    memory_system: AgentMemorySystem,
    assoc_id: str,
    amount: float = 0.1,
    max_strength: float = 1.0
) -> Dict[str, Any]:
    """
    Increase the strength of an association.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        assoc_id: ID of the association to strengthen
        amount: Amount to increase strength by
        max_strength: Maximum allowed strength
        
    Returns:
        Updated association
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    associations_collection = memory_system.config["associations_collection"]
    
    # Update association strength
    aql_query = f"""
    FOR assoc IN {associations_collection}
        FILTER assoc._key == @assoc_id
        UPDATE assoc WITH {{
            strength: (assoc.strength + @amount > @max_strength ? @max_strength : assoc.strength + @amount),
            last_updated: DATE_ISO8601(DATE_NOW())
        }} IN {associations_collection}
        RETURN NEW
    """
    
    bind_vars = {
        "assoc_id": assoc_id,
        "amount": amount,
        "max_strength": max_strength
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return next(cursor)