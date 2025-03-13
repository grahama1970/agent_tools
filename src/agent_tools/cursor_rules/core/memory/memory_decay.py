#!/usr/bin/env python3
"""
Memory Decay Module for Agent Memory System

This module implements the decay functionality for the agent memory system.
Important information persists longer, while less relevant details fade over time.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem

def apply_memory_decay(
    memory_system: AgentMemorySystem,
    days_passed: float = 1.0,
    importance_factor: Optional[float] = None,
    domains_to_preserve: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Apply memory decay to all facts in the system.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        days_passed: Number of days to simulate in the decay process
        importance_factor: Override for the importance factor (uses system default if None)
        domains_to_preserve: Optional list of domains to protect from decay
        
    Returns:
        Statistics about the decay operation
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    # Use system default if not provided
    if importance_factor is None:
        importance_factor = memory_system.config["importance_decay_factor"]
    
    collection_name = memory_system.config["facts_collection"]
    # Decay algorithm implemented in AQL with domain protection
    aql_query = f"""
    LET day_factor = @day_factor
    
    FOR fact IN {collection_name}
        // Check if fact is in a protected domain
        LET is_protected_domain = @domains_to_preserve != null AND 
                                 LENGTH(INTERSECTION(fact.domains, @domains_to_preserve)) > 0
        
        // Check if fact is temporary
        LET is_temporary = "temporary" IN fact.domains
        
        // Apply higher day factor for temporary facts
        LET effective_day_factor = is_temporary ? day_factor * 3.0 : day_factor
        
        // Calculate decay factors
        LET importance_protection = fact.importance * @importance_factor
        LET access_protection = (fact.access_count / 10.0 < 1.0 ? fact.access_count / 10.0 : 1.0) * 0.5
        // Add domain protection if applicable, or penalty if temporary
        LET domain_protection = is_protected_domain ? 0.8 : (is_temporary ? -0.5 : 0)
        LET total_protection = importance_protection + access_protection + domain_protection
        
        // Calculate TTL reduction (less protection = faster decay)
        // Protected domains decay much slower
        LET ttl_reduction = effective_day_factor * (1.0 - (total_protection < 0.95 ? total_protection : 0.95))
        
        // Calculate new TTL but never below 0
        LET new_ttl = fact.ttl_days - ttl_reduction > 0 ? fact.ttl_days - ttl_reduction : 0
        
        // Update the fact with new TTL
        UPDATE fact WITH {{ 
            ttl_days: new_ttl,
            decay_factor: total_protection,  // Store for analysis
            decay_details: {{
                initial_ttl: fact.ttl_days,
                importance_protection: importance_protection,
                access_protection: access_protection,
                domain_protection: domain_protection,
                total_protection: total_protection,
                ttl_reduction: ttl_reduction,
                new_ttl: new_ttl
            }}
        }} IN {collection_name}
        
        // Return stats for different decay categories
        COLLECT decay_status = 
            new_ttl <= 0 ? "expired" : 
            new_ttl < fact.ttl_days * 0.5 ? "critical" :
            new_ttl < fact.ttl_days * 0.8 ? "decaying" : 
            "stable"
        
        RETURN {{
            status: decay_status,
            count: LENGTH(1)
        }}
    """
    
    bind_vars = {
        "importance_factor": importance_factor,
        "day_factor": days_passed,
        "domains_to_preserve": domains_to_preserve
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    results = list(cursor)
    
    # Format results as a statistics dictionary
    stats = {status["status"]: status["count"] for status in results}
    
    # Get detailed decay info for debugging
    debug_query = f"""
    FOR fact IN {collection_name}
        RETURN {{
            content: fact.content,
            ttl_days: fact.ttl_days,
            decay_details: fact.decay_details
        }}
    """
    
    debug_cursor = memory_system.db.aql.execute(debug_query)
    debug_results = list(debug_cursor)
    
    logger.info("Decay results:")
    for fact in debug_results:
        logger.info(f"Content: {fact['content']}")
        logger.info(f"TTL: {fact['ttl_days']}")
        logger.info(f"Decay details: {fact['decay_details']}\n")
    
    # Delete expired memories if needed
    if stats.get("expired", 0) > 0:
        removed_count = remove_expired_memories(memory_system)
        stats["removed"] = removed_count
    
    return stats

def remove_expired_memories(memory_system: AgentMemorySystem) -> int:
    """
    Remove memories with TTL <= 0.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        
    Returns:
        Number of removed memories
    """
    collection_name = memory_system.config["facts_collection"]
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER fact.ttl_days <= 0
        REMOVE fact IN {collection_name}
        RETURN 1
    """
    
    cursor = memory_system.db.aql.execute(aql_query)
    return len(list(cursor))

def apply_targeted_decay(
    memory_system: AgentMemorySystem,
    domain: str,
    factor: float = 2.0
) -> int:
    """
    Apply targeted decay to a specific domain of memories.
    Useful for when certain types of information become obsolete more quickly.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        domain: The domain to target for accelerated decay
        factor: Multiplier for decay rate (higher = faster decay)
        
    Returns:
        Number of affected memories
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    aql_query = f"""
    FOR fact IN {collection_name}
        FILTER @domain IN fact.domains
        
        // Calculate accelerated TTL reduction
        LET ttl_reduction = @factor * (1.0 - fact.importance * 0.5)
        LET new_ttl = (fact.ttl_days - ttl_reduction > 0 ? fact.ttl_days - ttl_reduction : 0)
        
        // Update the fact with new TTL
        UPDATE fact WITH {{ 
            ttl_days: new_ttl,
            last_decay: {{
                type: "targeted",
                domain: @domain,
                factor: @factor,
                date: DATE_ISO8601(DATE_NOW())
            }}
        }} IN {collection_name}
        
        RETURN 1
    """
    
    bind_vars = {
        "domain": domain,
        "factor": factor
    }
    
    cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
    return len(list(cursor))

def boost_memories(
    memory_system: AgentMemorySystem,
    query: str,
    boost_amount: float = 30.0,
    limit: int = 10
) -> int:
    """
    Boost TTL for memories that match a query, extending their lifespan.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        query: Search query to find memories to boost
        boost_amount: Amount of days to add to TTL
        limit: Maximum number of memories to boost
        
    Returns:
        Number of boosted memories
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    view_name = f"{collection_name}_view"
    
    # Find relevant memories using text search
    aql_query = f"""
    FOR fact IN {view_name}
        SEARCH ANALYZER(
            fact.content IN TOKENS(@query, "text_en"),
            "text_en"
        )
        
        // Boost the TTL
        UPDATE fact WITH {{ 
            ttl_days: fact.ttl_days + @boost_amount,
            importance: (fact.importance + 0.1 < 1.0 ? fact.importance + 0.1 : 1.0),
            boost_history: APPEND(
                fact.boost_history || [],
                {{
                    date: DATE_ISO8601(DATE_NOW()),
                    query: @query,
                    amount: @boost_amount
                }}
            )
        }} IN {collection_name}
        
        LIMIT @limit
        RETURN 1
    """
    
    bind_vars = {
        "query": query,
        "boost_amount": boost_amount,
        "limit": limit
    }
    
    try:
        cursor = memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
        return len(list(cursor))
    except Exception as e:
        logger.error(f"Error boosting memories: {e}")
        return 0

def get_decay_statistics(memory_system: AgentMemorySystem) -> Dict[str, Any]:
    """
    Get statistics about the decay status of memories in the system.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        
    Returns:
        Dictionary with decay statistics
    """
    if not memory_system.initialized:
        memory_system.initialize()
    
    collection_name = memory_system.config["facts_collection"]
    
    # Simplified query that categorizes each fact exactly once
    aql_query = f"""
    LET total_count = LENGTH(FOR doc IN {collection_name} RETURN 1)
    
    // Query each category separately with non-overlapping conditions
    LET expired_facts = (
        FOR fact IN {collection_name}
            FILTER fact.ttl_days <= 0
            RETURN fact
    )
    
    LET critical_facts = (
        FOR fact IN {collection_name}
            FILTER fact.ttl_days > 0 AND fact.ttl_days < 7
            RETURN fact
    )
    
    LET short_term_facts = (
        FOR fact IN {collection_name}
            FILTER fact.ttl_days >= 7 AND fact.ttl_days < 30
            RETURN fact
    )
    
    LET medium_term_facts = (
        FOR fact IN {collection_name}
            FILTER fact.ttl_days >= 30 AND fact.ttl_days < 90
            RETURN fact
    )
    
    LET long_term_facts = (
        FOR fact IN {collection_name}
            FILTER fact.ttl_days >= 90
            RETURN fact
    )
    
    // Verify all facts are accounted for
    LET total_categorized = LENGTH(expired_facts) + LENGTH(critical_facts) + 
                           LENGTH(short_term_facts) + LENGTH(medium_term_facts) + 
                           LENGTH(long_term_facts)
    
    LET ttl_categories = [
        {"category": "expired", "count": LENGTH(expired_facts), "percentage": ROUND(LENGTH(expired_facts) * 100.0 / total_count)},
        {"category": "critical", "count": LENGTH(critical_facts), "percentage": ROUND(LENGTH(critical_facts) * 100.0 / total_count)},
        {"category": "short_term", "count": LENGTH(short_term_facts), "percentage": ROUND(LENGTH(short_term_facts) * 100.0 / total_count)},
        {"category": "medium_term", "count": LENGTH(medium_term_facts), "percentage": ROUND(LENGTH(medium_term_facts) * 100.0 / total_count)},
        {"category": "long_term", "count": LENGTH(long_term_facts), "percentage": ROUND(LENGTH(long_term_facts) * 100.0 / total_count)}
    ]
    
    LET domain_stats = (
        FOR fact IN {collection_name}
            FOR domain IN fact.domains
                COLLECT d = domain
                AGGREGATE avg_ttl = AVG(fact.ttl_days),
                          avg_importance = AVG(fact.importance),
                          count = LENGTH(1)
                
                RETURN {
                    domain: d,
                    count: count,
                    percentage: ROUND(count * 100.0 / total_count)
                }
    )
    
    RETURN {
        total_facts: total_count,
        total_categorized: total_categorized,
        verification_passed: total_count == total_categorized,
        ttl_categories: ttl_categories,
        domain_stats: domain_stats,
        timestamp: DATE_ISO8601(DATE_NOW())
    }
    """
    
    try:
        cursor = memory_system.db.aql.execute(aql_query)
        stats = next(cursor, None)
        
        # Add explicit verification and logging
        if stats and stats.get("verification_passed", False) is False:
            logger.error(f"Fact accounting mismatch: {stats['total_facts']} total facts but {stats['total_categorized']} categorized")
        
        # Log the TTL categories for debugging
        logger.info("TTL Categories:")
        for category in stats["ttl_categories"]:
            logger.info(f"{category['category']}: {category['count']} facts ({category['percentage']}%)")
        
        return stats
    except Exception as e:
        logger.error(f"Error getting decay statistics: {e}")
        return {
            "error": str(e),
            "total_facts": 0,
            "ttl_categories": [],
            "domain_stats": [],
            "verification_passed": False
        } 