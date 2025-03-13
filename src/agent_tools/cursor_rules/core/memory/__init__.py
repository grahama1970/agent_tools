#!/usr/bin/env python3
"""
Agent Memory System

This module maintains agent memory with decay and associations.
It enables an AI agent to store, recall, and manage knowledge over time.

Core features:
- Memory persistence with natural decay
- Recency boosting for recently accessed information
- Knowledge correction through fact updating
- Confidence scoring for knowledge quality
- Associative memory relationships

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
"""

# Main Memory System
from agent_tools.cursor_rules.core.memory.agent_memory import (
    AgentMemorySystem,
    MemoryFact
)

# Memory Decay Management
from agent_tools.cursor_rules.core.memory.memory_decay import (
    apply_memory_decay,
    remove_expired_memories,
    apply_targeted_decay,
    boost_memories,
    get_decay_statistics
)

# Knowledge Correction
from agent_tools.cursor_rules.core.memory.knowledge_correction import (
    update_knowledge,
    find_similar_facts,
    resolve_contradictions,
    analyze_contradictions,
    get_fact_history
)

# Confidence Scoring
from agent_tools.cursor_rules.core.memory.confidence_scoring import (
    evaluate_confidence,
    update_fact_confidence,
    batch_confidence_evaluation,
    evaluate_contradictions
)

# Associative Memory
from agent_tools.cursor_rules.core.memory.associative_memory import (
    AssociationType,
    create_association,
    get_associations
)

__all__ = [
    # Main Memory System
    'AgentMemorySystem', 'MemoryFact',
    
    # Memory Decay
    'apply_memory_decay', 'remove_expired_memories', 
    'apply_targeted_decay', 'boost_memories', 'get_decay_statistics',
    
    # Knowledge Correction
    'update_knowledge', 'find_similar_facts', 'resolve_contradictions',
    'analyze_contradictions', 'get_fact_history',
    
    # Confidence Scoring
    'evaluate_confidence', 'update_fact_confidence',
    'batch_confidence_evaluation', 'evaluate_contradictions',
    
    # Associative Memory
    'AssociationType', 'create_association', 'get_associations'
] 