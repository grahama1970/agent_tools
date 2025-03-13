# Agent Memory System

A sophisticated memory system for AI agents that implements memory persistence, decay, recency boosting, knowledge correction, confidence scoring, and associative relationships.

## Features

- **Memory Persistence**: Store facts with importance, confidence, and domain tags
- **Memory Decay**: Important information persists longer, less relevant details fade over time
- **Recency Boosting**: Recently accessed information receives priority in recall
- **Knowledge Correction**: Update existing knowledge when new information arrives
- **Confidence Scoring**: Assess and update confidence in facts based on evidence
- **Associative Memory**: Create and traverse relationships between related facts

## Architecture

The memory system is built on ArangoDB, using both document collections and graph capabilities:

- `facts_collection`: Stores individual memory facts
- `associations_collection`: Stores relationships between facts (edge collection)

Vector embeddings are used for semantic search and similarity detection.

## Usage Examples

### Basic Initialization and Memory Storage

```python
import asyncio
from agent_tools.cursor_rules.core.memory import AgentMemorySystem

async def main():
    # Initialize the memory system
    config = {
        "arango_host": "http://localhost:8529",
        "username": "root",
        "password": "",
        "db_name": "agent_memory",
        "facts_collection": "agent_facts",
        "associations_collection": "fact_associations",
        "embeddings_model": "all-MiniLM-L6-v2"  # Or any other sentence transformer model
    }
    
    memory = AgentMemorySystem(config)
    await memory.initialize(create_db=True)
    
    # Remember a fact
    fact_id = await memory.remember(
        content="The capital of France is Paris.",
        importance=0.7,
        confidence=0.95,
        domains=["geography", "europe"],
        ttl_days=730.0  # Fact will persist for ~2 years without access
    )
    
    # Recall facts
    results = await memory.recall("What is the capital of France?")
    for fact in results:
        print(f"Fact: {fact['content']}")
        print(f"Confidence: {fact['confidence']}")
    
    # Cleanup
    await memory.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Memory Decay Management

```python
from agent_tools.cursor_rules.core.memory import (
    AgentMemorySystem, apply_memory_decay, boost_memories
)

async def decay_example():
    # Initialize memory system
    memory = AgentMemorySystem(config)
    await memory.initialize()
    
    # Apply memory decay (simulate the passage of time)
    decay_stats = await apply_memory_decay(
        memory_system=memory,
        days_passed=30.0,
        domains_to_preserve=["critical_info"]  # Protect important domains
    )
    
    print(f"Decay statistics: {decay_stats}")
    
    # Boost memories related to a topic
    boosted = await boost_memories(
        memory_system=memory,
        query="artificial intelligence",
        boost_amount=60.0,  # Add 60 days to TTL
        limit=10
    )
    
    print(f"Boosted {boosted} memories related to AI")
```

### Knowledge Correction

```python
from agent_tools.cursor_rules.core.memory import (
    AgentMemorySystem, update_knowledge, resolve_contradictions
)

async def knowledge_correction_example():
    memory = AgentMemorySystem(config)
    await memory.initialize()
    
    # Update knowledge (will create new or update existing fact)
    fact_id, is_new, previous = await update_knowledge(
        memory_system=memory,
        fact_content="GPT-4 was released by OpenAI in March 2023.",
        confidence=0.9,
        domains=["ai", "llm", "history"]
    )
    
    if not is_new and previous:
        print(f"Updated existing fact. Previous version: {previous['content']}")
    
    # Resolve contradictions
    contradictions = await analyze_contradictions(memory)
    
    if contradictions:
        # Resolve the first contradiction
        await resolve_contradictions(
            memory_system=memory,
            fact_id=contradictions[0]["fact_id"],
            resolution_content="Combined and corrected version of the fact.",
            confidence=0.95,
            resolution_notes="Resolved based on latest information."
        )
```

### Associative Memory

```python
from agent_tools.cursor_rules.core.memory import (
    AgentMemorySystem, Association, create_association, traverse_associations
)

async def associative_memory_example():
    memory = AgentMemorySystem(config)
    await memory.initialize()
    
    # Store some related facts
    python_id = await memory.remember(
        content="Python is a high-level programming language.",
        importance=0.7,
        confidence=0.9,
        domains=["programming"]
    )
    
    tensorflow_id = await memory.remember(
        content="TensorFlow is a machine learning framework developed by Google.",
        importance=0.6,
        confidence=0.9,
        domains=["machine_learning", "programming"]
    )
    
    # Create association between facts
    await create_association(
        memory_system=memory,
        source_id=python_id,
        target_id=tensorflow_id,
        assoc_type=Association.RELATED_TO,
        bidirectional=True,
        weight=0.8,
        metadata={"reason": "Python is often used with TensorFlow"}
    )
    
    # Traverse the association graph
    related_facts = await traverse_associations(
        memory_system=memory,
        start_fact_id=python_id,
        max_depth=2
    )
    
    # Generate semantic associations automatically
    await create_semantic_associations(
        memory_system=memory,
        fact_id=python_id,
        similarity_threshold=0.7,
        max_associations=5
    )
```

### Confidence Scoring

```python
from agent_tools.cursor_rules.core.memory import (
    AgentMemorySystem, evaluate_confidence, update_fact_confidence
)

async def confidence_scoring_example():
    memory = AgentMemorySystem(config)
    await memory.initialize()
    
    # Evaluate confidence based on evidence
    fact_content = "The next total solar eclipse visible from North America will occur on April 8, 2024."
    evidence = [
        "NASA confirms a total solar eclipse will cross North America on April 8, 2024.",
        "The path of totality will span from Mexico through the US to Canada.",
        "The next total solar eclipse visible from parts of North America after 2024 will be in 2045."
    ]
    
    confidence, reasoning = await evaluate_confidence(
        fact_content=fact_content,
        evidence=evidence,
        model="gpt-3.5-turbo"  # Using LiteLLM
    )
    
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")
    
    # Update fact confidence based on new evidence
    fact_id = "existing-fact-id"
    result = await update_fact_confidence(
        memory_system=memory,
        fact_id=fact_id,
        evidence=evidence
    )
    
    if result["updated"]:
        print(f"Updated confidence from {result['original_confidence']} to {result['evaluated_confidence']}")
```

## Testing

The memory system includes comprehensive tests that validate all functionality:

```bash
# Run all memory system tests
pytest -xvs src/agent_tools/cursor_rules/tests/memory/

# Run specific test module
pytest -xvs src/agent_tools/cursor_rules/tests/memory/test_memory_decay.py
```

## Requirements

- Python 3.8+
- ArangoDB 3.9+
- Dependencies: 
  - python-arango
  - sentence-transformers (for embeddings)
  - pydantic
  - loguru
  - litellm (optional, for confidence scoring)

## Implementation Notes

- All database operations are implemented asynchronously using `asyncio.to_thread`
- The system maintains its own connection pool to ArangoDB
- Vector embeddings are generated using the specified embedding model
- The system is designed to handle concurrent access safely 