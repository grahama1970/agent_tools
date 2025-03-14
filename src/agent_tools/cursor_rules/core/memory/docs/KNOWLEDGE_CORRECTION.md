# Knowledge Correction System

The Knowledge Correction System is an enhancement layer built on top of the Agent Memory System that provides sophisticated knowledge management capabilities. It enables AI agents to update existing knowledge, detect and resolve contradictions, and merge related facts into more comprehensive information.

## Overview

The system handles:
- **Knowledge Updates:**  
  Checks for similar facts using vector similarity and updates them based on confidence levels.
  
- **Contradiction Detection:**  
  Identifies facts with contradicting alternatives and provides a mechanism to resolve them.
  
- **Fact Merging:**  
  Combines multiple related facts into a single, more comprehensive fact while preserving relationships.
  
- **History Tracking:**  
  Maintains a history of corrections and alternatives for each fact, enabling knowledge provenance.

## Relationship with Agent Memory System

The Knowledge Correction System builds on top of the core Agent Memory System:

1. **Foundation Layer:**  
   `agent_memory.py` provides the basic memory storage, retrieval, and management functions, handling direct interactions with the ArangoDB database.

2. **Enhancement Layer:**  
   `knowledge_correction.py` adds higher-level knowledge management capabilities without needing to understand the underlying storage details.

3. **Integration Points:**
   - Takes an initialized `AgentMemorySystem` instance as a parameter for all functions
   - Uses `_get_embedding` from `agent_memory.py` for consistent embedding generation
   - Calls `remember` from `agent_memory.py` to store and update facts
   - Extends basic storage with additional metadata like correction history

## Core Components

### 1. Knowledge Updates (`update_knowledge()`)

The `update_knowledge` function checks for similar facts and decides whether to update an existing fact or create a new one:

```python
fact_id, is_new, previous = update_knowledge(
    memory_system,
    fact_content="The Earth completes one orbit around the Sun in 365.25 days",
    similarity_threshold=0.7,
    confidence=0.92,
    domains=["astronomy", "physics"]
)
```

- If no similar facts are found, it creates a new fact
- If a similar fact is found and the new fact has higher confidence, it updates the existing fact
- If a similar fact is found but the new fact has lower confidence, it adds the new fact as an alternative

### 2. Finding Similar Facts (`find_similar_facts()`)

The `find_similar_facts` function uses vector similarity to find facts similar to a given content:

```python
similar_facts = find_similar_facts(
    memory_system,
    fact_content="The Earth orbits the Sun in approximately 365 days",
    similarity_threshold=0.85,
    limit=5
)
```

- Generates an embedding for the fact content using `_get_embedding` from `agent_memory.py`
- Uses vector distance to find similar facts in the database
- Returns a list of similar facts with similarity scores

### 3. Fact History Retrieval (`get_fact_history()`)

The `get_fact_history` function retrieves the full history of a fact, including all corrections and alternatives:

```python
history = get_fact_history(memory_system, fact_id)
```

- Returns the current content, confidence, domains, and other metadata
- Includes the correction history (previous versions)
- Includes alternatives (contradicting facts with lower confidence)

### 4. Contradiction Analysis (`analyze_contradictions()`)

The `analyze_contradictions` function identifies facts with contradicting alternatives:

```python
contradictions = analyze_contradictions(memory_system)
```

- Finds facts with alternatives that haven't been resolved
- Calculates a contradiction score based on the number of alternatives and confidence level
- Returns a list of facts with contradictions, sorted by contradiction score

### 5. Contradiction Resolution (`resolve_contradictions()`)

The `resolve_contradictions` function resolves contradictions by creating a new unified fact:

```python
resolved_id = resolve_contradictions(
    memory_system,
    fact_id="fact_123",
    resolution_content="Water freezes at 0°C (32°F) at standard atmospheric pressure",
    confidence=0.98,
    resolution_notes="Combined Celsius and Fahrenheit measurements"
)
```

- Creates a new fact with the resolution content
- Marks the original fact as resolved
- Preserves the relationship between the original fact and the resolution

### 6. Fact Merging (`merge_facts()`)

The `merge_facts` function combines multiple related facts into a single, more comprehensive fact:

```python
merged_fact = merge_facts(
    memory_system,
    fact_ids=["fact_123", "fact_456"],
    merged_content="The Earth orbits the Sun in 365.25 days, while water freezes at 0°C at standard pressure",
    confidence=0.9,
    merge_notes="Combined astronomy and chemistry facts"
)
```

- Combines domains from all merged facts
- Uses the highest importance plus a small boost
- Creates a new fact with the merged content
- Marks original facts as merged
- Preserves relationships between original facts and the merged fact

## Usage Example

Here's a complete example of how to use the Knowledge Correction System:

```python
from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem
from agent_tools.cursor_rules.core.memory.knowledge_correction import (
    update_knowledge, find_similar_facts, get_fact_history,
    analyze_contradictions, resolve_contradictions, merge_facts
)

# Initialize memory system
memory = AgentMemorySystem()
memory.initialize()

# Add initial facts
fact1_id = memory.remember(
    content="The Earth orbits the Sun in approximately 365 days",
    confidence=0.9,
    domains=["astronomy", "science"],
    importance=0.8
)["new"]["fact_id"]

fact2_id = memory.remember(
    content="Water freezes at 0 degrees Celsius at standard pressure",
    confidence=0.95,
    domains=["chemistry", "science"],
    importance=0.7
)["new"]["fact_id"]

# Update knowledge with a more precise fact
updated_fact = "The Earth completes one orbit around the Sun in 365.25 days"
fact_id, is_new, previous = update_knowledge(
    memory,
    fact_content=updated_fact,
    similarity_threshold=0.7,
    confidence=0.92,
    domains=["astronomy", "physics"]
)

# Add a contradicting fact with lower confidence
contradicting_fact = "Water freezes at 32 degrees Fahrenheit at standard pressure"
fact_id, is_new, previous = update_knowledge(
    memory,
    fact_content=contradicting_fact,
    similarity_threshold=0.7,
    confidence=0.85,
    domains=["chemistry", "science"]
)

# Analyze contradictions
contradictions = analyze_contradictions(memory)
if contradictions:
    # Resolve a contradiction
    contradiction = contradictions[0]
    resolution = "Water freezes at 0°C (32°F) at standard atmospheric pressure"
    resolved_id = resolve_contradictions(
        memory,
        fact_id=contradiction["fact_id"],
        resolution_content=resolution,
        confidence=0.98,
        resolution_notes="Combined Celsius and Fahrenheit measurements"
    )

# Merge related facts
merged_fact = merge_facts(
    memory,
    fact_ids=[fact1_id, fact2_id],
    merged_content="The Earth orbits the Sun in 365.25 days, while water freezes at 0°C at standard pressure",
    confidence=0.9,
    merge_notes="Combined astronomy and chemistry facts"
)
```

## Debugging

The Knowledge Correction System includes a debug utility function `debug_knowledge_correction()` that demonstrates the system's capabilities:

```bash
# Run the debug utility
python knowledge_correction.py --debug_type all

# Test specific functionality
python knowledge_correction.py --debug_type update
python knowledge_correction.py --debug_type contradiction
python knowledge_correction.py --debug_type merge

# Adjust similarity threshold
python knowledge_correction.py --similarity 0.8
```

The debug utility:
1. Initializes a memory system with test data
2. Adds initial facts
3. Tests updating knowledge with similar facts
4. Demonstrates contradiction detection and resolution
5. Shows fact merging

## Best Practices

1. **Confidence Levels:**
   - Use higher confidence for well-established facts
   - Use lower confidence for uncertain or speculative information
   - Consider the source reliability when setting confidence

2. **Similarity Thresholds:**
   - Use higher thresholds (0.8-0.9) for strict matching
   - Use lower thresholds (0.6-0.7) for more flexible matching
   - Adjust based on your specific use case and embedding model

3. **Domain Management:**
   - Use consistent domain names across facts
   - Include both specific and general domains (e.g., "physics" and "science")
   - Consider domain hierarchies when merging facts

4. **Contradiction Resolution:**
   - Resolve contradictions promptly to maintain knowledge consistency
   - Include both perspectives in the resolution when appropriate
   - Add resolution notes to explain the reasoning

5. **Fact Merging:**
   - Only merge facts that are truly related
   - Ensure the merged content accurately represents all original facts
   - Use merge notes to explain the rationale for merging 