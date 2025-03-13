# Agent Memory System

The Agent Memory System is a core component designed to help AI agents manage a dynamic, persistent store of "facts" with additional features such as memory decay, hybrid search, and associative (graph) relationships.

## Overview

The system handles:
- **Memory Persistence:**  
  Facts are stored in an ArangoDB collection along with metadata such as importance, confidence, time-to-live (TTL), and domains.
  
- **Hybrid Search:**  
  The system combines semantic similarity (via vector embeddings), BM25 text search, and glossary matching to recall the most relevant facts for a given query.
  
- **Domain (Keyword) Search:**  
  A separate search option matches query tokens against fact domains (e.g., "physics", "geography") and can boost certain facts based on pre-configured preserved domains.
  
- **Memory Decay:**  
  Over time, facts "decay" (i.e., their TTL is reduced based on importance and access frequency) and are eventually removed if they become irrelevant.
  
- **Associative Memory (Graph Relationships):**  
  The system can create graph relationships (edges) between facts. These associations help link related information together for graph traversal and further reasoning. Relationships include attributes like confidence, rationale, and timestamps.

## Core Components

1. **Fact Storage (`remember()`):**
   - Generates a unique fact ID.
   - Computes a TTL based on the fact's importance.
   - Generates a vector embedding of the fact’s content.
   - Extracts glossary terms from the fact.
   - Upserts (inserts/updates) the fact into an ArangoDB collection.
   - Logs a debug-friendly version of the fact (with truncated embeddings).

2. **Hybrid Search (`recall()` and `hybrid_search()`):**
   - Combines semantic similarity (using vector embeddings), BM25 text search, and glossary matching.
   - Merges the scores from different components (and optionally includes domain filtering).
   - Returns the top-N relevant facts.
   - Falls back to a separate keyword (domain) search if no results are found.

3. **Graph Relationships (`create_association()`):**
   - Once facts are stored, related facts can be linked by creating an edge in the `agent_associations` collection.
   - Each edge includes metadata such as confidence, rationale, and creation timestamp.
   - Graph relationships allow the system to perform graph traversal queries to explore associations among facts.

## Background Processing of Associations

To ensure that the insertion of a fact remains fast, the creation of graph relationships is handled separately. For example, you can use Python's `asyncio.to_thread` to offload the association creation to a background thread without blocking the main process.

### Example Using `asyncio.to_thread`

Below is an example snippet showing how you might create associations in the background using `asyncio.to_thread`:

```python
import asyncio
from agent_memory import AgentMemorySystem  # Import your agent_memory system

async def create_relationships_for_fact(agent_memory: AgentMemorySystem, new_fact_id: str, related_fact_ids: list):
    tasks = []
    for related_id in related_fact_ids:
        # Offload the creation of each association to a separate thread
        tasks.append(asyncio.to_thread(agent_memory.create_association, new_fact_id, related_id, association_type="related", weight=0.7))
    results = await asyncio.gather(*tasks)
    return results

async def main():
    # Initialize agent memory system
    agent_memory = AgentMemorySystem()
    agent_memory.initialize()

    # Assume we just inserted a fact with ID "fact_abc123"
    new_fact_id = "fact_abc123"

    # Assume these are the IDs of related facts determined via recall/hybrid search
    related_fact_ids = ["fact_def456", "fact_ghi789"]

    # Create associations in the background
    association_results = await create_relationships_for_fact(agent_memory, new_fact_id, related_fact_ids)
    print("Association Results:", association_results)

if __name__ == "__main__":
    asyncio.run(main())
