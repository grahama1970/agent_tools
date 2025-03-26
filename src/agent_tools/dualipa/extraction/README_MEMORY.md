# AI Memory System

A comprehensive memory management system for AI assistants to overcome context limitations.

## Overview

The AI Memory System provides persistent memory capabilities for AI assistants, enabling them to:

1. Maintain context across interaction boundaries
2. Store and retrieve documentation with semantic search
3. Track and learn from errors
4. Build knowledge relationships
5. Manage state during complex operations

This system addresses fundamental limitations in AI assistants by providing a robust, queryable persistence layer that retains state beyond the current conversation.

## Architecture

The memory system consists of three main components:

1. **TestStateManager**: Core state persistence using SQLite with tables for state, verification, checkpoints, context, documentation, file tracking, repo stats, and metadata
2. **EnhancedMemory**: Adds semantic search capabilities, relationship tracking, and error pattern learning on top of the base state manager
3. **AIMemorySystem**: Integrates proper embedding utilities for more accurate semantic search and automatic relationship discovery

### Database Schema

The system uses a SQLite database with the following tables:

- `state`: Key-value storage for basic state information
- `verification_log`: Tracks test assertions and verifications
- `file_tracking`: Monitors file extraction status
- `repo_stats`: Stores repository statistics
- `checkpoints`: Tracks named checkpoints in processes
- `metadata`: Stores general metadata information
- `context`: Manages AI context information
- `documentation`: Stores documentation for self-reference

## Using the Memory System

### Initialization

Initialize the memory system with:

```python
from agent_tools.dualipa.extraction.ai_memory_system import initialize_memory_system

# Initialize with default in-memory database
initialize_memory_system()

# Or with a persistent database file
initialize_memory_system("memory.db")
```

Or use the command-line script:

```bash
python initialize_memory.py --db-path memory.db
```

### Context Management

Remember your current context:

```python
from agent_tools.dualipa.extraction.ai_memory_system import remember

remember(
    task="Implementing the vector search functionality",
    goal="Make documentation search more accurate with embeddings",
    progress="Vector similarity function is working",
    next_steps="Add batch processing capabilities",
    priority=8,
    tags=["feature", "search"]
)
```

Recall your context:

```python
from agent_tools.dualipa.extraction.ai_memory_system import recall

# Get most recent context
current_context = recall(last=True)

# Search contexts by text
search_results = recall(search="vector search", semantic=True)

# Filter by tag and priority
filtered_contexts = recall(tag="feature", priority_min=7)
```

### Documentation Management

Save documentation:

```python
from agent_tools.dualipa.extraction.ai_memory_system import save_docs

save_docs(
    topic="vector-search",
    content="# Vector Search\n\nVector search uses embeddings to...",
    summary="Guide to vector-based semantic search",
    importance=8,
    tags=["search", "embeddings"]
)
```

Find documentation:

```python
from agent_tools.dualipa.extraction.ai_memory_system import find_docs

# Get specific topic
doc = find_docs(topic="vector-search")

# Search with semantic matching
results = find_docs(search="semantic similarity", semantic=True)

# Filter by tags and importance
filtered_docs = find_docs(tag="search", importance_min=7)
```

### Error Handling

Log errors for future pattern recognition:

```python
from agent_tools.dualipa.extraction.ai_memory_system import log_error

log_error(
    error_type="extraction_failure",
    details="Failed to extract nested classes from Python file",
    recovery_action="Add recursive extraction for nested class structures",
    severity=7,
    tags=["extraction", "python"]
)
```

Get recovery suggestions:

```python
from agent_tools.dualipa.extraction.ai_memory_system import suggest_recovery

# By error type
suggestion = suggest_recovery(error_type="extraction_failure")

# By error details with semantic matching
similar_errors = suggest_recovery(
    details="Cannot parse nested class structure with current parser",
    use_semantic=True
)
```

### Batch Operations

Execute multiple operations efficiently:

```python
from agent_tools.dualipa.extraction.ai_memory_system import batch_operation

results = batch_operation([
    {
        "op": "remember",
        "task": "Implementing batch operations",
        "goal": "Make memory operations more efficient",
        "progress": "Basic implementation complete",
        "next_steps": "Add error handling"
    },
    {
        "op": "save_docs",
        "topic": "batch-operations",
        "content": "# Batch Operations\n\nBatch operations allow...",
        "summary": "Guide to using batch operations"
    },
    {
        "op": "find_docs",
        "search": "efficiency",
        "semantic": True
    }
])
```

## Embedding Integration

The system supports different embedding methods:

1. **Transformer Embeddings**: When available, uses proper transformer embeddings from the fetch_docs module, which provides state-of-the-art semantic similarity
2. **TF-IDF Fallback**: If transformer embeddings are not available, falls back to TF-IDF vectorization

The embedding system automatically selects the best available method and builds a vector store for efficient similarity searches.

## Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Initialize
python initialize_memory.py --db-path memory.db

# Remember context
python -m agent_tools.dualipa.extraction.ai_memory_system remember "task" "goal" "progress" "next steps"

# Find documentation
python -m agent_tools.dualipa.extraction.ai_memory_system find-docs --search "query" --semantic

# Check system status
python -m agent_tools.dualipa.extraction.ai_memory_system status
```

## Best Practices

1. **Use Semantic Search**: Enable semantic search with `semantic=True` for more relevant results
2. **Set Appropriate Priorities**: Use priority levels (1-10) to control staleness thresholds
3. **Add Tags**: Tag all information for better organization and filtering
4. **Use Batch Operations**: When performing multiple memory operations at once
5. **Add Detailed Documentation**: More detailed documentation enables better semantic matching
6. **Log Error Patterns**: Consistently log errors with recovery suggestions for future reference