# Memory System Architecture for AST Extraction

This document provides an overview of the memory system architecture used in the AST extraction pipeline.

## System Layers

The memory system consists of multiple layers, each providing different functionality:

### 1. TestStateManager (Base Layer)

The `TestStateManager` (from `test_state_manager.py`) provides the foundational persistence layer:

- SQLite database storage with structured tables
- Basic CRUD operations for memory entities
- Transaction support for atomic operations
- Table schemas for various memory types (context, documentation, errors)

### 2. AIMemorySystem (Semantic Layer)

The `AIMemorySystem` (from `ai_memory_system.py`) builds on the base layer to provide:

- Vector-based semantic search using embeddings
- Contextual recall with relevance scoring
- Documentation storage with semantic indexing
- Error tracking and suggestion mechanisms

### 3. ExtractionMemory (Domain Layer)

The `ExtractionMemory` (from `extraction_memory.py`) provides domain-specific helpers:

- Extraction-specific memory operations
- Progress tracking through extraction stages
- Error recording with recovery suggestions
- Knowledge management for extraction patterns

### 4. AstExtractor (Application Layer)

The `AstExtractor` (from `ast_extractor.py`) integrates with the memory system to:

- Use memory during extraction operations
- Record successful extraction patterns
- Learn from errors and apply recovery strategies
- Track statistics and progress

## Memory Types

The system manages several types of memory:

### Context Memory

Tracks the current state of extraction operations:
- Current task and goal
- Progress status
- Next steps

Example:
```python
track_extraction_start(
    repo_name,
    "ast_extraction",
    {"source_path": file_path, "timestamp": time.time()}
)
```

### Documentation Memory

Stores knowledge about extraction patterns and techniques:
- Successful extraction patterns
- Code structure patterns
- Language-specific extraction knowledge

Example:
```python
save_extraction_knowledge(
    "nested_classes_python",
    "Successfully extracted nested classes in file.py",
    tags=["python", "nested_classes", "success_pattern"]
)
```

### Error Memory

Records errors encountered during extraction:
- Error types and details
- Recovery actions
- Severity and context

Example:
```python
record_extraction_error(
    "parsing_error",
    "Failed to parse nested class structure",
    file_path,
    "Use a more robust parser for complex class hierarchies"
)
```

## Memory Usage Patterns

The memory system is used in several key patterns:

### 1. Context Checking

Before extraction:
```python
context = get_extraction_context()
# Decide extraction strategy based on context
```

### 2. Pattern Recognition

During extraction:
```python
knowledge = find_extraction_knowledge("nested class extraction")
if knowledge:
    # Apply knowledge to current extraction
```

### 3. Error Recovery

When errors occur:
```python
similar_errors = find_similar_errors(error_details)
if similar_errors:
    # Apply recovery strategies from similar errors
```

### 4. Knowledge Accumulation

After successful extraction:
```python
save_extraction_knowledge(
    f"ast_pattern_{language}_{structure_type}",
    "Successful extraction pattern details...",
    tags=[language, structure_type]
)
```

## Integration Benefits

The memory system provides several benefits to the extraction process:

1. **Improved Accuracy**: Learning from past successes and errors
2. **Faster Processing**: Reusing known patterns and strategies
3. **Robust Recovery**: Applying known solutions to similar problems
4. **Continuous Improvement**: Building knowledge over time
5. **Cross-File Learning**: Applying patterns across different files