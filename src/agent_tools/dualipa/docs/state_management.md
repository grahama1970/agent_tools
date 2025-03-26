# State Management and Memory System for AI Operations

This document describes the comprehensive state management and memory system designed to overcome context limitations in AI assistants during extraction and testing operations.

## Core Design Principles

1. **Explicit State Persistence**: No reliance on in-context memory
2. **Context Awareness**: Always knowing what's being done and why
3. **Self-documentation**: Ability to recall stored knowledge
4. **Verification Checkpoints**: Creating a chain of evidence
5. **Statistical Tracking**: Maintaining metrics for completeness
6. **Recovery Mechanisms**: Learning from past errors

## Database Schema

The system uses an SQLite database with the following tables:

### Core State Tables

1. **state**: Key-value storage for general state persistence
   - `key`: Primary key for state entry
   - `value`: JSON-serialized value (allows storing complex structures)
   - `value_type`: Type information for proper deserialization
   - `updated_at`: Timestamp for tracking changes

2. **verification_log**: History of verification steps and outcomes
   - `id`: Auto-incremented ID
   - `checkpoint`: Current phase/checkpoint name
   - `step`: Specific verification step name
   - `expected`: Expected value (JSON serialized)
   - `actual`: Actual value (JSON serialized)
   - `passed`: Boolean indicating success/failure
   - `timestamp`: When verification occurred

3. **checkpoints**: Tracks test phase transitions
   - `name`: Checkpoint identifier
   - `timestamp`: When phase began
   - `description`: Human-readable description

### Context Awareness Tables

4. **context**: Tracks what the assistant is currently doing
   - `context_key`: Identifier for specific context
   - `task`: What I'm currently doing
   - `goal`: Why I'm doing it
   - `progress`: Current step in the process
   - `assumptions`: Key assumptions being made
   - `problems`: Issues encountered
   - `next_steps`: What to do next
   - `notes`: Additional context details
   - `updated_at`: Last update timestamp

5. **documentation**: Self-reference knowledge storage
   - `topic`: Unique topic identifier
   - `content`: Main documentation content
   - `summary`: Concise summary
   - `source`: Where this info came from
   - `examples`: Usage examples
   - `related_topics`: Cross-references
   - `importance`: Priority level (1-10)
   - `created_at` & `updated_at`: Timestamps

### Extraction-Specific Tables

6. **file_tracking**: Monitors extraction completeness
   - `file_path`: Primary key for tracking file
   - `file_type`: Extension or language
   - `extracted`: Whether file was processed
   - `extraction_time`: When extraction occurred
   - `extracted_uuid`: Reference to extraction block
   - `size`: File size in bytes
   - `included_in_output`: Whether included in final output

7. **repo_stats**: Repository statistics by file type
   - `extension`: File extension (primary key)
   - `count`: Total files with extension
   - `extracted_count`: How many were extracted
   - `percentage`: Percentage of repository

8. **metadata**: General information storage
   - `key`: Metadata identifier
   - `value`: JSON-serialized value
   - `updated_at`: Last update timestamp

## CLI Usage

The system provides two interfaces:

1. **Full CLI** (`cli.py`): Comprehensive management capabilities
2. **Memory Helper** (`memory.py`): Simplified interface for AI operations

### CLI Examples

```bash
# Initialize a database
python -m dualipa.extraction.cli init --db-path extraction_state.db

# Set a context
python -m dualipa.extraction.cli context current-task \
  --task "Extracting Python files" \
  --goal "Complete repository extraction" \
  --next-steps "Validate results"

# Add documentation
python -m dualipa.extraction.cli add-doc "extraction-process" \
  "Detailed extraction process..." \
  --summary "How extraction works"

# Show current context
python -m dualipa.extraction.cli show-context

# Load project documentation
python -m dualipa.extraction.cli load-docs
```

### Memory Helper Examples

```python
from dualipa.extraction.memory import remember, recall, think, remind_me, note

# Remember current context
remember(
    "Extracting ArangoDB docs",
    "Create comprehensive extraction",
    "Processing JavaScript files",
    "Next validate extraction quality"
)

# Take a quick note
note("extraction-bug", "JavaScript regex patterns need escaping")

# Get a reminder
remind_me()

# Record a thought process
think("The extraction quality depends on proper parent-child relationships")

# Find documentation
from dualipa.extraction.memory import find_docs, load_project_docs

# Load all project docs
load_project_docs()

# Search for relevant docs
find_docs(search="extraction")
```

## Advanced Features

### State Verification

The system provides robust mechanisms for verifying extraction completeness:

```python
from dualipa.extraction.test_state_manager import verify_extraction_completeness

# Verify extraction completeness
result = verify_extraction_completeness(
    repo_stats, 
    extraction_results,
    state_manager
)

if result:
    print("Extraction complete!")
else:
    print("Extraction incomplete - see verification log")
```

### Error Learning

The system can learn from past errors:

```python
from dualipa.extraction.memory import log_error, suggest_recovery

# Log an error pattern
log_error(
    "markdown_table_parsing",
    "Failed to extract tables with merged cells",
    "Use HTML extraction instead of markdown parser"
)

# Get suggested recovery
recovery = suggest_recovery("markdown_table_parsing")
print(f"Suggested recovery: {recovery}")
```

### Self-Documentation

The system maintains its own documentation and can add to it:

```python
from dualipa.extraction.memory import save_docs

# Document a complex process
save_docs(
    "extraction-algorithm",
    """
    The extraction algorithm works by:
    1. Identifying language
    2. Selecting appropriate parser
    3. Extracting blocks
    4. Validating output
    """,
    summary="How extraction works",
    importance=9
)
```

## Integration With Extraction Process

The state management system integrates with extraction by:

1. Tracking all files to be processed
2. Recording extraction attempts and results
3. Verifying extraction completeness
4. Maintaining statistics on extraction quality
5. Providing checkpoints for recovery

## Common Use Patterns

### Maintaining Context During Complex Operations

```python
# At the start of an operation
remember_context(
    "Extracting TypeScript files",
    "Complete type information extraction",
    "Beginning extraction",
    "Extract files, validate types, check completeness"
)

# Mid-operation
remember_context(
    "Extracting TypeScript files",
    "Complete type information extraction",
    "50% complete, processing interfaces",
    "Continue extraction, focus on generic types"
)
```

### Learning From Documentation

```python
# Load all project docs first
load_project_docs()

# When confused about a concept
docs = find_docs(search="parent-child")
if docs:
    print(f"Found documentation: {docs}")
```

### Monitoring Extraction Progress

```python
# Get extraction statistics
stats = get_state_manager().get_extraction_stats()
print(f"Extracted {stats['extraction_rate']}% of files")
```

## Common Problems and Solutions

1. **Context Loss During Long Operations**
   - Solution: Regularly update context with `remember_context()`
   - Example: `remember_context("task", "goal", "current step", "next steps")`

2. **Forgetting Documentation Details**
   - Solution: Use `load_project_docs()` and `find_docs(search="term")`
   - Example: `docs = find_docs(search="extraction")`

3. **Unclear What Step is Next in Process**
   - Solution: Use `remind_me()` to get current context
   - Example: `remind_me()`

4. **Repeating Past Errors**
   - Solution: Log errors and check for solutions with `log_error()` and `suggest_recovery()`
   - Example: `suggestion = suggest_recovery("error_type")`

## Best Practices

1. **Always Begin Important Tasks By Setting Context**
   ```python
   remember(
       "What you're doing",
       "Why you're doing it",
       "Current progress",
       "Next steps"
   )
   ```

2. **Document Key Concepts As You Learn Them**
   ```python
   save_docs(
       "important-concept",
       "Detailed explanation...",
       summary="Brief summary"
   )
   ```

3. **Check for Existing Documentation Before Implementing Something**
   ```python
   docs = find_docs(search="relevant term")
   ```

4. **Keep Context Updated As You Progress**
   ```python
   # Update every time you reach a milestone
   remember(
       "Same task",
       "Same goal",
       "Updated progress",
       "Updated next steps"
   )
   ```

5. **Use Checkpoints at Key Phase Transitions**
   ```python
   state_manager = get_state_manager()
   state_manager.set_checkpoint("extraction_complete", "Finished extraction phase")
   ```

## Conclusion

This state management system addresses the fundamental limitations of AI assistants in maintaining context over time. By providing explicit persistence, self-documentation capabilities, and verification mechanisms, it ensures reliable and consistent operation during complex extraction tasks.