# Cursor Rules Database

A simple yet powerful tool for managing and querying coding standards, patterns, and rules stored in ArangoDB with vector search capabilities.

## Overview

The Cursor Rules Database provides:

1. **Rule Management** - Store, retrieve, and search for coding rules and patterns
2. **Hybrid Search** - Combines BM25 text search with vector similarity for optimal results
3. **Vector Search** - Find semantically relevant rules using Nomic ModernBert embeddings
4. **Example Storage** - Associate good and bad examples with each rule
5. **Command Line Interface** - Easy-to-use CLI for interacting with the rules database
6. **Cursor Rules Agent** - Solve coding problems by finding and applying relevant rules

## Quick Start

### Setup

1. Ensure ArangoDB is running:
   ```
   docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=openSesame arangodb/arangodb:latest
   ```

2. Install dependencies:
   ```
   pip install python-arango rich
   ```

3. Initialize the database with rules from `.cursor/rules`:
   ```
   python cursor_rules_cli.py init
   ```

### Basic Usage

List all rules:
```
python cursor_rules_cli.py list
```

Show a specific rule:
```
python cursor_rules_cli.py show 001 --examples
```

Keyword search:
```
python cursor_rules_cli.py search "error handling"
```

Semantic search (vector similarity):
```
python cursor_rules_cli.py search "asynchronous operations" --semantic
```

Hybrid search (combining BM25 and vector similarity):
```
python cursor_rules_cli.py search "database operations" --hybrid
```

### Using the Cursor Rules Agent

Run the demo:
```
python cursor_rules_demo.py
```

This will:
1. Connect to the database
2. Search for rules related to sample problems using hybrid search
3. Find and apply the most relevant rules
4. Generate solutions based on rule examples

## Components

- **cursor_rules_simple.py** - Core implementation of the database connection and operations
- **cursor_rules_cli.py** - Command line interface for interacting with the database
- **cursor_rules_demo.py** - Demo of using rules to solve coding problems
- **test_cursor_rules.py** - Unit tests for the implementation

## Search Capabilities

The implementation offers three search methods:

1. **Keyword Search**: Simple text matching in rule titles, descriptions, and content
2. **Vector Search**: Semantic similarity using Nomic ModernBert embeddings 
3. **Hybrid Search**: Combines BM25 text relevance with vector similarity for optimal results

### Hybrid Search Details

The hybrid search implements the recommended approach from the ArangoDB documentation:

```aql
FOR doc IN rules_search_view
  SEARCH ANALYZER(
    doc.content IN TOKENS(@searchText, "text_analyzer") OR 
    doc.title IN TOKENS(@searchText, "text_analyzer"), 
    "text_analyzer"
  )
  LET bm25Score = BM25(doc)
  LET vectorScore = 1 - COSINE_SIMILARITY(doc.embedding, @queryVector)
  LET hybridScore = (bm25Score * 0.6) + (vectorScore * 0.4)
  SORT hybridScore DESC
```

This gives you the best of both worlds:
- BM25 text search for keyword matching
- Vector similarity for semantic understanding

## Vector Search Implementation

The implementation uses Nomic ModernBert embeddings for semantic search. Text is properly prefixed as required:

- "search_document: " for rule content
- "search_query: " for search queries

## Future Development

1. **CRUD Operations** - Add commands for creating, updating, and deleting rules
2. **Web Interface** - Create a simple web UI for browsing and searching rules
3. **Integration with IDEs** - Create plugins for VSCode and other editors
4. **Rule Suggestions** - Proactively suggest rules based on code being written
5. **Custom Weights** - Allow customizing the weights in hybrid search

## Troubleshooting

If you encounter issues:

1. **Database Connection** - Ensure ArangoDB is running with the correct password
2. **Missing Embeddings** - The implementation gracefully handles missing embedding utilities
3. **Rule Loading** - Check that rule files in `.cursor/rules` follow the naming pattern `NNN-name-of-rule.mdc`
4. **Search View Creation** - ArangoSearch views require Enterprise Edition for optimal performance

## Cursor Rules Package

A powerful tool for managing and enforcing coding rules and patterns in your projects.

### Directory Structure

The Cursor Rules package is organized according to the following structure:

```
cursor_rules/
├── __init__.py              # Package initialization
├── README.md                # This file
├── core/                    # Core functionality
│   ├── cursor_rules.py      # Main implementation
│   └── db.py                # Database connection handling
├── schemas/                 # JSON schemas and data models
│   ├── ai_knowledge_schema.json
│   └── db_schema.json
├── cli/                     # Command-line interfaces
│   ├── cli.py
│   └── commands/            # CLI subcommands
├── utils/                   # Utility functions
│   ├── helpers/             # General helpers
│   ├── ai/                  # AI-specific utilities
│   └── text/                # Text processing utilities
├── views/                   # Database view management
│   └── view_utils.py
├── scenarios/               # Scenario management
│   ├── sample_scenarios.json
│   └── scenario_management.py
├── docs/                    # Documentation
│   ├── retrieval_scenarios.md
│   └── task.md
├── scripts/                 # Utility scripts
│   ├── cleanup_databases.py
│   └── demo.py
└── tests/                   # Test suite
    ├── unit/                # Unit tests
    ├── integration/         # Integration tests
    └── end_to_end/          # End-to-end tests
```

### Environment Setup

To properly use this package, ensure your environment is set up correctly:

1. Add `PYTHONPATH` to your `.env` file:
   ```
   PYTHONPATH=/path/to/your/project/src
   ```

2. Install the package in development mode:
   ```
   uv pip install -e .
   ```

3. Run commands using the module syntax:
   ```
   python -m agent_tools.cursor_rules.cli
   ```

### Package Organization Guidelines

1. **Core Functionality:** Place main business logic in the `core` directory
2. **CLI Commands:** Add new commands to the `cli/commands` directory
3. **Utils:** Add utility functions to the appropriate subdirectory in `utils`
4. **Schemas:** Store all JSON schemas in the `schemas` directory
5. **Tests:** Organize tests to mirror the structure of the code they test

### Troubleshooting

If you encounter import errors or other issues, refer to the `.cursor/TROUBLESHOOTING.md` file for detailed solutions. 