# DuaLipa Extraction Module

This module provides functionality for extracting code blocks and content from various file types and repositories, with support for multiple programming languages.

## Module Structure

```
extraction/
├── __init__.py           # Public API
├── extractors/
│   ├── __init__.py      # Submodule organization
│   ├── code/            # Language-specific extraction
│   │   ├── __init__.py
│   │   ├── python_extractor.py
│   │   ├── js_ts_extractor.py
│   │   └── generic_extractor.py
│   ├── markdown/        # Markdown parsing and extraction
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── hierarchy.py
│   │   └── extractor.py
│   ├── github/         # Repository operations
│   │   ├── __init__.py
│   │   ├── repo_utils.py
│   │   └── api_utils.py
│   └── utils/          # Common utilities
│       ├── __init__.py
│       ├── language_utils.py
│       ├── validation_utils.py
│       ├── verification_utils.py
│       └── stats_utils.py
└── README.md
```

## Key Features

1. **Multi-language Code Extraction**
   - Python (AST-based)
   - JavaScript/TypeScript (tree-sitter)
   - Generic pattern-based extraction

2. **Markdown Content Extraction**
   - Section hierarchy
   - Code block extraction
   - Content organization

3. **GitHub Repository Handling**
   - Repository cloning
   - Metadata fetching
   - File operations

4. **Common Utilities**
   - Language detection
   - Block validation
   - Code verification
   - Statistics tracking

## Dependencies

- `ast`: Python AST parsing
- `tree-sitter`: JavaScript/TypeScript parsing
- `markdown-it-py`: Markdown parsing
- `requests`: GitHub API calls
- `loguru`: Logging

## Usage Examples

### Code Extraction

```python
from agent_tools.dualipa.extraction import extract_python_blocks

# Extract Python code blocks
blocks, stats = extract_python_blocks("script.py")
print(f"Found {stats['total_blocks']} blocks")
```

### Markdown Extraction

```python
from agent_tools.dualipa.extraction import extract_markdown_blocks

# Extract markdown content
blocks, stats = extract_markdown_blocks("README.md")
print(f"Found {stats['sections']} sections")
```

### GitHub Operations

```python
from agent_tools.dualipa.extraction import clone_repository

# Clone a repository
repo_path = clone_repository(
    "https://github.com/example/repo.git",
    target_dir="repos",
    depth=1
)
```

## Test Organization

Tests are organized in order of dependency:

1. Core Functionality (01-10)
   - Basic setup
   - Imports
   - Stats tracking
   - GitHub utils

2. Language and Parsing (15-25)
   - Language detection
   - Format validation
   - Block verification
   - Tree-sitter parsing

3. Basic Extraction (30-45)
   - Python extraction
   - JS/TS extraction
   - Markdown extraction
   - Generic extraction

4. Hierarchy and Parsing (51-55)
   - Markdown hierarchy
   - Code hierarchy

5. Integration (65-90)
   - Code extractor
   - Multi-language
   - Repository integration

## Contributing

1. Follow the module structure
2. Add comprehensive documentation
3. Include usage examples
4. Add tests in correct order
5. Update this README as needed 