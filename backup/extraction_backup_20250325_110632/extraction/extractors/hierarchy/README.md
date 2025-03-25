# Code Hierarchy Analysis Module

This module provides code structure analysis and hierarchy extraction for the DuaLipa extraction pipeline, focusing on class relationships, function dependencies, and module organization.

## Architecture

The hierarchy module has been refactored to improve maintainability and comply with the 500-line limit standard. The new structure is as follows:

```
hierarchy/
├── __init__.py           - Package exports
├── core.py               - Core functionality and dispatching
├── utils.py              - Common utility functions
├── python/               - Python-specific analysis
│   ├── __init__.py
│   └── parser.py
├── js_ts/                - JavaScript/TypeScript analysis
│   ├── __init__.py
│   └── parser.py
└── generic/              - Generic language analysis
    ├── __init__.py
    └── parser.py
```

## Key Features

1. **Language-specific Analysis**: Specialized parsers for Python (using AST) and JavaScript/TypeScript (using tree-sitter), with fallback to generic pattern matching for other languages.

2. **Class Hierarchy Detection**: Identifies class inheritance relationships, methods, and properties.

3. **Function Analysis**: Maps function dependencies, arguments, and return types.

4. **Import/Export Tracking**: Records module dependencies and exported symbols.

5. **Interface Analysis**: For TypeScript, analyzes interface definitions and implementations.

6. **QA Integration**: Produces output compatible with the QA module's requirements.

## Usage

### Basic Usage

```python
from agent_tools.dualipa.extraction.extractors.hierarchy import analyze_code_hierarchy

# Analyze a file
hierarchy, stats = analyze_code_hierarchy("path/to/file.py")

# Access hierarchy information
classes = hierarchy.get("classes", {})
functions = hierarchy.get("functions", {})
imports = hierarchy.get("imports", [])

# Check statistics
print(f"Found {stats.get('classes', 0)} classes and {stats.get('functions', 0)} functions")
```

### Building Hierarchical Relationships

```python
from agent_tools.dualipa.extraction.extractors.hierarchy import build_code_hierarchy

# Given a list of code blocks
blocks = [
    {
        "uuid": "123",
        "file_path": "example.py",
        "type": "code",
        "language": "python",
        "depth": 0
    },
    # ...more blocks
]

# Build parent-child relationships
stats = build_code_hierarchy(blocks)
```

### Using Utilities

```python
from agent_tools.dualipa.extraction.extractors.hierarchy.utils import format_hierarchy_summary

# Get a formatted summary of hierarchy information
hierarchy, _ = analyze_code_hierarchy("path/to/file.py")
summary = format_hierarchy_summary(hierarchy)
print(summary)
```

## Integration with QA Module

To ensure compatibility with the QA module, blocks extracted from the hierarchy should include:

- Required fields: `uuid`, `type`, `content`, `extraction_focus`, `summary_instructions`
- Type-specific fields for code blocks: `language`, `file_path`, `dependencies`
- Hierarchical relationship fields: `parent_uuid`, `child_uuids`, `breadcrumb`, `depth`

Example of converting hierarchy data to QA-compatible blocks:

```python
def create_qa_compatible_blocks(hierarchy_data, content):
    blocks = []
    
    # Create file-level block
    file_block = {
        "uuid": generate_uuid(),
        "type": "code",
        "content": content,
        "language": hierarchy_data["language"],
        "file_path": hierarchy_data["file_path"],
        "extraction_focus": "code structure",
        "summary_instructions": "Generate QA pairs about this code",
        "parent_uuid": None,
        "child_uuids": [],
        "breadcrumb": [Path(hierarchy_data["file_path"]).name],
        "depth": 0
    }
    blocks.append(file_block)
    
    # Add class blocks as children of the file block
    for class_name, class_info in hierarchy_data.get("classes", {}).items():
        # Extract class content using line numbers
        # Create class block with parent_uuid referring to file block
        # Add class block UUID to file block's child_uuids
        # ...
    
    return blocks
```

## Running Tests

Tests for the hierarchy module are in the `tests/dualipa/extraction/hierarchy/` directory:

```bash
# Run all hierarchy tests
pytest tests/dualipa/extraction/hierarchy/

# Run specific test files
pytest tests/dualipa/extraction/hierarchy/test_hierarchy.py
pytest tests/dualipa/extraction/hierarchy/test_qa_integration.py
```

## Known Issues and Limitations

There are several areas of the code that have been refactored for organization and to meet the 500-line limit, but contain provisional implementations that need to be improved for production use.

Please refer to [TECHNICAL_DEBT.md](./TECHNICAL_DEBT.md) for a comprehensive list of:
- Non-production implementations that need proper solutions
- Test-specific workarounds that need to be replaced
- Integration issues that need to be addressed
- Performance concerns for large codebases

These issues are tracked to ensure they're properly addressed in future work.