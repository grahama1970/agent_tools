# DuaLipa Extraction Module

This document provides comprehensive information about the DuaLipa extraction module, including code extraction and documentation integration capabilities.

## Overview

The DuaLipa extraction module processes repositories to extract:

1. **Code blocks** - Functions, classes, methods, and other code structures
2. **Documentation** - Both inline documentation and external documentation sources
3. **Relationships** - Parent-child hierarchies and semantic connections

The extraction maintains hierarchical structures and relationships, making the output suitable for integration with QA systems and LLMs.

## Code Extraction

The core extraction process includes:

1. Repository scanning and file discovery
2. Language detection and parsing
3. AST-based or regex-based extraction (depending on language)
4. Metadata collection and relationship building
5. Format conversion for QA compatibility

### Supported Languages

- Python (AST-based)
- JavaScript/TypeScript (Tree-sitter)
- HTML/CSS (specialized parsers)
- Markdown (specialized parsers)
- Other languages (regex-based fallback)

## Documentation Integration

The system now supports integration with external documentation through the fetch_docs module, which:

1. Automatically detects documentation links in markdown files
2. Downloads and processes external documentation from supported sources
3. Extracts sections, code blocks, tables, and images
4. Maintains the hierarchical structure of the documentation
5. Integrates with code extraction in a unified format

### Supported Documentation Sources

- ReadTheDocs (`*.readthedocs.io`, `readthedocs.org`)
- ArangoDB Documentation (`docs.arangodb.com`)
- Generic HTML documentation with proper section headings
- Markdown documentation

## Extraction Format

All extracted blocks follow a consistent format with parent-child relationships:

```json
{
  "uuid": "<unique-id>",
  "id": "<type>_<name>",
  "name": "<human-readable-name>",
  "type": "<block-type>",
  "language": "<language>",
  "content": "<content-text>",
  "parent_uuid": "<parent-block-uuid>",
  "child_uuids": ["<child-block-uuids>"],
  "metadata": {
    "language": "<language>",
    "file_path": "<relative-path>",
    "source_url": "<url-if-applicable>",
    "line_start": <start-line>,
    "line_end": <end-line>,
    "doc_type": "<documentation-type-if-applicable>"
  }
}
```

### Block Types

- `file` - Source code file
- `class` - Class definition
- `function` - Function or method definition
- `documentation` - External documentation site
- `doc_page` - Documentation page
- `doc_section` - Documentation section
- `code_block` - Code example in documentation
- `table` - Table in documentation

## Hierarchical Structure

The extraction maintains proper parent-child relationships, which is crucial for:

1. **Context preservation** - Understanding where blocks appear in the source
2. **Semantic relationships** - Connecting related blocks (e.g., methods to classes)
3. **Documentation connections** - Linking code to relevant documentation
4. **LLM processing** - Enabling navigation of relationships during QA

### Example Hierarchy

```
file
├── class
│   ├── method
│   ├── method
│   └── nested_class
│       └── method
└── function

documentation
├── doc_page
│   ├── doc_section
│   │   ├── doc_section
│   │   ├── code_block
│   │   └── table
│   └── doc_section
└── doc_page
    └── doc_section
```

## Validation Framework

A comprehensive validation framework ensures extraction quality:

1. **Structure validation** - Verifies hierarchical relationships
2. **Content validation** - Checks semantic content against expectations
3. **Format validation** - Ensures compatibility with QA systems
4. **Hierarchy validation** - Validates parent-child relationships

### Validation Features

- Expected format templates for different document types
- Automatic document type detection
- Format conversion for validation compatibility
- Visual hierarchy representation
- Detailed validation reports

## Usage

### Basic Code Extraction

```python
from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks

# Extract code blocks
blocks = extract_all_blocks("/path/to/repository")
```

### Documentation Integration

```python
from agent_tools.dualipa.fetch_docs_integration import integrate_docs_with_extraction

# Get code blocks from extraction
code_blocks = extract_all_blocks("/path/to/repository")

# Enhance with documentation
all_blocks = integrate_docs_with_extraction("/path/to/repository", code_blocks)
```

### Format Conversion

```python
from agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import (
    create_qa_compatible_blocks,
    create_qa_compatible_output
)

# Convert blocks to QA-compatible format
qa_blocks = create_qa_compatible_blocks(blocks)
qa_output = create_qa_compatible_output(qa_blocks)
```

### Validation

```python
from agent_tools.dualipa.extraction.examples.end_to_end.validation import (
    validate_extraction_result,
    load_expected_format
)

# Validate extraction results
expected_format = load_expected_format("expected_format.json")
validation_result = validate_extraction_result(blocks, expected_format)
```

### Hierarchy Validation

```python
# Run hierarchy validation for a single extraction
python validate_hierarchy.py --input extraction_output.json

# Validate all extractions
python validate_all_hierarchies.py
```

## Best Practices

1. **Maintain parent-child relationships** - Ensure every block (except root blocks) has a parent reference
2. **Bidirectional references** - Both parent and child should reference each other
3. **Consistent metadata** - Include language, file path, and other relevant metadata
4. **Unique IDs** - Each block should have a unique UUID
5. **Complete hierarchy** - Include all intermediate blocks in the hierarchy

## Integration with DuaLipa

To use the full extraction pipeline with documentation integration:

```python
from agent_tools.dualipa.cli import extract_repository

# Full extraction with documentation integration
result = extract_repository("/path/to/repository", include_docs=True)
```

## Future Enhancements

- Support for additional documentation sources
- Enhanced language support (Rust, Go, etc.)
- Improved semantic relationship detection
- Real-time documentation updates