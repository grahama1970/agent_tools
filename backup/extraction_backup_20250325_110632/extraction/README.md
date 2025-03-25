# DuaLipa Extraction Module

The DuaLipa extraction module provides robust tools for extracting code and documentation from repositories, maintaining proper hierarchical structure for LLM processing.

## Core Features

- **Code Extraction**: Extract functions, classes, and methods from source code
- **Documentation Integration**: Integrate external documentation from supported sources
- **Hierarchical Structure**: Maintain parent-child relationships throughout extraction
- **Validation Framework**: Ensure extraction quality and structural integrity
- **QA Format Compatibility**: Generate outputs compatible with QA systems

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
│   ├── html/            # HTML documentation extraction
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
├── examples/
│   └── end_to_end/     # End-to-end examples and validation tools
├── EXTRACTION_DOCUMENTATION.md
├── PARENT_CHILD_REQUIREMENTS.md
└── README.md
```

## Quick Start

```python
# Basic extraction with documentation integration
from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
from agent_tools.dualipa.fetch_docs_integration import integrate_docs_with_extraction

# Extract code blocks
code_blocks = extract_all_blocks("/path/to/repository")

# Enhance with documentation
all_blocks = integrate_docs_with_extraction("/path/to/repository", code_blocks)

# Convert to QA-compatible format
from agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import (
    create_qa_compatible_blocks,
    create_qa_compatible_output
)

qa_blocks = create_qa_compatible_blocks(all_blocks)
qa_output = create_qa_compatible_output(qa_blocks)
```

## Documentation

- [Extraction Documentation](EXTRACTION_DOCUMENTATION.md) - Comprehensive guide to extraction functionality
- [Parent-Child Requirements](PARENT_CHILD_REQUIREMENTS.md) - Requirements for hierarchical structure
- [Code Extraction](examples/end_to_end/CODE_EXTRACTION.md) - Details on the code extraction process
- [Validation Framework](examples/end_to_end/VALIDATION_FRAMEWORK.md) - Guide to the validation framework
- [Hierarchy Validation](examples/end_to_end/HIERARCHY_VALIDATION.md) - Tools for validating parent-child relationships

## Validation Tools

```bash
# Validate a single extraction
python examples/end_to_end/validate_hierarchy.py --input extraction_output.json

# Validate all extractions
python examples/end_to_end/validate_all_hierarchies.py
```

## Supported Documentation Sources

- ReadTheDocs (`*.readthedocs.io`, `readthedocs.org`)
- ArangoDB Documentation (`docs.arangodb.com`)
- Generic HTML documentation with proper section headings
- Markdown documentation

## Multi-language Support

- **Python**: AST-based extraction
- **JavaScript/TypeScript**: Tree-sitter based extraction
- **HTML/CSS**: Specialized parsers
- **Markdown**: Specialized parsers
- **Other languages**: Regex-based fallback

## Command Line Interface

```bash
# Extract a repository with documentation integration
python -m agent_tools.dualipa.cli extract /path/to/repository --include-docs

# Convert extraction to QA format
python -m agent_tools.dualipa.cli convert extraction_output.json --format qa

# Validate extraction output
python -m agent_tools.dualipa.cli validate extraction_output.json --expected expected_format.json
```

## Requirements for LLM Integration

For proper integration with LLMs, extraction outputs must maintain:

1. **Bidirectional References**: Both parent → child and child → parent
2. **Complete Hierarchy**: No orphaned blocks or broken relationships
3. **Consistent Structure**: Follow expected parent-child patterns
4. **Metadata Consistency**: Include required metadata fields

See [Parent-Child Requirements](PARENT_CHILD_REQUIREMENTS.md) for detailed requirements.

## Testing

```bash
# Run extraction tests
python -m pytest examples/end_to_end/test_extraction_e2e.py

# Test documentation integration
python examples/end_to_end/test_fetch_docs_integration.py

# Run blind tests for specific documentation sources
python examples/end_to_end/blind_test.py
```

## Contributing

1. Follow the module structure
2. Maintain bidirectional parent-child relationships
3. Include comprehensive documentation
4. Add tests for new functionality
5. Validate extractions using the validation framework