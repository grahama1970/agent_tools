# Code Extraction Modules

This document outlines how the code extraction modules work for Python (using AST) and other languages (using tree-sitter). These extractors transform source code into structured JSON objects following a similar format to the markdown extraction.

## Python AST Extraction

The Python AST extractor uses Python's built-in Abstract Syntax Tree parser to extract structured information from Python files.

### Input
- Python source code files (`.py`)

### Output
The output follows a similar structure to the markdown extraction, with sections being replaced by code entities:

```json
[
  {
    "uuid": "a1b2c3d4-e5f6-4a5b-9c3d-1e2f3a4b5c6d",
    "type": "file",
    "name": "example.py",
    "language": "python",
    "content": "...",
    "imports": [...],
    "classes": [...],
    "functions": [...]
  },
  {
    "uuid": "b2c3d4e5-f6a7-5b6c-0d1e-2f3a4b5c6d7e",
    "type": "class",
    "name": "ExampleClass",
    "language": "python",
    "content": "...",
    "parent_uuid": "a1b2c3d4-e5f6-4a5b-9c3d-1e2f3a4b5c6d",
    "methods": [...],
    "properties": [...]
  },
  {
    "uuid": "c3d4e5f6-a7b8-6c7d-1e2f-3a4b5c6d7e8f",
    "type": "function",
    "name": "example_function",
    "language": "python",
    "content": "...",
    "parent_uuid": "a1b2c3d4-e5f6-4a5b-9c3d-1e2f3a4b5c6d",
    "parameters": [...],
    "return_type": "..."
  }
]
```

### Key Features
1. **AST Parsing**: Uses Python's built-in `ast` module to parse Python code
2. **Entity Extraction**: Identifies classes, functions, methods, and imports
3. **Relationship Tracking**: Maintains parent-child relationships between entities
4. **Docstring Extraction**: Preserves docstrings for documentation generation
5. **Type Hints**: Extracts type hints from function signatures and return values

### Customization (5%)
- Conversion from AST node format to our unified JSON structure
- Addition of UUIDs for database compatibility
- Hierarchy tracking for nested entities
- Extraction of metadata like line numbers and positions

## Tree-Sitter Language Pack Extraction

The tree-sitter extractor uses language-specific parsers to extract structured information from various programming languages.

### Supported Languages
- JavaScript/TypeScript
- Java
- C/C++
- Go
- Ruby
- Rust
- And others supported by tree-sitter

### Input
- Source code files in various languages

### Output
The output follows the same format as the Python AST extraction, adapted for each language's specific features:

```json
[
  {
    "uuid": "d4e5f6a7-b8c9-7d8e-2f3a-4b5c6d7e8f9a",
    "type": "file",
    "name": "example.js",
    "language": "javascript",
    "content": "...",
    "imports": [...],
    "classes": [...],
    "functions": [...]
  },
  {
    "uuid": "e5f6a7b8-c9d0-8e9f-3a4b-5c6d7e8f9a0b",
    "type": "class",
    "name": "ExampleClass",
    "language": "javascript",
    "content": "...",
    "parent_uuid": "d4e5f6a7-b8c9-7d8e-2f3a-4b5c6d7e8f9a",
    "methods": [...],
    "properties": [...]
  }
]
```

### Key Features
1. **Language-Specific Parsing**: Uses tree-sitter grammars for accurate parsing
2. **Common Output Format**: Normalizes different language structures to a consistent output format
3. **Cross-Language Linking**: Enables connections between entities in different languages
4. **Preservation of Language Features**: Retains language-specific features while providing a unified view

### Customization (5%)
- Conversion from tree-sitter CST (Concrete Syntax Tree) to our JSON structure
- Normalization of language-specific features to common patterns
- Addition of UUIDs and relationship tracking
- Metadata enrichment for database queries

## Integration with ArangoDB

Both extraction systems are designed to produce output that can be directly inserted into ArangoDB:

1. **Document Collections**:
   - Files become documents in a "files" collection
   - Classes become documents in a "classes" collection
   - Functions become documents in a "functions" collection
   - Methods become documents in a "methods" collection

2. **Edge Collections**:
   - Parent-child relationships become edges in a "contains" collection
   - Import relationships become edges in a "imports" collection
   - Inheritance relationships become edges in a "extends" collection
   - Function calls become edges in a "calls" collection

3. **Query Examples**:
   ```aql
   // Find all methods in a class
   FOR method IN methods
     FILTER method.parent_type == 'class' AND method.parent_uuid == 'class-uuid'
     RETURN method
   
   // Find all classes that inherit from a base class
   FOR edge IN extends
     FILTER edge._to == 'classes/base-class-uuid'
     FOR class IN classes
       FILTER class._id == edge._from
       RETURN class
   ```

## Usage Example

```python
from pathlib import Path
from code_extractors import extract_python_ast, extract_with_treesitter

# Extract from Python files
py_blocks = extract_python_ast(Path("example.py"))

# Extract from TypeScript files
ts_blocks = extract_with_treesitter(Path("example.ts"), language="typescript")

# Combine all blocks
all_blocks = py_blocks + ts_blocks

# Write to JSON file
import json
with open("code_extraction.json", "w") as f:
    json.dump(all_blocks, f, indent=2)
```