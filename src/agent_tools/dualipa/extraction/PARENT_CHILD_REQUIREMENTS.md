# Parent-Child Relationship Requirements in DuaLipa Extraction

This document outlines the essential requirements for parent-child relationships in DuaLipa extraction outputs, which are crucial for proper LLM processing.

## Core Requirements

1. **Complete Bidirectional References**
   - Every child block MUST reference its parent via `parent_uuid`
   - Every parent block MUST reference its children via `child_uuids`
   - Both references must be consistent (parent → child and child → parent)

2. **Hierarchical Structure**
   - Blocks must maintain a logical hierarchical structure
   - Documentation blocks follow: documentation → doc_page → doc_section → (code_block, table)
   - Code blocks follow: file → class → method → nested elements

3. **No Orphaned Blocks**
   - Every block (except designated root blocks) must have a parent
   - Root blocks should be either file blocks or documentation blocks
   - No block should be disconnected from the hierarchy

4. **No Circular References**
   - The parent-child hierarchy must form a directed acyclic graph (DAG)
   - A block cannot be both an ancestor and descendant of another block
   - Example of invalid circular reference: A → B → C → A

5. **Ordered Relationships**
   - Children should be listed in `child_uuids` in a meaningful order
   - For documentation: sections should follow document flow
   - For code: elements should follow source code order

## Block Types and Relationships

The following block types have specific relationship requirements:

### Documentation Blocks

```
documentation (root)
└── doc_page
    └── doc_section
        ├── doc_section (nested)
        ├── code_block
        └── table
```

- `documentation`: Root block representing a documentation site
- `doc_page`: Child of documentation, representing an HTML page
- `doc_section`: Child of doc_page or another doc_section, representing a content section
- `code_block`, `table`: Children of doc_section, representing special content elements

### Code Blocks

```
file (root)
├── class
│   ├── method
│   └── attribute
└── function
```

- `file`: Root block representing a source code file
- `class`: Child of file, representing a class definition
- `function`: Child of file, representing a top-level function
- `method`: Child of class, representing a class method
- `attribute`: Child of class, representing a class attribute

## Validation Criteria

The extraction output must pass the following validation checks:

1. **Structure Validation**
   - Required block types are present
   - Parent-child relationships follow expected hierarchy
   - Metadata is consistent and complete

2. **Relationship Validation**
   - All references between blocks are valid
   - No circular references exist
   - No orphaned blocks exist

3. **Bidirectional Validation**
   - Every parent-child relationship is referenced from both sides
   - UUIDs are consistent between references

## Example of Valid Structure

```json
[
  {
    "uuid": "doc-123",
    "type": "documentation",
    "child_uuids": ["page-456"]
  },
  {
    "uuid": "page-456",
    "type": "doc_page",
    "parent_uuid": "doc-123",
    "child_uuids": ["section-789"]
  },
  {
    "uuid": "section-789",
    "type": "doc_section",
    "parent_uuid": "page-456",
    "child_uuids": ["code-012"]
  },
  {
    "uuid": "code-012",
    "type": "code_block",
    "parent_uuid": "section-789",
    "child_uuids": []
  }
]
```

## Validation Tools

Use the provided validation tools to verify parent-child relationships:

```bash
# Validate a single extraction
python validate_hierarchy.py --input extraction_output.json

# Validate all extractions and generate reports
python validate_all_hierarchies.py
```

These tools generate:
- Visual hierarchy representations
- Validation reports with errors and warnings
- Statistics on hierarchy structure

## Integration with LLMs

Proper parent-child relationships enable LLMs to:

1. **Navigate document structure**: Moving up and down the hierarchy
2. **Understand context**: Recognizing where code elements belong
3. **Connect related elements**: Identifying relationships between blocks
4. **Process document flow**: Following the logical order of content

Failure to maintain these relationships will result in degraded LLM performance during question answering and code understanding tasks.