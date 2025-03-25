# Parent-Child Relationship Requirements

This document outlines the requirements for parent-child relationships in the DuaLipa extraction module. These requirements are enforced by the hierarchy validation framework to ensure proper structure of extraction outputs.

## 1. Bidirectional References

All parent-child relationships must be bidirectional:

1. **Parent → Child**: Parent blocks must list their children in the `child_uuids` field.
2. **Child → Parent**: Child blocks must reference their parent in the `parent_uuid` field.

Example:
```json
// Parent block
{
  "uuid": "parent-uuid-123",
  "name": "Parent Block",
  "type": "documentation",
  "child_uuids": ["child-uuid-456", "child-uuid-789"]
}

// Child block
{
  "uuid": "child-uuid-456",
  "name": "Child Block",
  "type": "doc_page",
  "parent_uuid": "parent-uuid-123"
}
```

## 2. Block Type Hierarchy

Parent-child relationships must follow the hierarchical document structure:

1. **Documentation Hierarchy**:
   - `documentation` → `doc_page` → `doc_section` → `code_block`/`table`

2. **Code Hierarchy**:
   - `file` → `class` → `method`/`function`
   - `file` → `function`

3. **Type Relationship Matrix**:

   | Parent Type      | Allowed Child Types                          |
   |-------------------|----------------------------------------------|
   | `documentation`   | `doc_page`                                   |
   | `doc_page`        | `doc_section`                                |
   | `doc_section`     | `doc_section`, `code_block`, `table`         |
   | `file`            | `class`, `function`                          |
   | `class`           | `method`, `function`, `property`             |

## 3. UUID Requirements

1. **Uniqueness**: UUIDs must be unique across all blocks.
2. **Format**: UUIDs must follow a valid UUID format or be consistently identifiable strings.
3. **Persistence**: UUIDs must remain consistent in subsequent extractions of the same content.

## 4. Root Blocks

Certain blocks are expected to be root blocks (no parent):

1. `documentation` blocks for document hierarchies
2. `file` blocks for code hierarchies

Root blocks should have:
- `parent_uuid` set to `null` or missing
- A unique, stable UUID
- A clear descriptive name

## 5. Common Problems to Avoid

1. **Circular References**: A block cannot be both a parent and child of another block.
2. **Orphaned Blocks**: All blocks should be part of the hierarchy (exception: root blocks).
3. **Missing References**: All child UUIDs must reference valid blocks.
4. **Type Mismatches**: Child blocks must be of allowed types for their parent.

## 6. Metadata Requirements

Blocks should include hierarchical metadata:

1. **Documentation Blocks**:
   - `section_hierarchy`: Array of section titles from root to current
   - `level`: Nesting level (1, 2, 3, etc.)

2. **Code Blocks**:
   - `class_name`: For methods, the containing class name
   - `file_path`: Path to the source file

Example:
```json
{
  "uuid": "section-uuid-123",
  "name": "Method Description",
  "type": "doc_section",
  "parent_uuid": "page-uuid-456",
  "metadata": {
    "section_hierarchy": ["Class Documentation", "Methods", "Method Description"],
    "level": 3
  }
}
```

## 7. Special Block Types

1. **Table Blocks**:
   - Must have a parent `doc_section`
   - Content should be structured (headers/rows)

2. **Code Blocks**:
   - Must have a parent `doc_section` or `function` or `method`
   - Must include language identification

## 8. Validation Thresholds

The validation framework allows specifying thresholds:

1. **Structure Validation**: Default 75% pass rate
2. **Content Validation**: Default 85% pass rate
3. **Structure Consistency**: Default 75% pass rate

These thresholds determine if validation passes or fails for each category.

## 9. Format Variations

The framework supports different extraction formats with type mappings:

```
"documentation" → ["documentation", "file"]
"doc_page" → ["doc_page", "file"]
"doc_section" → ["doc_section", "section"]
"code_block" → ["code_block", "code"]
"table" → ["table"]
```

This allows validating extractions with slight type naming variations.

## 10. Implementation Guidelines

When implementing extraction logic:

1. Always set both parent_uuid and child_uuids fields
2. Validate hierarchy before returning results
3. Follow the type hierarchy defined above
4. Ensure uniqueness of UUIDs
5. Include comprehensive metadata
6. Test with the hierarchy validation tools