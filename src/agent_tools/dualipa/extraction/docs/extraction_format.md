# Extraction Format Specification

## Block Format

All extracted blocks must follow this format to ensure compatibility with downstream processing (especially the QA module):

```python
{
    "uuid": str,          # Unique identifier - Must be consistent and preserved
    "id": str,            # Alias for uuid (for backward compatibility)
    "type": str,          # Block type (function, class, method, section, code)
    "name": str,          # Block name - Used for QA context and reference
    "content": str,       # Block content - Complete and well-formatted
    "language": str,      # Programming language - Affects QA strategies
    "metadata": {         # Block metadata - Critical for QA generation
        "line_start": int,        # Starting line number
        "line_end": int,          # Ending line number
        "language": str,          # Programming language - Affects QA strategies 
        "imports": List[str],     # Import statements - Provides crucial context
        "source_file": str,       # Source file path - For traceability and context
        ...                       # Additional metadata
    }
}
```

### Field Standardization

The extraction module performs field name standardization to ensure consistency:

1. **ID Fields**:
   - `uuid` is the primary identifier field
   - `id` is maintained as an alias for backward compatibility
   - Both fields will contain the same value

2. **Path Fields**:
   - `source_file` is the standardized field for file paths
   - `file` and `path` are automatically converted to `source_file`

3. **Line Number Fields**:
   - `line_start` and `line_end` are standard field names
   - `start_line` and `end_line` are automatically converted

4. **Language Field**:
   - `language` exists at both the top level and in metadata
   - This redundancy ensures compatibility with different consumers

> **Important**: The QA module depends on this structure for context-aware question generation. Missing or incorrect fields will directly impact QA quality.

## Output Formats

The extraction module supports multiple output formats:

1. **JSON Format**
```python
format_output_as_json(extraction_data) -> str
```
Produces a standardized JSON representation with all blocks and statistics.

2. **Markdown Format**
```python
format_output_as_md(extraction_data) -> str
```
Produces a human-readable markdown document with code blocks properly formatted with language syntax highlighting.

3. **HTML Format**
```python
format_output_as_html(extraction_data) -> str
```
Produces a standalone HTML document with styled code blocks and navigation.

4. **Generic Format**
```python
format_output(extraction_data, format_type: str) -> str
```
Unified interface supporting 'json', 'md', and 'html' format types.

## Block Types

1. **Function Block**
```python
{
    "type": "function",
    "name": "function_name",
    "content": "def function_name():\n    ...",
    "metadata": {
        "line_start": 1,
        "line_end": 5,
        "language": "python",
        "imports": ["from typing import List"],
        "source_file": "script.py",
        "returns": "Optional[str]",  # Return type if available
        "decorators": ["@staticmethod"]  # Decorators if present
    }
}
```

2. **Class Block**
```python
{
    "type": "class",
    "name": "ClassName",
    "content": "class ClassName:\n    ...",
    "metadata": {
        "line_start": 1,
        "line_end": 10,
        "language": "python",
        "imports": ["from typing import Dict"],
        "source_file": "script.py",
        "bases": ["BaseClass"],  # Base classes
        "decorators": ["@dataclass"]  # Decorators if present
    }
}
```

3. **Method Block**
```python
{
    "type": "method",
    "name": "method_name",
    "content": "def method_name(self):\n    ...",
    "metadata": {
        "line_start": 5,
        "line_end": 8,
        "language": "python",
        "imports": ["from typing import Any"],
        "source_file": "script.py",
        "class_name": "ClassName",  # Parent class
        "returns": "None",  # Return type if available
        "decorators": ["@property"]  # Decorators if present
    }
}
```

4. **React Component Block**
```python
{
    "type": "react_component",
    "name": "ComponentName",
    "content": "export function ComponentName() {\n    ...",
    "metadata": {
        "line_start": 1,
        "line_end": 20,
        "language": "typescript",
        "imports": ["import React from 'react'"],
        "source_file": "Component.tsx",
        "framework": "react"  # Framework identifier
    }
}
```

5. **Section Block**
```python
{
    "type": "section",
    "name": "Section Title",
    "content": "# Section Title\n...",
    "metadata": {
        "line_start": 1,
        "line_end": 10,
        "language": "markdown",
        "source_file": "README.md",
        "level": 1,  # Heading level
        "breadcrumb": ["Title", "Section Title"],
        "has_code": True  # Contains code blocks
    }
}
```

## Statistics Format

All extraction operations must track statistics:

```python
{
    "total_blocks": int,      # Total blocks extracted
    "total_files": int,       # Total files processed
    "languages": {            # Language statistics
        "python": int,        # Files per language
        "typescript": int,
        ...
    },
    "block_types": {          # Block type counts
        "function": int,
        "class": int,
        "method": int,
        ...
    },
    "errors": List[str],      # Error messages
    "file_blocks": {          # Blocks by file
        "file.py": List[Dict]
    },
    "extraction_time": float, # Time taken in seconds
    "validation_errors": int, # Number of validation errors
    "verification_errors": int # Number of verification errors
}
```

## Error Handling

1. **File Errors**
```python
{
    "errors": [
        "File not found: script.py",
        "Failed to parse file: syntax error"
    ]
}
```

2. **Validation Errors**
```python
{
    "validation_errors": 2,
    "errors": [
        "Missing required field: type",
        "Invalid UUID format"
    ]
}
```

3. **Verification Errors**
```python
{
    "verification_errors": 1,
    "errors": [
        "Python syntax error: invalid syntax"
    ]
}
```

## Integration with QA Module

The extraction output serves as direct input to the QA module, which generates question-answer pairs based on the extracted content. The following considerations are critical for ensuring high-quality QA generation:

1. **Content Completeness**
   - Code blocks must be syntactically valid to generate meaningful questions
   - Context (imports, surrounding code) affects QA quality significantly
   - Section content should preserve hierarchical structure for proper context

2. **Performance Considerations**
   - Large extraction outputs can impact QA module performance
   - The QA module implements adaptive worker pools and chunk-based processing based on extraction size
   - Section types are sorted for better cache locality in the QA process
   - Memory consumption scales with extraction size, requiring resource-aware processing

3. **Error Propagation**
   - Extraction errors cascade into QA generation issues
   - Tree-sitter parsing failures can produce incomplete or incorrect blocks
   - QA generation requires recovery strategies for handling extraction imperfections

4. **Bidirectional Processing**
   - The QA module supports bidirectional generation (Q→A and A→Q)
   - Extraction format must provide sufficient context for both directions
   - Code structure information affects reversal quality in QA pairs

## Best Practices

1. **Content Handling**
   - Always dedent content using `textwrap.dedent()`
   - Preserve original indentation in class methods
   - Keep complete content for React components
   - Handle multi-line strings properly
   - **Ensure code blocks compile/parse** - The QA module works best with valid code

2. **Metadata**
   - Include all required fields
   - Add language-specific metadata
   - Track imports for context
   - Include source information
   - **Add relationship information** - Class-method relationships provide critical context for QA

3. **Error Handling**
   - Log all errors with context
   - Track error counts in stats
   - Provide helpful error messages
   - Handle edge cases gracefully
   - **Implement fallback extraction** - When tree-sitter fails, use alternative methods

4. **Statistics**
   - Track all block types
   - Track language usage
   - Monitor error rates
   - Include performance metrics
   - **Track extraction complexity** - Helps predict QA generation resource needs

5. **Performance Optimization**
   - **Consider section sorting** - The QA module sorts by section type for cache locality
   - **Implement chunking for large repositories** - Helps manage memory consumption
   - **Track dependency information** - Improves context-aware QA generation
   - **Use consistent IDs** - Ensures proper tracking through the pipeline

## Tree-Sitter Challenges and Mitigations

Tree-sitter has proven challenging for reliable JavaScript/TypeScript parsing, particularly with complex codebases. Based on our experience with the QA module, we recommend the following approaches:

1. **Fallback Strategy Implementation**
   - Implement a cascading parser approach - try tree-sitter first, then fallback to regex-based extraction
   - Add generic pattern-based extraction as a final fallback
   - Always prioritize complete extraction over perfect parsing

2. **Error Detection and Recovery**
   - Monitor tree-sitter parsing failures and log detailed error information
   - Implement partial extraction recovery to salvage usable content from failed parses
   - Flag extraction issues in metadata for QA module awareness

3. **Alternative Approaches**
   - Consider ESTree/Acorn for JavaScript as an alternative parser
   - For TypeScript, investigate TypeScript Compiler API as a more reliable alternative
   - For JSX/TSX content, specialized React extractors may be more effective

4. **Performance Considerations**
   - Tree-sitter initialization is expensive - implement a parser pool
   - Cache parsed ASTs for repeated access
   - Use incremental parsing for large files

5. **Testing and Validation**
   - Create comprehensive test suite focused on edge cases
   - Include validation step that verifies extracted code can be reparsed
   - Implement extraction quality metrics to monitor production usage 