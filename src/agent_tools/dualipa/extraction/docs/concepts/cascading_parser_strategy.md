# Cascading Parser Strategy for JS/TS Extraction

## Overview

This document details the implementation of a cascading parser approach for JavaScript and TypeScript extraction in the DuaLipa pipeline, addressing the reliability issues with tree-sitter.

## Problem Statement

Tree-sitter has proven particularly challenging for reliable JS/TS parsing due to:

1. **Parsing inconsistency** with complex TypeScript constructs
2. **Memory leaks** during large-scale extraction
3. **Initialization overhead** impacting performance
4. **Dependency conflicts** between tree-sitter packages

These issues resulted in extraction failures and inconsistent output that affected downstream QA generation quality.

## Solution: Cascading Parser Approach

The implementation follows a cascading extraction strategy that uses multiple parsing approaches in a fallback sequence:

```
┌───────────────┐     ┌─────────────┐     ┌──────────────┐
│  React        │     │  Tree-sitter │     │  Generic     │
│  Component    │────>│  AST-based   │────>│  Regex-based │
│  Detection    │     │  Extraction  │     │  Extraction  │
└───────────────┘     └─────────────┘     └──────────────┘
```

### 1. React Component Detection

As a first step, files are analyzed for React component signatures. If detected, the file is extracted as a React component block without requiring AST parsing.

### 2. Tree-sitter AST-based Extraction

If not a React component, tree-sitter is used for precise AST-based extraction:

1. The appropriate language parser is loaded (JavaScript/TypeScript)
2. The content is parsed into an AST
3. Node traversal extracts functions, classes, interfaces, and methods
4. Blocks are created with high-quality metadata indicators

### 3. Generic Regex-based Fallback

If tree-sitter fails or produces no blocks:

1. Switch to pattern-based extraction using predefined regular expressions
2. Extract functions, classes, and interfaces based on language-specific patterns
3. Mark blocks with `extraction_method: "generic_fallback"` and `extraction_quality: "low"`
4. Record fallback usage in statistics

## Implementation Details

### Added JavaScript/TypeScript Patterns

Regular expressions were added to the `PATTERNS` dictionary in `generic_extractor.py`:

```python
# JavaScript function patterns
"javascript": r"(?:function\s+)(\w+)\s*\([^)]*\)\s*{|(?:const|let|var)\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*{|(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",

# TypeScript function patterns (includes type annotations)
"typescript": r"(?:function\s+)(\w+)(?:<[^>]*>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*{|(?:const|let|var)\s+(\w+)(?:<[^>]*>)?\s*=\s*function\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*{|(?:const|let|var)\s+(\w+)(?:<[^>]*>)?\s*=\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*=>\s*{",

# JavaScript class patterns
"javascript": r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*{",

# TypeScript class and interface patterns
"typescript": r"class\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+\w+(?:<[^>]*>)?)?(?:\s+implements\s+[^{]+)?\s*{|interface\s+(\w+)(?:<[^>]*>)?\s*(?:extends\s+[^{]+)?\s*{",
```

### New Fallback Function

Added `_fallback_to_generic_extractor` function in `js_ts_extractor.py`:

```python
def _fallback_to_generic_extractor(file_path: str, content: str, language: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fallback to generic pattern-based extraction when tree-sitter fails."""
    # Create temporary file
    # Run generic extraction
    # Add extraction quality metadata
    # Return blocks and stats
```

### Enhanced Extract Function

Modified `extract_js_ts_blocks` function to implement the cascading approach:

1. Check for React components first
2. Try tree-sitter parsing
3. Fall back to generic extraction if tree-sitter fails
4. Add final exception fallback as last resort

### Statistics Tracking

Enhanced `merge_stats` in `stats_utils.py` to:
- Merge statistics from different extraction methods
- Track fallback usage with a dedicated flag
- Preserve extraction quality indicators for downstream awareness

## Metadata Additions

The following metadata fields were added to extracted blocks:

```json
"metadata": {
    "extraction_method": "tree_sitter" | "generic_fallback",
    "extraction_quality": "high" | "low",
    "language": "javascript" | "typescript",
    "file": "/path/to/file.js"
}
```

This metadata enables downstream components (like the QA module) to adjust their processing based on the extraction quality.

## Testing Strategy

The implementation includes comprehensive testing with:

1. **Simple TypeScript files** that work well with tree-sitter
2. **Complex TypeScript files** with generics, mapped types, and complex React components
3. **Intentionally malformed code** to test the fallback mechanism
4. **Edge cases** like JSX/TSX files and files with mixed content

## Impact on QA Module

The cascading parser approach provides several benefits for the QA module:

1. **Increased reliability** by extracting content even when tree-sitter fails
2. **Quality awareness** through metadata that indicates extraction confidence
3. **Graceful degradation** by prioritizing content completeness over parsing perfection
4. **Format consistency** by using the same output structure regardless of extraction method

## Future Improvements

Future enhancements could include:

1. **Hybrid extraction** that combines results from multiple parsing methods
2. **Partial recovery** for extracting usable portions from files that partially fail
3. **Alternative parsers** like ESTree/Acorn or TypeScript Compiler API
4. **Extraction quality confidence scores** using more granular metrics
5. **Performance optimization** through caching and lazy loading