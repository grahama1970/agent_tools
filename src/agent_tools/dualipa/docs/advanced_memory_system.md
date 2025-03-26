# Advanced Memory System With Tree-sitter Integration

This document outlines the next steps for enhancing our AI memory system with robust AST-based parsing using tree-sitter for more accurate code structure extraction.

## Current Status

We have successfully implemented a comprehensive memory system with:

1. **SQLite-based state persistence** - Reliable storage of extraction state
2. **Vector embedding integration** - Semantic search capabilities with proper embeddings
3. **Error pattern recognition** - Learning from past extraction errors
4. **Relationship discovery** - Automatic connection of related knowledge items
5. **Memory CLI** - Command-line interface for interacting with the memory system
6. **Extraction helpers** - Simplified API for extraction workflows

The current extraction capabilities use a basic line-by-line approach for Python and a regex-based approach for JavaScript/TypeScript extraction, which works well for simple structures but struggles with:

- Nested class definitions
- Complex function signatures
- Method decorators
- Complex inheritance hierarchies
- Generic type parameters
- Conditional exports/imports

## Tree-sitter Integration Plan

### Phase 1: Enhanced Code Structure Extraction

1. **Create the AST-based Extraction Module**
   - Implement `AstExtractor` class in `src/agent_tools/dualipa/extraction/extractors/code/ast_extractor.py`
   - Leverage existing `tree_sitter_utils.py` utilities
   - Support Python, JavaScript, TypeScript, Go, and Rust initially

2. **Design a Unified Structure Format**
   - Define a consistent output structure for all languages
   - Include: classes, methods, functions, imports, exports, and relationships
   - Support nested structures (classes within classes, etc.)

3. **Implement Memory-Aware Error Handling**
   - Record specific tree-sitter parsing failures in memory
   - Implement cascading fallback mechanisms
   - Learn from past errors to improve extraction success

### Phase 2: Memory Integration

1. **Extend Extraction Memory API**
   - Add AST-specific memory helpers in `extraction_memory.py`
   - Track AST extraction statistics
   - Store successful extraction patterns

2. **Implement Structural Relationship Discovery**
   - Use AST information to map relationships between code elements
   - Store inheritance hierarchies
   - Track import/export dependencies
   - Create a navigable code graph

3. **Add Semantic Code Search**
   - Enable search by code structure (e.g., "find classes that implement X")
   - Support code-specific filtering
   - Leverage embeddings for "similar code" search

### Phase 3: End-to-End Implementation

1. **Update End-to-End Extraction Pipeline**
   - Modify `end_to_end_extraction.py` to use AST extraction
   - Add support for more languages
   - Implement batched processing for better performance

2. **Create QA-Compatible Output Enhancements**
   - Extend JSON output format with AST-derived relationships
   - Include type information
   - Add method signatures and parameter details
   - Preserve docstrings and comments

3. **Add Visualization Tools**
   - Create tools to visualize code structures
   - Generate hierarchical diagrams
   - Display relationship graphs

## Implementation Details

### AstExtractor Class Structure

```python
class AstExtractor:
    """AST-based code structure extraction using tree-sitter."""
    
    def __init__(self, memory_db_path=None):
        # Initialize memory system
        self.memory_available = False
        if memory_db_path:
            try:
                from src.agent_tools.dualipa.extraction.extraction_memory import (
                    init_extraction_memory,
                    track_extraction_start,
                    record_extraction_error
                )
                init_extraction_memory(memory_db_path)
                self.memory_available = True
            except ImportError:
                pass
                
        # Initialize parsers
        self.parsers = {}
        
    def extract_file(self, file_path, language=None):
        """Extract code structure from a file with memory integration."""
        # Determine language if not provided
        if not language:
            language = self._detect_language(file_path)
            
        # Initialize parser
        parser = self._get_parser(language)
        if not parser:
            return {"error": f"Unsupported language: {language}"}
            
        # Parse file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the content
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Extract structure based on language
            if language in ['python']:
                return self._extract_python_ast(file_path, content, tree)
            elif language in ['javascript', 'typescript']:
                return self._extract_js_ts_ast(file_path, content, tree)
            # Add more language handlers
            
        except Exception as e:
            # Record error in memory
            if self.memory_available:
                record_extraction_error(
                    f"{language}_extraction_error",
                    f"Error extracting AST: {str(e)}",
                    file_path,
                    severity=7
                )
            return {"error": str(e), "file_path": file_path, "language": language}
    
    def _extract_python_ast(self, file_path, content, tree):
        """Extract Python code structure from AST."""
        # Implementation details for Python extraction
        
    def _extract_js_ts_ast(self, file_path, content, tree):
        """Extract JavaScript/TypeScript code structure from AST."""
        # Implementation details for JS/TS extraction
```

### Memory Integration Enhancement

Extend the extraction memory API to support AST-specific operations:

```python
def track_ast_extraction_success(file_path, language, structure_type, details=None):
    """
    Record successful AST extraction pattern.
    
    Args:
        file_path: Path to the file
        language: Programming language
        structure_type: Type of structure extracted (class, function, etc.)
        details: Optional extraction details
    """
    if not MEMORY_AVAILABLE:
        return "Memory not available"
        
    return save_extraction_knowledge(
        f"ast_pattern_{language}_{structure_type}",
        f"# Successful {language} {structure_type} extraction pattern\n\n"
        f"File: {file_path}\n\n"
        f"Details: {details or 'No details provided'}",
        summary=f"Successful {language} {structure_type} extraction pattern",
        tags=["ast", "extraction", language, structure_type]
    )
```

### Code Relationship Tracking

Add relationship tracking to store code dependencies:

```python
def store_code_relationship(source, target, relationship_type, details=None):
    """
    Store a relationship between code elements.
    
    Args:
        source: Source code element (class, function, etc.)
        target: Target code element
        relationship_type: Type of relationship (inherits, calls, imports, etc.)
        details: Optional relationship details
    """
    if not MEMORY_AVAILABLE:
        return "Memory not available"
        
    topic = f"code_rel_{relationship_type}_{source}__{target}"
    
    return save_docs(
        topic,
        f"# Code Relationship: {source} {relationship_type} {target}\n\n"
        f"Source: {source}\n"
        f"Target: {target}\n"
        f"Relationship: {relationship_type}\n"
        f"Details: {details or 'No additional details'}",
        summary=f"{source} {relationship_type} {target}",
        tags=["code", "relationship", relationship_type],
        related=[f"code_{source}", f"code_{target}"]
    )
```

## Integration Testing Plan

1. Create test cases with a variety of complex code structures:
   - Nested classes
   - Complex inheritance hierarchies
   - Decorated methods
   - Generic types
   - Multi-file dependencies

2. Test memory-aware extraction with error recovery:
   - Introduce deliberate parsing errors
   - Verify fallback mechanisms work
   - Confirm error patterns are learned

3. Verify relationship discovery:
   - Check inheritance hierarchies are captured
   - Verify import/export relationships
   - Validate cross-file dependencies

## Expected Benefits

1. **More Accurate Extraction**
   - Proper handling of nested structures
   - Preservation of relationships between code elements
   - Better type information

2. **Improved Memory System**
   - Richer documentation of code structures
   - Better error recovery through learning
   - More meaningful semantic search

3. **Enhanced QA Integration**
   - More detailed structure information for QA generation
   - Better context for code-related questions
   - Support for "how does this work with that" type questions

4. **Performance and Reliability**
   - More reliable extraction of complex codebases
   - Better error handling and recovery
   - Efficient processing of large repositories