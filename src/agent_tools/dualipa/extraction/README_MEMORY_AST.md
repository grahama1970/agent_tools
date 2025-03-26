# AST Extraction with Memory Integration

This module provides advanced AST-based code structure extraction using tree-sitter with memory integration. The system builds on the existing memory framework to enhance code extraction capabilities by:

1. Using abstract syntax trees (ASTs) for accurate code parsing
2. Tracking extraction patterns and errors in memory
3. Learning from past extractions to improve future results
4. Supporting complex code structures like nested classes and inheritance
5. Providing memory-based insights for extraction processes

## Key Components

### AST Extractor

The `AstExtractor` class in `extractors/code/ast_extractor.py` provides a comprehensive extraction system with:

- Memory-aware extraction with error tracking and learning
- Language-agnostic interface with specialized extractors
- Rich structure extraction with nested elements and relationships
- Fallback mechanisms for robust extraction

### Memory System Integration

The extraction process integrates with the memory system through:

- `TestStateManager`: Base SQLite persistence layer
- `EnhancedMemory`: Advanced search capabilities
- `AIMemorySystem`: Vector embedding integration
- `ExtractionMemory`: Domain-specific extraction helpers

### Memory-Aware Extraction

The memory integration enables:

1. **Pattern Learning**: Learn successful extraction patterns over time
2. **Error Tracking**: Remember and suggest fixes for common errors
3. **Context Awareness**: Maintain context between extraction runs
4. **Relationship Discovery**: Find connections between code elements

## Supported Languages

The AST extractor supports multiple languages with specialized extractors:

- **Python**: Classes, nested classes, inheritance, decorators
- **JavaScript/TypeScript**: Classes, interfaces, functions, exports
- **Go**: Packages, structs, interfaces, functions
- **Rust**: Modules, structs, traits, functions

## Usage Examples

### Single File Extraction

```python
from src.agent_tools.dualipa.extraction.extractors.code.ast_extractor import AstExtractor
from src.agent_tools.dualipa.extraction.extraction_memory import init_extraction_memory

# Initialize with memory
init_extraction_memory("extraction_memory.db")
extractor = AstExtractor(memory_db_path="extraction_memory.db")

# Extract file
result = extractor.extract_file("path/to/file.py")
```

### Bulk Directory Extraction

```python
from src.agent_tools.dualipa.extraction.extractors.code.ast_extractor import AstExtractor
from src.agent_tools.dualipa.extraction.extraction_memory import init_extraction_memory

# Initialize with memory
init_extraction_memory("extraction_memory.db")
extractor = AstExtractor(memory_db_path="extraction_memory.db")

# Extract all files in a directory
results = extractor.extract_directory("path/to/repo", 
                                    languages=['python', 'javascript'])
```

## Testing & Evaluation

The system includes several testing utilities:

1. `test_ast_extraction.py`: A comprehensive test harness
2. `single_test.py`: Single file testing with memory inspection
3. `end_to_end_extraction_ast.py`: End-to-end testing pipeline

To run the end-to-end test:

```bash
python end_to_end_extraction_ast.py
```

To test a single file with memory integration:

```bash
python single_test.py path/to/file.py
```

## Architecture

The architecture follows a layered approach:

1. **Core Extraction**: AST parsing using tree-sitter
2. **Language Specialization**: Language-specific extraction logic
3. **Memory Integration**: State tracking through extraction_memory
4. **Output Formatting**: Standardized structure output compatible with QA systems

## Future Enhancements

- Cross-file relationship tracking
- Enhanced embedding-based semantic search for code structures
- Multi-modal memory for source code representations
- Incremental extraction based on file changes

## Dependencies

- tree-sitter: For AST parsing
- tree-sitter-language-pack: Prebuilt language parsers
- SQLite: For persistence layer
- sentence-transformers (optional): For improved semantic search