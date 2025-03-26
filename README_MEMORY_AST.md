# Memory-Enhanced AST Extraction System

This document describes the implementation of the memory-enhanced AST extraction system that leverages tree-sitter for robust code structure extraction.

## Overview

The system combines the power of tree-sitter's AST parsing with an AI memory system to improve extraction quality over time. By tracking extraction patterns, errors, and successes, the system builds up knowledge that helps it handle complex code structures more effectively.

## Key Components

### 1. AST Extractor

The AST extractor (`ast_extractor.py`) is the core component that processes source code files using tree-sitter. It provides:

- Language-specific parsing for Python, JavaScript/TypeScript, Go, and Rust
- Rich structure extraction including nested classes, inheritance, and methods
- Fallback mechanisms for handling parsing failures

### 2. Memory Integration

The memory system is integrated through:

- `extraction_memory.py`: Domain-specific helpers for extraction memory
- `ai_memory_system.py`: Vector-based semantic search and state tracking
- `test_state_manager.py`: SQLite persistence layer

### 3. Test and Execution Scripts

- `run_ast_extraction.py`: Main execution script with QA format conversion
- `test_ast_extraction.py`: Comprehensive test harness for extraction testing

## Usage

### Basic Extraction

```bash
python run_ast_extraction.py path/to/file.py
```

This will extract the code structure, save it to `extraction_output.json`, and also create a QA-compatible version at `qa_compatible_output.json`.

### Testing Complex Structures

```bash
python test_ast_extraction.py file path/to/complex/code.py
```

For testing a directory with multiple files:

```bash
python test_ast_extraction.py dir path/to/directory --pattern "**/*.py"
```

## How Memory Improves Extraction

The memory system tracks:

1. **Extraction Patterns**: Successful extractions are saved as patterns that can be referenced in future extractions.

2. **Error Recovery**: When errors occur, the system records them and looks for similar past errors to suggest recovery actions.

3. **Context Tracking**: The extraction process maintains context about the current operation, improving continuity across extractions.

4. **Knowledge Accumulation**: The system accumulates knowledge about language-specific structures and complex patterns.

## Extraction Capabilities

The system successfully extracts:

- **Nested Classes**: Classes defined within other classes, multiple levels deep
- **Complex Inheritance**: Multi-level inheritance, mixins, and interface implementations
- **Decorators**: Method and class decorators with complex patterns
- **Type Information**: Generic types and type parameters when available

## QA Integration

The extraction output is automatically converted to a format compatible with the QA module, which can be used to generate question-answer pairs from the extracted code structures.

## Future Improvements

- Enhanced cross-file relationship tracking
- Incremental extraction based on file changes
- More specialized extractors for framework-specific code
- Multi-modal memory for improved context representation