# Stage 2: Code and Documentation Extraction [DATA EXTRACTION]

This directory contains tests for Stage 2 of the DuaLipa pipeline: Code and Documentation Extraction.

## Purpose

These tests verify that code and documentation are correctly extracted from repositories:

- Code block extraction using AST for Python
- Code block extraction using tree-sitter for other languages
- Markdown section extraction
- Language detection

## Components Tested

- `code_extractor.py`: Content filtering and processing (`extract_repository()`)
- `language_detection.py`: Programming language identification
- `markdown_parser.py`: Documentation parsing and section extraction

## Running the Tests

From the project root, run:

```bash
python -m pytest tests/dualipa/stage2
```

## Tests Overview

- `test_block_extraction.py`: Tests for block extraction functionality
- `test_block_extractor.py`: Tests for code block extraction
- `test_python_extractor.py`: Focused tests for Python code extraction
- `test_multilang_extractor.py`: Tests for multi-language code extraction
- `test_markdown_parser.py`: Tests for markdown parsing and section extraction 