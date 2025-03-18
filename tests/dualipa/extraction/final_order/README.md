# DuaLipa Pipeline Extraction Tests

This directory contains test files that are numbered according to their position in the pipeline execution order. This ensures that tests run in the correct sequence, respecting dependencies between pipeline stages.

## Pipeline Stages and Test Order

### Stage 1: Smoke Tests (01-09)
- `test_01_simple.py` - Basic smoke test to verify environment setup
- `test_02_import.py` - Verify that all modules can be imported

### Stage 2: Repository Operations (10-19)
- `test_10_github_utils.py` - Test GitHub repository downloading/cloning functionality

### Stage 3: Python AST Extraction (20-29) 
- `test_20_python_extractor.py` - Test extraction using Python's AST parser

### Stage 4: Tree-sitter Extraction (30-39)
- `test_30_js_ts_extraction.py` - Test JavaScript and TypeScript extraction
- `test_31_tree_sitter_hierarchy.py` - Test tree-sitter based extraction hierarchies

### Stage 5: General Extraction (40-49)
- `test_40_code_extractor.py` - Test main code extraction functionality
- `test_41_block_extractor.py` - Test code block extraction
- `test_42_block_extraction.py` - Test detailed block extraction

### Stage 6: Markdown Extraction (50-59)
- `test_50_markdown_parser.py` - Test basic markdown parsing
- `test_51_markdown_hierarchy.py` - Test markdown structure analysis
- `test_52_markdown_it_parser.py` - Test markdown-it parsing

### Stage 7: Verification and Integration (60-99)
- `test_60_block_verification.py` - Test verification of extracted blocks
- `test_61_code_hierarchy.py` - Test code structure analysis
- `test_70_multilang_extractor.py` - Test multiple language extraction
- `test_80_output_examples.py` - Test generation of examples
- `test_90_repo_operations.py` - Test full repository operations

## Why Test Order Matters

The DuaLipa pipeline processes data sequentially through multiple stages. Each stage depends on the output of previous stages:

1. **Repository acquisition** must happen before any extraction can occur
2. **Code extraction** must happen before block extraction
3. **Block extraction** must happen before verification and integration
4. **Verified data** must exist before final output can be generated

By running tests in the correct order, we avoid cascading failures where a test fails because earlier pipeline stages haven't been verified.

## Running Tests in Order

With this numbering scheme, tests will naturally run in the correct order when using:

```bash
pytest -v
```

Or to run a specific stage:

```bash
pytest -v test_3*.py  # Run only Stage 4 tree-sitter extraction tests
``` 