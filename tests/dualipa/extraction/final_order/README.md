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

# DuaLipa Testing Guidelines

## Critical Rules

### 🚫 ABSOLUTELY FORBIDDEN
1. NEVER modify tests to make them pass
2. NEVER change file extensions or test data
3. NEVER weaken assertions or expected results
4. NEVER remove tests that expose bugs

### ✅ Correct Approach
1. Fix implementation to handle test cases correctly
2. Understand why tests are structured as they are
3. Respect test data choices (file types, extensions, etc.)
4. Use test failures to guide implementation

## Test Organization

### Progressive Complexity
Tests are organized to build complexity gradually:
```python
# 1. Basic functionality
test_01_simple.py  # Basic imports and setup

# 2. Core features
test_20_python_extractor.py    # Python extraction
test_30_js_ts_extraction.py    # JavaScript/TypeScript

# 3. Advanced features
test_31_tree_sitter_hierarchy.py  # Hierarchical extraction
```

### Test Data Integrity
- Test data is carefully chosen to verify specific behaviors
- File extensions matter and are deliberately selected
- Real-world examples are used to ensure practical functionality

Example:
```python
# This test deliberately uses a .js file and converts to .tsx
# to verify proper React/TSX component handling
test_file = Path("test_repos/react/.../ListItem.js")
tsx_file = temp_dir_path / "ListItem.tsx"
```

## Debugging Guide

1. **First Steps**:
   - Read the ENTIRE test file
   - Understand test progression
   - Verify test data exists
   - Check file paths and extensions

2. **Common Pitfalls**:
   - ❌ Changing test extensions
   - ❌ Modifying test data
   - ❌ Weakening assertions
   - ❌ Ignoring test docstrings

3. **Correct Approach**:
   - ✅ Fix implementation
   - ✅ Respect test requirements
   - ✅ Maintain test integrity
   - ✅ Document lessons learned

## Example: Tree-Sitter Integration

The `test_30_js_ts_extraction.py` file demonstrates proper test design:

1. **Progressive Complexity**:
   ```python
   test_js_function_extraction()    # Basic JS
   test_ts_class_extraction()       # TypeScript
   test_tsx_component_extraction()  # React/TSX
   ```

2. **Real-World Examples**:
   - Uses actual React components
   - Preserves file extensions
   - Tests practical use cases

3. **Clear Requirements**:
   ```python
   """
   This test verifies that:
   1. Tree-sitter correctly extracts React components
   2. Component content is preserved completely
   3. Methods aren't extracted separately
   """
   ```

Remember: Tests are documentation of expected behavior. They should never be modified to accommodate implementation bugs. 