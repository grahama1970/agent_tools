# Deprecated Tests

This directory contains test files that have been deprecated but are kept for reference until fully integrated or removed.

## What's Here

- Duplicate test files that have been consolidated into other directories
- Tests that use outdated patterns or approaches
- Tests that may contain useful code snippets for reference

## Test Files

- **`test_bm25_search_from_tests.py`**: Duplicate of integration/test_bm25_search.py
- **`test_bm25_search_simple.py`**: Simplified version of BM25 search tests
- **`test_cursor_rules_from_tests.py`**: Duplicate of end_to_end/test_cursor_rules.py
- **`test_vector_functions_from_tests.py`**: Duplicate of integration/test_vector_functions.py

## Import Structure Issues in Deprecated Tests

These files often demonstrate problematic import patterns that should be avoided:

### Examples of Problematic Import Patterns

```python
# AVOID: Direct import that assumes files are in a specific location
from src.agent_tools.cursor_rules.utils import some_function  # ❌ Incorrect

# AVOID: Inconsistent relative imports
from ..utils import some_function  # ❌ May break if directory structure changes
```

### Common Import Issues Found in Deprecated Tests

1. **Incorrect Directory References**:
   - ❌ `from src.agent_tools...` - relies on 'src' being in path
   - ❌ Incorrect number of dots in relative imports 
   - ✅ Should use `from agent_tools.cursor_rules...` or well-managed relative imports

2. **Mixed Import Styles**:
   - ❌ Mixing absolute and relative imports inconsistently
   - ✅ Should use consistent import style throughout a file

3. **Missing Dependency Checks**:
   - ❌ Assuming all dependencies are installed without checks
   - ✅ Should use try/except for optional dependencies:
     ```python
     try:
         import some_optional_package
         HAS_OPTIONAL_PACKAGE = True
     except ImportError:
         HAS_OPTIONAL_PACKAGE = False
     ```

## Learning from Import Mistakes

Review these files to understand what **not** to do in your tests. Common patterns to avoid:

1. **Hardcoded Paths**: Avoid hardcoded paths that won't work in different environments
2. **Direct src References**: Avoid direct references to the 'src' directory 
3. **Inconsistent Relative Imports**: Don't use relative imports inconsistently
4. **Missing Module Error Handling**: Always handle potential import errors for optional dependencies

## Why Keep These Files?

1. **Historical reference** - These files show the evolution of testing approaches
2. **Code snippets** - They may contain useful patterns or edge cases not yet integrated
3. **Transition period** - Keeping them temporarily ensures no functionality is lost during reorganization

## Next Steps

These files should be reviewed to ensure all valuable test cases have been integrated into the main test directories. Once confirmed, they can be safely removed.

## Best Practice from LESSONS_LEARNED.md

From the "Phase Completion and Test Validation" section:

> "All tests for the current phase must pass before moving to the next phase"
> "**CRITICAL ERROR: Moving to a new phase while tests in the current phase are failing**"

This directory helps maintain this principle by keeping old tests accessible while transitioning to a more organized structure. 