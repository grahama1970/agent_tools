# End-to-End Tests

This directory contains end-to-end tests that verify complete workflows in the Cursor Rules system.

## What Belongs Here

- Tests that validate entire user workflows
- Tests that verify system behavior from input to output
- Tests that simulate real-world usage patterns

## Test Files

- **`test_cursor_rules.py`**: Tests for the complete cursor rules workflow
- **`test_common_queries.py`**: Tests for common query patterns
- **`test_rule_search.py`**: Tests for rule search functionality
- **`test_scenario_management.py`**: Tests for scenario management
- **`test_test_state.py`**: Tests for test state management

## Import Structure for End-to-End Tests

End-to-end tests typically need to import from multiple modules across the codebase, requiring careful import management:

### Example of Correct Imports for End-to-End Tests

```python
# Import patterns for end-to-end tests
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Importing directly from the package (preferred for end-to-end tests)
from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    semantic_search,
    hybrid_search,
    EMBEDDING_AVAILABLE
)

from agent_tools.cursor_rules.embedding import (
    create_embedding_sync,
    ensure_text_has_prefix
)
```

### End-to-End Test-Specific Import Guidance

1. **Importing Multiple Components**:
   ```python
   # Importing all necessary components for complete workflow
   from agent_tools.cursor_rules.cursor_rules import (
       setup_cursor_rules_db,
       load_rules_from_directory,
       bm25_keyword_search,
       semantic_search,
       hybrid_search
   )
   ```

2. **Config and Environment Imports**:
   ```python
   import os
   from pathlib import Path
   from dotenv import load_dotenv
   
   # Load test environment variables
   load_dotenv()
   ```

3. **Temporary File and Directory Management**:
   ```python
   import tempfile
   import shutil
   
   @pytest.fixture
   def temp_rules_dir():
       temp_dir = tempfile.mkdtemp()
       yield temp_dir
       shutil.rmtree(temp_dir)
   ```

## Common Import Issues in End-to-End Tests

1. **Failing to Import All Required Components**:
   - ❌ Importing each component separately with inconsistent paths
   - ✅ Importing related components together with a consistent approach

2. **Import Conflicts Between Real and Test Components**:
   - ❌ Mixing imports without proper namespace management
   - ✅ Using clear namespaces or aliases:
     ```python
     from agent_tools.cursor_rules.cli import app as cli_app
     from agent_tools.cursor_rules.utils import db as db_utils
     ```

3. **Path Resolution Issues**:
   - ❌ Using hardcoded paths
   - ✅ Using relative path resolution:
     ```python
     # Correctly resolve paths relative to the test file
     TEST_DIR = Path(__file__).parent
     FIXTURES_DIR = TEST_DIR / 'fixtures'
     ```

## Best Practices from TESTING_LESSONS.md

### Learning from Existing Tests

1. **ALWAYS examine existing passing tests before writing new ones**
   - Existing tests provide valuable insights into how the system actually works
   - Passing tests demonstrate the correct patterns for interacting with the system
   - Don't reinvent testing patterns - reuse successful approaches from existing tests

2. **Understand the system through its tests**
   - Existing tests reveal assumptions and constraints that may not be documented elsewhere
   - When adding new functionality, start by understanding how similar functionality is tested
   - Consistency in testing patterns makes the test suite more maintainable

### Documentation-First Testing

1. **Start with documentation**
   - Read the official documentation for any library or component before testing it
   - Include documentation links in test files for reference
   - Understand the intended behavior before writing tests

2. **Progress to integration testing**
   - After components work in isolation, test their integration
   - Verify that components work together as expected
   - Test with real-world data and scenarios

### Real-World Verification

1. **Verify with real-world examples**
   - Test with actual data and scenarios
   - Don't rely solely on contrived examples
   - Ensure the system works in practice, not just in theory

2. **End-to-end testing**
   - Test the complete workflow from start to finish
   - Verify that all components work together correctly
   - Ensure the system produces the expected results

### Lessons from Our Experience

1. **Understand the implementation before testing**
   - Check how functions are implemented before writing tests
   - Look for specific checks like `has_collection()` vs. `collections()`
   - Mock the correct methods based on the actual implementation

2. **Verify mocks are working as expected**
   - Ensure mocked methods are actually being called
   - Use `assert_called_once()` to verify method calls
   - Check that mocked return values are being used

3. **Iterative testing approach**
   - Start with simple tests that verify basic functionality
   - Gradually add more complex tests
   - Fix issues as they arise before moving on

## Documentation References

- pytest: https://docs.pytest.org/
- python-arango: https://python-arango.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html 