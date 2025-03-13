# Unit Tests

This directory contains isolated unit tests for individual components of the Cursor Rules system.

## What Belongs Here

- Tests that focus on a single function or class
- Tests with minimal or mocked external dependencies
- Tests that validate core logic in isolation

## Test Files

- **`test_field_translation.py`**: Tests for field translation utilities
- **`test_query_construction.py`**: Tests for query building functionality
- **`test_rule_loading.py`**: Tests for loading rule definitions
- **`test_arango_connection.py`**: Tests for database connection utilities

## Import Structure for Unit Tests

Unit tests should use the appropriate import patterns based on the `pythonpath = ["src"]` setting in `pyproject.toml`:

### Example of Correct Imports for Unit Tests

```python
# Example from test_field_translation.py
import pytest
from agent_tools.cursor_rules.utils.field_translation import translate_fields

# OR using relative imports
# from ...cursor_rules.utils.field_translation import translate_fields

def test_translate_fields():
    # Test implementation
    pass
```

### Unit Test-Specific Import Guidance

1. **Prefer absolute imports for clarity**: 
   ```python
   from agent_tools.cursor_rules.utils import some_function
   ```

2. **Mock external dependencies**: For true unit tests, external dependencies should be mocked:
   ```python
   @pytest.fixture
   def mock_db():
       return Mock()
   
   def test_function(mock_db):
       # Test with mock dependencies
       pass
   ```

3. **Import test utilities from conftest.py**: Fixtures defined in the closest conftest.py are automatically available.

## Best Practices

1. **Isolate dependencies** - Mock external services and databases
2. **Focus on single components** - Test one function or class per test
3. **Test edge cases** - Include tests for failure modes
4. **Reference documentation** - Include links to relevant documentation
5. **Avoid database integration** - Use mocks instead of real database connections

## Documentation References

- pytest: https://docs.pytest.org/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html 