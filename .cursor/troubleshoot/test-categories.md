# Test Categories and Organization

## Component Tests (`test_component_*.py`)
- Focus on individual unit functionality
- Mock external dependencies
- Fast execution
- Should run first in test order

## Real-World Tests (`test_realworld_*.py`)
- Test full integration scenarios
- Use actual external services
- Produce verifiable outputs
- Should run after component tests

## Test Ordering
- Use `@pytest.mark.order(n)` for explicit ordering
- Critical initialization tests should use `order(0)`
- File naming convention helps with natural ordering
- Cursor rules test must run first (`test_000_cursor_rules.py`)

## Test Dependencies
- Use `@pytest.mark.depends(on=["test_name"])` for dependencies
- Document dependencies in test docstrings
- Avoid circular dependencies
- Keep dependency chains short

## Simplified Test Collection
The project uses a simplified approach to test collection in `conftest.py`:

```python
def pytest_collection_modifyitems(session, config, items):
    """
    Simple hook to ensure tests with explicit order markers run first.
    This relies on the pytest-order plugin for the actual ordering logic.
    
    Tests without order markers will run after tests with order markers.
    """
    # The pytest-order plugin will handle the ordering of tests with order markers
    # We don't need to do anything special here
    pass
```

This approach:
1. Relies on the pytest-order plugin for ordering
2. Avoids complex manual filtering of tests
3. Ensures all tests are included in the test run
4. Prevents silent test exclusion

## Adding New Tests
When adding new tests:
1. Follow the naming convention (`test_component_*.py` or `test_realworld_*.py`)
2. Use `@pytest.mark.order(n)` if specific ordering is required
3. Use `@pytest.mark.depends(on=["test_name"])` for dependencies
4. Document test requirements in docstrings
5. Ensure tests can run in isolation 