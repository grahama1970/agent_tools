# Cursor Rules Test Organization

This directory contains tests for the Cursor Rules system, organized according to the principles outlined in `TESTING_LESSONS.md` and `LESSONS_LEARNED.md`.

## Test Directory Structure

- **`unit/`**: Single-component unit tests with minimal dependencies
- **`integration/`**: Tests that integrate multiple components, often with database access
- **`end_to_end/`**: Full system tests that verify complete workflows
- **`cli/`**: CLI-specific tests for command line interfaces
- **`ai_knowledge/`**: Tests focused on AI knowledge integration
- **`models/`**: Test models and sample data
- **`deprecated/`**: Old test files kept for reference until fully integrated

## Reorganization Summary (March 11, 2023)

The test directory has been reorganized to follow best practices:

1. **Files Relocated by Category**:
   - Unit tests moved to `unit/`
   - Integration tests moved to `integration/`
   - End-to-end tests moved to `end_to_end/`
   - CLI tests moved to `cli/`
   - AI knowledge tests moved to `ai_knowledge/`
   - Duplicate/deprecated tests moved to `deprecated/`

2. **Documentation Added**:
   - README files added to each directory explaining:
     - Purpose of the directory
     - Import patterns and guidance
     - Common import issues to avoid
     - Best practices for that test category
     - Documentation references

3. **Import Structure Clarified**:
   - All READMEs now document correct import patterns
   - Examples of import mistakes to avoid
   - Guidance on using relative vs. absolute imports

4. **conftest.py Updated**:
   - Improved documentation and organization
   - Added shared database test fixture
   - Set proper event_loop scope
   - Organized imports by category

5. **Next Steps**:
   - Review deprecated tests to ensure all valuable test cases are preserved
   - Update any remaining tests with consistent import patterns
   - Ensure all tests pass with the new organization

## Import Structure

The project uses a src-based layout with the following import configuration in `pyproject.toml`:

```python
[tool.pytest.ini_options]
pythonpath = ["src"]
```

This means that when running tests, the `src` directory is automatically added to the Python path.

### Correct Import Patterns

1. **Absolute imports from src**:
   ```python
   from agent_tools.cursor_rules.utils import some_function
   ```

2. **Relative imports within the test directory**:
   ```python
   from ...cursor_rules.utils import some_function  # From a test file to src modules
   from ..fixtures import some_fixture  # From a test file to another test module
   ```

3. **Import from conftest.py**:
   ```python
   import pytest
   # Fixtures from conftest.py are automatically available
   ```

### Common Import Mistakes to Avoid

1. ❌ **Importing directly from src**: `from src.agent_tools.cursor_rules import X`
2. ❌ **Using incorrect relative paths**: Too many or too few dots in relative imports
3. ❌ **Circular imports**: Creating dependency cycles between test modules

Always verify your imports by running a simple test before implementing complex functionality.

## Testing Principles

1. **Examine existing tests first** - Look at similar passing tests before writing new ones
2. **Document-first testing** - Always reference official documentation 
3. **Verify with real-world examples** - Don't rely solely on contrived test data
4. **Ensure proper database isolation** - Tests should clean up after themselves
5. **Use consistent async patterns** - Always use `asyncio.to_thread()` with ArangoDB
6. **Match fixture scopes carefully** - The default `event_loop` fixture is function-scoped
7. **Complete each phase before moving on** - All tests must pass before new features are added

## Running Tests

To run tests for a specific directory:

```bash
python -m pytest src/agent_tools/cursor_rules/tests/unit/
```

To run all tests:

```bash
python -m pytest src/agent_tools/cursor_rules/tests/
```

For more verbose output, add the `-v` flag. 