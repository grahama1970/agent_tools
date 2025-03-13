# CLI Tests

This directory contains tests for the command-line interface (CLI) components of the Cursor Rules system.

## What Belongs Here

- Tests for CLI commands and arguments
- Tests for CLI output formatting
- Tests for CLI error handling

## Test Files

- **`test_rule_search_cli.py`**: Tests for the rule search CLI commands
- **`test_cli_bm25_search.py`**: Tests for BM25 search CLI functionality

## Import Structure for CLI Tests

CLI tests have specific import patterns to properly test command-line interfaces:

### Example of Correct Imports for CLI Tests

```python
# Example from test_rule_search_cli.py
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

# Import the CLI app/module being tested
from agent_tools.cursor_rules.cli import search_command, app
```

### CLI Test-Specific Import Guidance

1. **Testing Framework Imports**:
   ```python
   from click.testing import CliRunner  # For Click-based CLIs
   # OR
   from typer.testing import CliRunner  # For Typer-based CLIs
   ```

2. **Mocking External Components**:
   ```python
   @patch('agent_tools.cursor_rules.cli.setup_cursor_rules_db')
   @patch('agent_tools.cursor_rules.cli.bm25_keyword_search')
   def test_search_command(mock_search, mock_db_setup):
       mock_db_setup.return_value = MagicMock()
       mock_search.return_value = [{"rule": {"title": "Test"}, "score": 0.95}]
       # Test implementation
   ```

3. **Handling CLI Arguments and Options**:
   ```python
   def test_cli_with_args():
       runner = CliRunner()
       result = runner.invoke(app, ['search', '--query', 'test query'])
       assert result.exit_code == 0
       assert 'Test' in result.output
   ```

## Common Import Issues in CLI Tests

1. **Incorrect CLI Framework Imports**:
   - ❌ Mixing Click and Typer testing utilities
   - ✅ Using the correct testing tools:
     ```python
     # For Click-based CLI:
     from click.testing import CliRunner
     # For Typer-based CLI:
     from typer.testing import CliRunner
     ```

2. **Import Path Inconsistencies**:
   - ❌ Inconsistent import paths between implementation and tests
   - ✅ Consistent imports matching the project structure:
     ```python
     # If the CLI module is at agent_tools.cursor_rules.cli
     from agent_tools.cursor_rules.cli import app
     ```

3. **Failing to Import Mock Dependencies**:
   - ❌ Testing with real dependencies
   - ✅ Properly mocking dependencies:
     ```python
     from unittest.mock import patch, MagicMock
     ```

## Best Practices from LESSONS_LEARNED.md

### CLI Testing Best Practices

1. **Mock external dependencies**
   - Mock external dependencies (e.g., virtual environment paths, analyzers) in CLI tests
   - Use the actual command structure from the implementation in tests
   - Test both success and failure paths with appropriate exit codes
   - When testing CLI output, verify both the exit code and the expected content
   - Keep test mock data realistic but minimal

2. **Maintain Framework Consistency in CLI Development**
   - When a framework choice is made (e.g., typer vs click), stick with it consistently
   - Even if alternatives seem viable, consistency within the project is more important
   - Document framework choices and rationale to prevent accidental mixing

### Documentation-First Testing

1. **Start with documentation**
   - Read the official documentation for the CLI framework before testing it
   - Include documentation links in test files for reference
   - Understand the intended behavior before writing tests

2. **Test in isolation first**
   - Test each CLI command in isolation before integration
   - Understand the behavior of each command individually
   - Avoid interactions between commands that might mask issues

### Real-World Verification

1. **Verify with real-world examples**
   - Test with actual command-line arguments and scenarios
   - Don't rely solely on contrived examples
   - Ensure the CLI works in practice, not just in theory

2. **End-to-end testing**
   - Test the complete CLI workflow from start to finish
   - Verify that all commands work together correctly
   - Ensure the CLI produces the expected results

## Documentation References

- Click: https://click.palletsprojects.com/
- Typer: https://typer.tiangolo.com/
- pytest: https://docs.pytest.org/
- pytest-mock: https://pytest-mock.readthedocs.io/ 