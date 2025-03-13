# Troubleshooting Guides

This directory contains guides to help troubleshoot common issues with the fetch-page project.

## Available Guides

- [Pytest Debugging Guide](pytest-debugging.md) - How to debug pytest issues
- [Test Categories and Organization](test-categories.md) - Understanding test organization
- [Environment Setup](environment-setup.md) - Setting up the environment for testing

## When to Use These Guides

- When tests are not running as expected
- When you encounter dependency issues
- When you need to add new tests
- When you need to set up a new environment

## Key Principles

1. **Keep test configuration simple**
   - Avoid complex test ordering logic
   - Use pytest-order plugin for explicit ordering
   - Document test dependencies clearly

2. **Improve error visibility**
   - Add explicit print statements in tests
   - Use `pytest.fail()` with descriptive messages
   - Log test setup and teardown actions

3. **Isolate test environments**
   - Use fixtures with proper scope
   - Clean up resources after tests
   - Avoid global state changes

4. **Document dependencies**
   - Update `pyproject.toml` with all required dependencies
   - Document environment variables in `.env.example`
   - Keep track of model dependencies

## Quick Reference

```bash
# Run all tests
pytest

# Run with maximum verbosity and debug logging
pytest -vvs --log-cli-level=DEBUG

# Run a specific test
pytest tests/test_file.py::test_name

# Install missing dependencies
uv pip install package_name
``` 