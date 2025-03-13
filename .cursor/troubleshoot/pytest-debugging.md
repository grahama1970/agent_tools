# Pytest Debugging Guide

## Common Issues and Solutions

### Tests Not Running (Collected but Not Executed)

1. **Check for filtering in `conftest.py`**:
   - Look for `pytest_collection_modifyitems` function that might be filtering tests
   - Ensure all tests are included in the final `items` list
   - Prefer using pytest-order plugin over manual ordering

2. **Increase verbosity and logging**:
   - Run with `-vvs --log-cli-level=DEBUG` flags
   - Check for warnings about configuration issues
   - Use `--trace` or `--debug` for more detailed information

3. **Test in isolation**:
   - Move test file to a different directory
   - Create a minimal test file with no dependencies
   - Run with `--no-header --no-summary` to see raw output

4. **Check for plugin conflicts**:
   - Try disabling plugins with `-p no:plugin_name`
   - Check for version compatibility issues
   - Look for warnings about deprecated features

### Dependency Issues

1. **Missing packages**:
   - Check error messages for missing imports
   - Install dependencies with `uv pip install package_name`
   - Update `pyproject.toml` to include all required dependencies

2. **Environment variables**:
   - Ensure all required environment variables are set
   - Check for `HF_HUB_ENABLE_HF_TRANSFER` and other config variables
   - Use `.env` file for local development

## Best Practices

1. **Keep test configuration simple**:
   - Avoid complex test ordering logic
   - Use pytest-order plugin for explicit ordering
   - Document test dependencies clearly

2. **Improve error visibility**:
   - Add explicit print statements in tests
   - Use `pytest.fail()` with descriptive messages
   - Log test setup and teardown actions

3. **Isolate test environments**:
   - Use fixtures with proper scope
   - Clean up resources after tests
   - Avoid global state changes

## Quick Reference Commands

```bash
# Run with maximum verbosity and debug logging
pytest -vvs --log-cli-level=DEBUG

# Run a specific test with debug output
pytest tests/test_file.py::test_name -vvs

# Disable specific plugins
pytest -p no:asyncio -p no:xdist

# Run with trace output
pytest --trace

# Debug test collection
pytest --collect-only -v

# Run tests in a different directory
mkdir -p /tmp/test_dir && cp test_file.py /tmp/test_dir/ && cd /tmp/test_dir && pytest

# Check for missing dependencies
pip list | grep package_name
``` 