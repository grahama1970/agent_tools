# Cursor Rules Project Organization Guide

## Directory Structure Best Practices

### Consistent Module Organization

Based on our lessons learned, the following structure is recommended for the `cursor_rules` package:

```
cursor_rules/
├── __init__.py              # Package initialization
├── README.md                # Project documentation
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── cursor_rules.py      # Main implementation
│   └── db.py                # Database connection handling
├── schemas/                 # JSON schemas and data models
│   ├── __init__.py
│   ├── ai_knowledge_schema.json
│   └── db_schema.json
├── cli/                     # Command-line interfaces
│   ├── __init__.py
│   ├── cli.py
│   └── commands/            # CLI subcommands
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── helpers/             # General helpers
│   ├── ai/                  # AI-specific utilities
│   └── text/                # Text processing utilities
├── views/                   # Database view management
│   ├── __init__.py
│   └── view_utils.py
├── scenarios/               # Scenario management
│   ├── __init__.py
│   ├── sample_scenarios.json
│   └── scenario_management.py
├── docs/                    # Documentation
│   ├── retrieval_scenarios.md
│   └── task.md
├── scripts/                 # Utility scripts
│   ├── cleanup_databases.py
│   └── demo.py
└── tests/                   # Test suite
    ├── __init__.py
    ├── conftest.py
    ├── unit/                # Unit tests
    ├── integration/         # Integration tests
    └── end_to_end/          # End-to-end tests
```

## Troubleshooting Common Issues

### Import Errors

If you encounter import errors:

1. Check that your `PYTHONPATH` includes the `src` directory:
   ```
   export PYTHONPATH=/home/grahama/workspace/experiments/agent_tools/src
   ```

2. Verify that your `pyproject.toml` has correct package discovery:
   ```toml
   [tool.hatch.build]
   packages = ["find:src"]

   [tool.hatch.build.targets.wheel]
   packages = ["find:src"]
   ```

3. Ensure that your virtual environment is activated:
   ```
   source .venv/bin/activate
   ```

4. Reinstall the package in development mode:
   ```
   uv pip install -e .
   ```

### Module Not Found Errors

When seeing "No module named 'cursor_rules'" or similar:

1. Check if `__init__.py` exists in all directories in the import path
2. Verify that imports use the correct paths (e.g., `from agent_tools.cursor_rules.core import ...`)
3. Run commands with the `-m` flag from the project root:
   ```
   python -m agent_tools.cursor_rules.cli
   ```

### Environment Configuration

Add to your `.env` file:
```
PYTHONPATH=/home/grahama/workspace/experiments/agent_tools/src
```

## File Organization Recommendations

1. **Avoid Duplication**: Don't keep the same files in multiple locations
2. **Follow Consistent Paths**: Use absolute imports with the full package path
3. **Separate Concerns**: Keep database, CLI, and business logic separated
4. **Maintain Test Structure**: Ensure tests mirror the structure of the code they test 

## Test Debugging Techniques

### Common Test Failures and Solutions

1. **Missing Functions / Methods**
   - **Symptom**: `ImportError: cannot import name '_function_name' from 'package'`
   - **Cause**: Function was deleted or renamed during development
   - **Solution**: Check git history to find the original function definition, or recreate the function by analyzing its usage in tests
   - **Prevention**: Use the `grep_search` tool to find all usages before renaming or deleting functions

2. **Parameter Signature Mismatches**
   - **Symptom**: `TypeError: function() missing 1 required positional argument: 'param'`
   - **Cause**: Tests call functions with incorrect parameters or wrong order
   - **Solution**: Always check the actual function signature in the implementation file:
     ```python
     # Find the signature
     grep -r "def function_name" src/
     
     # Read the first ~10 lines of the function definition
     head -n 10 src/path/to/file.py
     ```
   - **Prevention**: Add docstrings with complete parameter definitions to all functions

3. **String vs Path Objects**
   - **Symptom**: `TypeError: unsupported operand type(s) for /: 'str' and 'str'`
   - **Cause**: Passing string paths where `pathlib.Path` objects are expected
   - **Solution**: Wrap string paths with `Path()` objects before passing to functions
   - **Prevention**: Use type hints in function definitions: `def func(path: Path, ...)`

4. **Pytest Collection Issues**
   - **Symptom**: `ERROR: found no collectors for path/to/test.py::test_function`
   - **Cause**: Python path issues or incorrectly structured tests
   - **Solution**:
     - Add `conftest.py` to set Python path correctly
     - Run with explicit PYTHONPATH: `PYTHONPATH=$PWD/src pytest ...`
     - Verify test function names match the pattern `test_*`
   - **Prevention**: Always run simpler tests first (e.g., `test_file.py` before `test_file.py::test_function`)

5. **Module Skipping Issues**
   - **Symptom**: `collected 0 items / 1 skipped`
   - **Cause**: Module-level `pytest.skip()` with `allow_module_level=True`
   - **Solution**: 
     - Replace module-level skip with conditional skip in individual tests
     - Print detailed error messages before skipping
     - Use a direct test script for simple functionality checks
   - **Prevention**: Include clear error messages in skip conditions

### Effective Test Troubleshooting Strategy

1. **Direct Verification Script**
   - Create a simple standalone script that tests just one function
   - Example: `test_one_function.py` with explicit imports
   - Run outside the test framework: `python test_one_function.py`
   - Verify basic functionality first before running complex tests

2. **Python Path Debugging**
   - Print the Python path at the beginning of tests:
     ```python
     import sys
     print(f"Python path: {sys.path}")
     ```
   - Verify that source directories are included
   - Add source directories explicitly if needed:
     ```python
     sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
     ```

3. **Incremental Test Running**
   - Start with the simplest test that should work
   - Run one test at a time to isolate issues
   - Add complexity only after basics are working
   - Example progression:
     ```bash
     # 1. Run just one simple test function
     pytest tests/package/test_simple.py::test_basic_function -v
     
     # 2. Run all tests in one file
     pytest tests/package/test_simple.py -v
     
     # 3. Run all tests in a module
     pytest tests/package/ -v
     
     # 4. Run all tests
     pytest
     ```

4. **Error Message Analysis**
   - Pay close attention to the full traceback
   - Look for the root cause, not just the final error
   - Check for import errors, parameter mismatches, and type errors
   - Use print statements to debug variable values

5. **Test Structure Review**
   - Ensure test files have proper module structure
   - Verify presence of `__init__.py` files in test directories
   - Check test function naming (`test_*`)
   - Validate fixture scope and usage 