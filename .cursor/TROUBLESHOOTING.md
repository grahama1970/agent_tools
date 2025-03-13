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