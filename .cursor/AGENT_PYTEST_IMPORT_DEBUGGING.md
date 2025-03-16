# Python Import Path Conflicts: How to Diagnose and Fix

## The Problem: Conflicting Package Names

When you have a Python project with the same name as the directory containing it, import path conflicts can occur. This happens when Python finds multiple packages with the same name in different locations on the `sys.path`.

In our case, we had:
- A project directory named `agent_tools`
- An `__init__.py` file in the project root, making it a package
- A proper package in `src/agent_tools/` with submodules

Python was importing from the wrong location, causing `ModuleNotFoundError` for submodules.

## How to Identify the Issue

When you see this error pattern:
```
ModuleNotFoundError: No module named 'package.submodule'
```

But you're certain the submodule exists, check where Python is finding the main package:

```python
import package
print(f"Package found at: {package.__file__}")
```

If the path points to an unexpected location (like the project root instead of `src/`), you have an import conflict.

## Key Diagnostic Steps

1. **Check the import path**: Add debugging code to see where Python finds the package:
   ```python
   try:
       import agent_tools
       print(f"agent_tools found at: {agent_tools.__file__}")
   except ImportError:
       print("Could not import agent_tools at all")
   ```

2. **Examine sys.path**: Print the Python path to see all locations being searched:
   ```python
   import sys
   print(f"Python path: {sys.path}")
   ```

3. **Look for duplicate package names**: Check if the package name appears in multiple locations in `sys.path`

## Common Fixes

1. **Remove `__init__.py` from project root**:
   ```bash
   rm /path/to/project_root/__init__.py
   ```

2. **Clear Python's import cache**:
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

3. **Use a unique name for your top-level package**

4. **Install your package in development mode**:
   ```bash
   pip install -e .
   ```

5. **Use absolute imports** in your code:
   ```python
   # Instead of: from . import submodule
   from package.submodule import function
   ```

## Prevention Strategies

1. **Never create `__init__.py` files in your project root**
2. **Keep a clean separation between your package and project directories**
3. **Use a proper package structure with `src/` layout**
4. **Configure `pyproject.toml` or `setup.py` correctly**
5. **Use virtual environments to isolate dependencies**

## Quick Checklist for Fixing Import Errors

- [ ] Check where the package is being imported from (`package.__file__`)
- [ ] Remove any `__init__.py` files from the project root
- [ ] Clear Python cache directories (`__pycache__`)
- [ ] Verify your `PYTHONPATH` environment variable
- [ ] Check for duplicate package names in different directories
- [ ] Reinstall the package in development mode (`pip install -e .`)

---
Answer from Perplexity: pplx.ai/share