# Integration Tests

This directory contains tests that verify the interaction between multiple components of the Cursor Rules system.

## What Belongs Here

- Tests that connect multiple components together
- Tests that interact with a real database
- Tests that verify proper data flow between components

## Test Files

- **`test_arango_async_patterns.py`**: Tests for asynchronous database operations
- **`test_arango_base.py`**: Tests for basic ArangoDB functionality
- **`test_bm25_search.py`**: Tests for BM25 text search with database
- **`test_embedding_utils.py`**: Tests for embedding utilities with database
- **`test_hybrid_search.py`**: Tests for hybrid search functionality
- **`test_semantic_search.py`**: Tests for semantic search with embeddings
- **`test_vector_functions.py`**: Tests for vector operations with database

## Import Structure for Integration Tests

Integration tests must carefully handle imports to ensure components are correctly connected:

### Example of Correct Imports for Integration Tests

```python
# Example from test_semantic_search.py
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch, MagicMock

# Importing directly from the package (preferred for integration tests)
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding, semantic_search
from agent_tools.cursor_rules.embedding import create_embedding_sync, ensure_text_has_prefix

# OR using relative imports
# from ...cursor_rules.core.cursor_rules import generate_embedding, semantic_search
```

### Integration Test-Specific Import Guidance

1. **Database and External Service Imports**:
   ```python
   import pytest
   import pytest_asyncio
   from python_arango import ArangoClient
   ```

2. **Async Support**:
   ```python
   @pytest_asyncio.fixture(scope="function")
   async def db_connection():
       # Create database connection for test
       yield connection
       # Clean up after test
   ```

3. **Handle dependency errors properly**:
   ```python
   try:
       from agent_tools.cursor_rules.embedding import create_embedding_sync
   except (ImportError, ModuleNotFoundError):
       pytest.skip("Embedding utilities not available")
   ```

## Common Import Issues in Integration Tests

1. **Incorrect Relative Paths**: 
   - ❌ `from ..cursor_rules.utils import X`
   - ✅ `from ...cursor_rules.utils import X` or `from agent_tools.cursor_rules.utils import X`

2. **Missing Async Imports**:
   - ❌ Missing `import asyncio` or `import pytest_asyncio`
   - ✅ Include proper async support: `import asyncio, pytest_asyncio`

3. **Invalid Database Import Patterns**:
   - ❌ Directly importing without error handling
   - ✅ Using proper error handling and async wrappers:
     ```python
     async def db_operation(db, data):
         return await asyncio.to_thread(lambda: db.collection('docs').insert(data))
     ```

## Best Practices

1. **Clean up after tests** - Remove test data after each test
2. **Isolate test environments** - Use unique collection names for tests
3. **Use correct async patterns** - Always use `asyncio.to_thread()` with ArangoDB
4. **Match fixture scopes** - Ensure event loop and database fixtures have matching scopes
5. **Document dependencies** - Include setup requirements in test docstrings

## Documentation References

- python-arango: https://python-arango.readthedocs.io/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
- ArangoDB: https://www.arangodb.com/docs/stable/ 