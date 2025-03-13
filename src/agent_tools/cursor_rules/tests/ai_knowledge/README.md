# AI Knowledge Tests

This directory contains tests specific to the AI knowledge integration functionality of the Cursor Rules system.

## What Belongs Here

- Tests for AI knowledge database operations
- Tests for method validation with AI knowledge
- Tests for AI knowledge integration with other components

## Test Files

- **`test_ai_knowledge_basic.py`**: Basic tests for AI knowledge functionality
- **`test_ai_knowledge_integration.py`**: Tests for integrating AI knowledge with other components
- **`test_ai_knowledge_method_validation.py`**: Tests for validating methods with AI knowledge

## Import Structure for AI Knowledge Tests

AI Knowledge tests deal with multiple external dependencies and need to handle imports carefully:

### Example of Correct Imports for AI Knowledge Tests

```python
# Example from test_ai_knowledge_integration.py
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch, MagicMock
import os
import json

# Import main AI knowledge functionality
from agent_tools.cursor_rules.ai_knowledge import (
    register_method_knowledge,
    update_method_knowledge,
    get_method_knowledge
)

# Import database utilities
from agent_tools.cursor_rules.utils.db import setup_database
```

### AI Knowledge Test-Specific Import Guidance

1. **Conditional Imports for Optional Dependencies**:
   ```python
   try:
       import openai
       from litellm import completion
       HAS_LITELLM = True
   except ImportError:
       HAS_LITELLM = False
       
   @pytest.mark.skipif(not HAS_LITELLM, reason="LiteLLM not installed")
   def test_ai_integration():
       # Test implementation
   ```

2. **Mock LLM and API Dependencies**:
   ```python
   @patch('agent_tools.cursor_rules.ai_knowledge.completion')
   def test_llm_interaction(mock_completion):
       mock_completion.return_value = {
           "choices": [{"message": {"content": "Test response"}}]
       }
       # Test implementation
   ```

3. **Database Connection Setup**:
   ```python
   @pytest_asyncio.fixture(scope="function")
   async def ai_knowledge_db():
       db = await asyncio.to_thread(setup_database, "test_ai_knowledge")
       yield db
       # Clean up test data
       await asyncio.to_thread(lambda: db.drop() if db.has_database("test_ai_knowledge") else None)
   ```

## Common Import Issues in AI Knowledge Tests

1. **Missing LLM/API Dependencies**:
   - ❌ Not handling potential import errors for optional dependencies
   - ✅ Using try/except with skip markers:
     ```python
     try:
         import transformers
         HAS_TRANSFORMERS = True
     except ImportError:
         HAS_TRANSFORMERS = False
         
     @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="Transformers not installed")
     def test_embedding_generation():
         # Test implementation
     ```

2. **Database Access Pattern Issues**:
   - ❌ Accessing database synchronously in async code
   - ✅ Using proper async wrappers:
     ```python
     # Correct async database access
     async def test_knowledge_retrieval(ai_knowledge_db):
         result = await asyncio.to_thread(
             lambda: ai_knowledge_db.collection('ai_knowledge').get('method_id')
         )
     ```

3. **Incorrect Path to AI Knowledge Modules**:
   - ❌ Using inconsistent import paths
   - ✅ Following the src-based layout structure:
     ```python
     # If ai_knowledge is a submodule of cursor_rules
     from agent_tools.cursor_rules.ai_knowledge import method_validator
     # If it's a utility
     from agent_tools.cursor_rules.utils.ai_knowledge import method_validator
     ```

## Best Practices from TESTING_LESSONS.md and LESSONS_LEARNED.md

### Documentation-First Testing

1. **Start with documentation**
   - Read the official documentation for any library or component before testing it
   - Include documentation links in test files for reference
   - Understand the intended behavior before writing tests

2. **Test in isolation first**
   - Test each component in isolation before integration
   - Understand the behavior of each component individually
   - Avoid interactions between components that might mask issues

### Method Validation

1. **Using method_validator to Verify Methods**
   - The `method_validator` tool can verify that methods actually exist in packages
   - It can check method signatures to ensure parameters are correct
   - Use it when uncertain about whether a method exists or its signature
   - Particularly useful for less common packages or recent updates

2. **When to Use Method Validation**
   - When implementing functionality with unfamiliar libraries
   - When updating code to use newer versions of packages
   - When troubleshooting unexpected behavior
   - When working with packages that have similar names but different interfaces
   - Before spending time debugging code that might use non-existent methods

### Database Integration

1. **ArangoDB Driver is Synchronous Only**
   - The `python-arango` driver is primarily a synchronous library
   - It does not provide native asynchronous API methods
   - All database operations will block the event loop if called directly in async code
   - Any async wrapper must handle this blocking nature appropriately

2. **Integrating ArangoDB with Async Functions**
   - Always use `asyncio.to_thread()` when calling ArangoDB operations from async code
   - This runs the synchronous operation in a separate thread, preventing blocking
   - **CRITICAL: You MUST await the result of asyncio.to_thread() - it returns a coroutine**

3. **Testing Async Database Operations with pytest-asyncio**
   - **CRITICAL**: Match fixture scopes carefully when using pytest-asyncio
   - The default `event_loop` fixture is function-scoped
   - Database fixtures should match the event_loop scope

## Documentation References

- python-arango: https://python-arango.readthedocs.io/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
- method_validator: Internal tool for validating method existence 