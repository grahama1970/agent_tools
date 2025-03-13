# AI Agent Instructions: Fetch Page Examples

This directory contains reference implementations that you, as an AI agent, should use when assisting with the Fetch Page project. These examples demonstrate the preferred patterns and practices you should follow when generating or modifying code.

## How to Use These Examples

As an AI agent, you should:

1. **Reference these examples** when generating code for similar functionality
2. **Follow the patterns** demonstrated in these examples
3. **Maintain consistency** with the established code style and error handling approaches
4. **Prioritize real connections over mocks** in test code when appropriate

## Key Principles to Follow

### 1. Minimize Mocking in Tests

When generating test code:
- Only mock external APIs and computationally expensive operations
- Use real database connections when possible
- Mock at the appropriate boundaries (e.g., HTTP requests, not internal functions)

Example pattern to follow:
```python
# GOOD: Only mock the embedding computation, use real DB connection
with patch("module.create_embedding_for_page", return_value=sample_embedding):
    # Use actual database connection
    key = await store_document_with_embedding(document, db)
    
    # Verify with real database query
    retrieved = await retrieve_document(key, db)
    assert retrieved["embedding"] == sample_embedding
```

### 2. Use Proper Async Patterns

When generating async code:
- Always use `asyncio.to_thread` for CPU-bound operations
- Implement proper error handling with try/except blocks
- Use timeouts to prevent hanging
- Handle cancellation appropriately
- Clean up resources in finally blocks

Example pattern to follow:
```python
async def process_data(data: List[int]) -> Dict[str, Any]:
    try:
        # Use to_thread for CPU-bound operations
        result = await asyncio.to_thread(cpu_intensive_calculation, data)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise
```

### 3. Implement Comprehensive Error Handling

When generating code with error handling:
- Use specific exception types when possible
- Log errors with appropriate context
- Implement retries with exponential backoff for transient errors
- Provide meaningful error messages

### 4. Include Documentation References

When generating code, include:
- References to official documentation
- References to related rules in `.cursor/rules/`
- Comprehensive docstrings
- Complete type hints

## Directory Structure Reference

Use these examples as reference implementations for specific tasks:

### 01-async
- `to_thread_example.py`: Reference for CPU-bound operations with `asyncio.to_thread`
- `error_handling.py`: Reference for error handling, timeouts, and retries

### 02-embedding
- `nomic_embed_example.py`: Reference for generating embeddings
- `embedding_storage.py`: Reference for storing and retrieving embeddings

### 03-llm
- `litellm_basic.py`: Reference for LLM integration with caching and streaming

### 04-testing
- `embedding_test_example.py`: Reference for testing with minimal mocking

### 05-db
- `arangodb_basic.py`: Reference for database operations

## Related Rules

When generating code, also reference these rules:

- `003-package-usage.mdc` - For package usage patterns
- `004-testing-practices.mdc` - For testing practices
- `005-async-patterns.mdc` - For async programming patterns
- `011-embedding-practices.mdc` - For embedding practices 