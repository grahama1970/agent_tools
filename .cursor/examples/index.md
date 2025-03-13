# AI Agent Reference: Fetch Page Examples Index

This index provides quick references to example implementations you should follow when generating code for the Fetch Page project.

## Async Programming Examples

| Example | When to Use This Reference |
|---------|-------------|
| [to_thread_example.py](01-async/to_thread_example.py) | When implementing CPU-bound operations that need to be run asynchronously |
| [error_handling.py](01-async/error_handling.py) | When implementing error handling, timeouts, and retries in async functions |

## Embedding Examples

| Example | When to Use This Reference |
|---------|-------------|
| [nomic_embed_example.py](02-embedding/nomic_embed_example.py) | When implementing code that generates embeddings with Nomic Embed |
| [embedding_storage.py](02-embedding/embedding_storage.py) | When implementing code that stores or retrieves embeddings in ArangoDB |

## LLM Integration Examples

| Example | When to Use This Reference |
|---------|-------------|
| [litellm_basic.py](03-llm/litellm_basic.py) | When implementing LLM completions with caching and streaming |

## Testing Examples

| Example | When to Use This Reference |
|---------|-------------|
| [embedding_test_example.py](04-testing/embedding_test_example.py) | When writing tests for embedding functionality with minimal mocking |

## Database Operation Examples

| Example | When to Use This Reference |
|---------|-------------|
| [arangodb_basic.py](05-db/arangodb_basic.py) | When implementing ArangoDB operations using async patterns |

## Key Patterns to Follow

When generating code, always adhere to these patterns:

1. **Minimize Mocking**: Only mock external APIs and computationally expensive operations
2. **Use Real Connections**: Implement tests with actual database connections when possible
3. **Implement Proper Error Handling**: Use specific exception types and comprehensive logging
4. **Follow Async Patterns**: Use `asyncio.to_thread` for CPU-bound operations
5. **Include Documentation**: Add references to official docs and related rules

Refer to the [README.md](README.md) for more detailed instructions on how to use these examples when generating code. 