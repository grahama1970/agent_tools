# ArangoDB Best Practices

## Async Operations
1. `await asyncio.to_thread(db_function, arg1, arg2)` - Always wrap DB operations in async code
2. NEVER use lambda functions with `asyncio.to_thread` - this can cause issues with thread safety
3. Break down complex operations into separate steps with individual `await asyncio.to_thread` calls
4. If NOT within an async function, use ArangoDB operations normally without `asyncio.to_thread`

## AQL Queries
1. `SEARCH ANALYZER(doc.field IN TOKENS(@query, "text_en"), "text_en")` - Use for text search
2. NEVER use `LIKE` for text search - use `TOKENS` with appropriate analyzers instead
3. Use `STARTS_WITH(doc._id, 'collection/')` instead of `doc._id LIKE 'collection/%'` for prefix matching
4. `SORT BM25(doc) DESC` - Always use BM25 for relevance ranking in search results

## Error Handling
1. Always check for None database connections before performing operations
2. Use try/except blocks to handle database connection errors gracefully
3. In tests, ensure proper cleanup of database resources even if tests fail

## Testing
1. Use module-scoped event_loop fixtures when working with module-scoped database fixtures
2. Add `@pytest.mark.asyncio` to async test functions
3. Mock device-specific operations (like CUDA) in tests to ensure they run in all environments

## Performance
1. Use batch operations when possible for better performance
2. Create appropriate indexes for frequently queried fields
3. Use ArangoSearch views with proper analyzers for text search operations
