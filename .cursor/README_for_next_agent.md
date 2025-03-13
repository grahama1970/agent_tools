# Cursor Rules Project Fix - Next Steps

## Current Situation
We've been working on fixing test failures in the `agent_tools/cursor_rules` project after some refactoring. The main issue we're facing is with the `update_scenario` function in `scenario_management.py` which fails its test.

## Code Structure
- `agent_tools/cursor_rules/scenario_management.py`: Manages scenario data in ArangoDB
- `agent_tools/cursor_rules/core/cursor_rules.py`: Core functionality for cursor rules
- `agent_tools/cursor_rules/tests/test_scenario_management.py`: Tests for scenario management

## Key Findings
1. The `update_scenario` function in `scenario_management.py` fails its test
2. The test expects the updated document to have the new title, but it appears the update isn't being applied
3. The ArangoDB Python driver requires specific patterns for async operations and document updates

## Learning Issues Encountered
1. **ArangoDB Async Pattern**: I kept using lambda functions incorrectly when the codebase follows a clear pattern of defining inner functions and passing them to `asyncio.to_thread`
2. **Document Update Mechanics**: The ArangoDB Python driver requires specific ways to update documents that I'm struggling with

## Working Pattern
The working pattern for ArangoDB operations in this codebase is:

```python
async def some_function(db, data):
    await ensure_collection(db)
    
    def db_operation():
        collection = db.collection(COLLECTION_NAME)
        # Perform operation
        return collection.some_operation(data)
    
    result = await asyncio.to_thread(db_operation)
    return result
```

## What Needs to Be Fixed
1. Fix the `update_scenario` function in `scenario_management.py` to correctly update documents in ArangoDB
2. The key is likely in understanding how the Python ArangoDB driver's `update` method works
3. The issue might be related to the `merge` parameter or how we're constructing the update document

## Most Recent Error
```
AssertionError: assert 'Common Code Pattern Query' == 'Updated Pattern Query'
```

The test expects that after updating, the document will have the new title, but it's still showing the old title.

## Next Steps
1. Investigate the ArangoDB Python driver documentation for the correct update pattern
2. Study the working functions in the same file (like `store_scenario`)
3. Try different approaches to document updates (using merge, replace, etc.)
4. Run tests after each change to verify progress

This fix should be straightforward once we understand the correct pattern for updating documents in ArangoDB using the Python driver. 