# Testing Lessons Learned

## Learning from Existing Tests

1. **ALWAYS examine existing passing tests before writing new ones**
   - Existing tests provide valuable insights into how the system actually works
   - Passing tests demonstrate the correct patterns for interacting with the system
   - Don't reinvent testing patterns - reuse successful approaches from existing tests

2. **Understand the system through its tests**
   - Existing tests reveal assumptions and constraints that may not be documented elsewhere
   - When adding new functionality, start by understanding how similar functionality is tested
   - Consistency in testing patterns makes the test suite more maintainable

3. **Avoid common pitfalls**
   - Existing tests often contain workarounds for known issues or limitations
   - CRITICAL ERROR: Creating new tests without examining existing passing tests
   - CRITICAL ERROR: Ignoring successful testing patterns established in earlier phases

## Documentation-First Testing

1. **Start with documentation**
   - Read the official documentation for any library or component before testing it
   - Include documentation links in test files for reference
   - Understand the intended behavior before writing tests

2. **Test in isolation first**
   - Test each component in isolation before integration
   - Understand the behavior of each component individually
   - Avoid interactions between components that might mask issues

3. **Progress to integration testing**
   - After components work in isolation, test their integration
   - Verify that components work together as expected
   - Test with real-world data and scenarios

## Real-World Verification

1. **Verify with real-world examples**
   - Test with actual data and scenarios
   - Don't rely solely on contrived examples
   - Ensure the system works in practice, not just in theory

2. **End-to-end testing**
   - Test the complete workflow from start to finish
   - Verify that all components work together correctly
   - Ensure the system produces the expected results

3. **Continuous verification**
   - Regularly run tests to ensure continued functionality
   - Verify that changes don't break existing functionality
   - Maintain a comprehensive test suite

## Lessons from Our Experience

1. **Understand the implementation before testing**
   - Check how functions are implemented before writing tests
   - Look for specific checks like `has_collection()` vs. `collections()`
   - Mock the correct methods based on the actual implementation

2. **Verify mocks are working as expected**
   - Ensure mocked methods are actually being called
   - Use `assert_called_once()` to verify method calls
   - Check that mocked return values are being used

3. **Iterative testing approach**
   - Start with simple tests that verify basic functionality
   - Gradually add more complex tests
   - Fix issues as they arise before moving on

4. **Proper error handling in tests**
   - Understand how errors are handled in the code
   - Test both success and failure paths
   - Ensure error messages are informative 