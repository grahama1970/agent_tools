# Testing Guidelines

## Mocking Best Practices

### When to Mock
1. **Only Mock When Absolutely Necessary**
   - External API calls (e.g., LLM services)
   - Database operations
   - File system operations that can't be handled by tmp_path
   - Network requests

2. **Don't Mock When**
   - Testing actual functionality that can run quickly
   - Working with temporary files (use pytest's tmp_path instead)
   - Testing configuration or CLI structure
   - Validating input/output formats

### How to Mock Effectively
1. **Scope Your Mocks**
   ```python
   # BAD: Global mock fixture used everywhere
   @pytest.fixture
   def mock_service(mocker):
       return mocker.patch("service.call")
   
   # GOOD: Mock only in tests that need it
   def test_error_case(mocker):
       mocker.patch("service.call", side_effect=ValueError)
   ```

2. **Mock at the Right Level**
   ```python
   # BAD: Mocking too many things
   mocker.patch("service.validate")
   mocker.patch("service.process")
   mocker.patch("service.store")
   
   # GOOD: Mock at the highest necessary level
   mocker.patch("service.handle_request")
   ```

3. **Verify Mock Usage**
   ```python
   # BAD: Not verifying mock was used
   mock_service = mocker.patch("service.call")
   run_operation()
   
   # GOOD: Verify mock was called correctly
   mock_service = mocker.patch("service.call")
   run_operation()
   assert mock_service.called
   ```

## Test Categories

### Component Tests
- Focus on single units of functionality
- Mock external dependencies
- Fast execution
- Clear failure messages

### Integration Tests
- Test real functionality where possible
- Minimize mocking
- Use actual services when practical
- Verify end-to-end behavior

## Common Pitfalls

1. **Over-mocking**
   - Problem: Mocking everything "just in case"
   - Solution: Only mock what's necessary for the specific test

2. **Under-testing**
   - Problem: Relying too much on mocks, not testing real behavior
   - Solution: Balance mocked and real tests

3. **Brittle Tests**
   - Problem: Tests break when implementation changes
   - Solution: Test behavior, not implementation details

## Best Practices

1. **Use Descriptive Test Names**
   ```python
   # BAD
   def test_process():
   
   # GOOD
   def test_batch_processing_handles_network_errors():
   ```

2. **One Assert Per Concept**
   ```python
   # BAD
   assert result.status == "success"
   assert result.data == expected
   assert result.errors == []
   
   # GOOD
   def test_successful_processing():
       assert result.status == "success"
   
   def test_processed_data_matches_expected():
       assert result.data == expected
   ```

3. **Clear Test Structure**
   ```python
   def test_feature():
       # Arrange
       input_data = prepare_test_data()
       
       # Act
       result = process_data(input_data)
       
       # Assert
       assert result.is_valid
   ```

## Troubleshooting Guide

1. **Test Failures**
   - Check if mocks are at correct level
   - Verify mock return values match expected types
   - Ensure all necessary dependencies are mocked

2. **Slow Tests**
   - Look for unnecessary mocks
   - Check for redundant setup
   - Consider moving to integration tests

3. **Flaky Tests**
   - Check for timing dependencies
   - Verify mock consistency
   - Look for shared state between tests 