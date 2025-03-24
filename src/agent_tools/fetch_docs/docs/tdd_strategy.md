# Test-Driven Development Strategy for fetch_docs

## TDD Philosophy

The development of the fetch_docs module follows a test-driven development approach with a focus on "blind tests" that validate real-world functionality rather than implementation details. This approach ensures that:

1. Requirements are clearly understood before implementation begins
2. Code is naturally modular and testable
3. Regressions are caught early
4. Integration with other modules is verified

## Testing Strategy

### 1. Blind Tests

We use "blind tests" as our primary validation approach. This means:

- Tests are written to check what the code *should* do, not how it's implemented
- Tests use real-world documentation sources rather than contrived examples
- Expected outputs are explicitly defined before implementation

### 2. Test Hierarchy

Tests are organized into three levels:

1. **Unit Tests**: Validate individual functions (e.g., HTML cleaning, section extraction)
2. **Integration Tests**: Verify that components work together (e.g., download + processing)
3. **End-to-End Tests**: Validate the entire pipeline using real documentation sites

### 3. Real-World Validation

For each major component, we:

1. Select a representative documentation site (e.g., ReadTheDocs, ArangoDB docs)
2. Define expected outputs in detail
3. Implement tests that verify the actual output matches expectations
4. Use these tests to guide implementation

## Test Implementation

### Download Tests

- Verify that the download_site function creates the expected directory structure
- Check that correct files are downloaded
- Validate handling of errors (network issues, 404s, etc.)

### HTML Processing Tests

- Compare cleaned HTML against known, expected output
- Verify removal of unwanted elements (scripts, styles, etc.)
- Test handling of malformed HTML

### Section Extraction Tests

- Validate header detection and section creation
- Test hierarchy construction (parent-child relationships)
- Verify special element detection (code blocks, tables, images)

### Metadata Tests

- Verify token counting functionality
- Test hierarchy metadata correctness
- Validate special element metadata

### Pipeline Tests

- Test end-to-end processing from download to JSON output
- Verify output format matches specifications
- Test with multiple documentation sources (ReadTheDocs, ArangoDB, etc.)

### DuaLipa Integration Tests

- Verify link detection in repositories
- Test combined extraction (code + docs)
- Validate format compatibility

## Test Data Management

To avoid repeated downloads during testing while ensuring real-world validation:

1. Use cached copies of downloaded sites for most tests
2. Periodically refresh test data to ensure compatibility with current sites
3. Include sample data in the repository for quick testing

## TDD Workflow

For each feature or component:

1. Write a failing test that defines expected behavior
2. Implement the minimum code needed to pass the test
3. Refactor for cleanliness and performance
4. Verify that tests still pass
5. Document the implementation

## Critical Test Cases

1. **HTML Cleaning**: Verify that content is preserved while unwanted elements are removed
2. **Section Extraction**: Test proper identification of headers and content sections
3. **Hierarchy**: Validate that parent-child relationships are correctly established
4. **Special Elements**: Test detection of code blocks, tables, and images
5. **Integration**: Verify proper integration with DuaLipa extraction

## Code Standards

For all implementation and tests:
- Maximum 500 lines of code per file
- Include descriptions and third-party documentation links at the top of every file
- Include sample inputs and expected outputs in documentation
- Follow type hints and docstring conventions
- Use uv instead of pip for package management (as specified in pyproject.toml)