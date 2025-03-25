# Test-Driven Development Strategy for Fetch Docs Integration

This document outlines the TDD strategy for integrating fetch_docs capabilities into the DuaLipa extraction pipeline. The goal is to enhance the extraction with external documentation, enabling more comprehensive context for QA.

## Collaborative Test Design First

1. **Collaborate on Sample Inputs and Expected Outputs**:
   - BEFORE writing any tests, define sample inputs collaboratively
   - Document expected outputs in precise JSON format
   - Create example extraction scenarios with edge cases
   - Establish validation criteria for successful extraction

2. **Test Link Detection**: 
   - Create test repositories with various documentation links
   - Test detection of different doc site patterns (ReadTheDocs, ArangoDB, etc.)
   - Pre-define expected link patterns and test against them

3. **Test Documentation Download**:
   - Mock download responses for testing
   - Handle various HTTP errors gracefully
   - Test with real documentation sites
   - Define expected structure of downloaded content

4. **Test HTML Processing**:
   - Verify HTML cleaning and section extraction 
   - Test hierarchy preservation
   - Test handling of special elements (tables, code blocks, images)
   - Create benchmarks for expected output structure

5. **Test Conversion to DuaLipa Format**:
   - Validate format compatibility with extraction blocks
   - Test parent-child relationship creation
   - Verify metadata is properly attached
   - Compare against pre-defined expected output objects

6. **Test Integration with Extraction Pipeline**:
   - Test graceful fallback if fetch_docs is unavailable
   - Test combined extraction results (code + docs)
   - Verify QA compatibility of final output
   - Benchmark against sample QA interactions

## Testing Tools

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the full extraction pipeline
- **Blind Tests**: Test with previously unseen repositories
- **Manual Validation**: Visual inspection of extraction results
- **Spot Checks**: Manual verification of test results AFTER automated tests pass
- **Comparative Analysis**: Compare extraction results with human-annotated ground truth
- **Cross-Validation**: Verify extraction works across different documentation systems

## Test Data

- **Mock Repositories**: Custom repositories with ReadTheDocs/ArangoDB links
- **Real-World Examples**: Public repositories with documentation links
- **Sample Documentation**: Pre-downloaded documentation for reproducible testing

## Success Criteria

1. All documentation links are correctly detected
2. Documentation is downloaded and processed without errors
3. Hierarchical structure is preserved in the extraction
4. Special elements (tables, code, images) are properly extracted
5. Output format is compatible with DuaLipa QA system
6. Integration gracefully handles failures
7. Extraction format meets all requirements for frictionless validation
8. Manual spot checks confirm automated test results
9. Extraction achieves the same results with both Playwright and wget approaches (where applicable)
10. Human review confirms extracted content is usable for downstream QA tasks

## Implementation Path

1. **Collaborative Design Phase**:
   - Define sample inputs and expected outputs BEFORE implementation
   - Create JSON schema for validation
   - Establish frictionless validation patterns and examples
   - Document parent-child relationship requirements

2. **Implementation Phase**:
   - Start with ReadTheDocs support (most common)
   - Add ArangoDB documentation support
   - Implement Playwright support for JavaScript-rendered sites
   - Extend to other documentation formats as needed
   - Continuously improve error handling and robustness

3. **Verification Phase**:
   - Run automated tests to verify functionality
   - Perform manual spot checks AFTER tests pass
   - Compare extraction with expected output examples
   - Validate with human review of extraction quality
   - Test with real-world documentation sources