# Test-Driven Development Strategy for Extraction Modules

This document outlines the revised TDD strategy for DuaLipa extraction modules, accounting for the specific limitations of AI assistants in maintaining context and systematically verifying assumptions.

## Fundamental Principles for AI-Compatible TDD

1. **Explicit Context Verification**:
   - Each test MUST begin with explicit verification of the environment
   - Never assume context is maintained between test steps
   - Required verification script runs BEFORE any extraction logic

2. **Chain of Evidence Pattern**:
   - Every step must produce verifiable output artifacts
   - Each subsequent step must validate the outputs of previous steps
   - Example: `assert repository_stats["python_files"] == 70, "Expected 70 Python files"`

3. **State-Preserving Test Structure**:
   - Pass complete state explicitly between test phases
   - Avoid relying on assistant's memory of previous outputs
   - Structure tests as data pipelines with verification checkpoints

## Repository Analysis First

**ALWAYS start with repository analysis.**

```python
def test_extraction_pipeline():
    # Step 1: MANDATORY repository analysis
    repo_stats = analyze_repository(REPO_PATH)
    
    # Step 2: Verify expected files exist
    assert repo_stats["file_counts"][".py"] >= 70, f"Expected at least 70 Python files, found {repo_stats['file_counts'].get('.py', 0)}"
    assert "tests/js/common/test-data/search/docs/generate_ii_sa_dataset.py" in repo_stats["important_files"], "Missing critical test file"
    
    # Step 3: Plan extraction based on verified counts
    extraction_plan = plan_extraction(repo_stats)
    
    # Step 4: Extract with continuous verification
    extraction_results = run_extraction_with_verification(extraction_plan)
    
    # Step 5: Validate extraction completeness
    verify_extraction_coverage(extraction_results, repo_stats)
```

## Collaborative Test Design First

1. **Collaborate on Sample Inputs and Expected Outputs**:
   - BEFORE writing any tests, define sample inputs collaboratively
   - Document expected outputs in precise JSON format
   - Create example extraction scenarios with edge cases
   - Establish validation criteria for successful extraction
   - **NEW: Create repository analysis baseline for verification**

2. **Test Link Detection**: 
   - Create test repositories with various documentation links
   - Test detection of different doc site patterns (ReadTheDocs, ArangoDB, etc.)
   - Pre-define expected link patterns and test against them
   - **NEW: Verify EXACT number of links detected matches analysis**

3. **Test Documentation Download**:
   - Mock download responses for testing
   - Handle various HTTP errors gracefully
   - Test with real documentation sites
   - Define expected structure of downloaded content
   - **NEW: Verify download completeness against pre-counted expectations**

4. **Test HTML Processing**:
   - Verify HTML cleaning and section extraction 
   - Test hierarchy preservation
   - Test handling of special elements (tables, code blocks, images)
   - Create benchmarks for expected output structure
   - **NEW: Validate section counts match pre-analysis expectations**

5. **Test Conversion to DuaLipa Format**:
   - Validate format compatibility with extraction blocks
   - Test parent-child relationship creation
   - Verify metadata is properly attached
   - Compare against pre-defined expected output objects
   - **NEW: Include explicit counts in validation**

6. **Test Integration with Extraction Pipeline**:
   - Test graceful fallback if fetch_docs is unavailable
   - Test combined extraction results (code + docs)
   - Verify QA compatibility of final output
   - Benchmark against sample QA interactions
   - **NEW: Verify ALL expected files are included in final output**

## Testing Tools and Methods

### New Required Tools

1. **Repository Analysis Tool**: 
   - Count files by type BEFORE extraction
   - Generate baseline statistics
   - Create list of important files to verify
   - Save analysis as artifact for later verification

2. **Extraction Verification Tool**:
   - Compare extraction results to baseline
   - Flag missing files or discrepancies
   - Generate coverage metrics
   - Fail test if coverage is below threshold

3. **Checkpoint Validation**:
   - Explicit validation at each processing step
   - Save intermediate results as artifacts
   - Verify consistency at each stage

### Traditional Tools

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
- **NEW: Repository Statistics**: Pre-computed file counts and important files lists

## Success Criteria

1. Repository analysis verifies ALL expected files are present
2. All documentation links are correctly detected
3. Documentation is downloaded and processed without errors
4. Hierarchical structure is preserved in the extraction
5. Special elements (tables, code, images) are properly extracted
6. Output format is compatible with DuaLipa QA system
7. Integration gracefully handles failures
8. Extraction format meets all requirements for frictionless validation
9. Manual spot checks confirm automated test results
10. Extraction achieves the same results with both Playwright and wget approaches (where applicable)
11. Human review confirms extracted content is usable for downstream QA tasks
12. **NEW: Extracted file counts match repository analysis counts**
13. **NEW: ALL critical files are included in extraction**

## Revised Implementation Path

1. **Repository Analysis Phase (NEW)**:
   - Run analysis script to understand repository structure
   - Count files by type and record statistics
   - Identify important files that must be included
   - Generate validation artifacts for later steps

2. **Collaborative Design Phase**:
   - Define sample inputs and expected outputs BEFORE implementation
   - Create JSON schema for validation
   - Establish frictionless validation patterns and examples
   - Document parent-child relationship requirements
   - **NEW: Define file count expectations and coverage thresholds**

3. **Implementation Phase**:
   - Start with comprehensive repository analysis
   - Implement file discovery based on analysis
   - Build extraction with continuous verification
   - Maintain coverage metrics during extraction
   - **NEW: Validate against pre-analysis expectations at each step**

4. **Verification Phase**:
   - Run automated tests to verify functionality
   - Perform manual spot checks AFTER tests pass
   - Compare extraction with expected output examples
   - Validate with human review of extraction quality
   - Test with real-world documentation sources
   - **NEW: Verify 100% coverage of critical files**

## AI-Specific Adaptation

Given the assistant's limitations in maintaining context:

1. **Self-Contained Test Functions**:
   - Each test function must be completely self-contained
   - Include all verification steps within the function
   - Do not rely on previous test state or context

2. **Inline Expectations**:
   - Include expected counts and values directly in the test
   - Use assertions to verify actual values match expectations
   - Document expected values in comments for clarity

3. **File-Based Communication**:
   - Use files to pass state between test phases
   - Write intermediate verification results to files
   - Read verification artifacts at each step

4. **Continuous Logging**:
   - Log all verification steps and results
   - Create detailed, timestamped logs
   - Include counts and values in log messages

This revised strategy acknowledges the specific limitations of AI assistants and builds verification into every step of the process, treating context limitations as a design constraint rather than an implementation detail.