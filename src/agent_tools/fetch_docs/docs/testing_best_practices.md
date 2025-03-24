# Testing Best Practices for fetch_docs

This document outlines testing strategies and best practices for the fetch_docs module, with a focus on human-verifiable testing patterns.

## Testing Challenges

Documentation extraction presents unique testing challenges:

1. **Complex Input/Output Relationships**: HTML input transforms into hierarchical data structures
2. **Varied Formats**: Different documentation sites use different HTML structures
3. **Visual Context**: Some relationships are clear visually but hard to verify programmatically
4. **Scale**: Documentation can be large and complex, making full coverage difficult

## Human-Verifiable Testing

For complex extraction tasks, we've found that human-verifiable testing provides significant advantages:

### Key Principles

1. **Transparency**: Save all inputs, intermediate steps, and outputs
2. **Visual Comparison**: Create side-by-side views of input and output
3. **Statistics**: Provide quantitative metrics about extraction results
4. **Artifacts**: Generate files that can be inspected independently
5. **Documentation**: Include clear instructions for verification

### Implementation

We've implemented transparent testing in `test_arangodb_extraction_transparent.py` and `test_readthedocs_extraction_transparent.py`. These tests:

1. Download documentation from source
2. Process it through the extraction pipeline
3. Generate HTML reports with visual comparisons
4. Save all artifacts for inspection

### Benefits

- **Better Debugging**: When issues occur, all intermediate artifacts are available
- **Stakeholder Verification**: Non-technical reviewers can visually verify results
- **Comprehensiveness**: Both automated checks and human intelligence are applied
- **Documentation**: The tests themselves serve as documentation of expected behavior

## Test Types

### 1. Unit Tests

For individual components like HTML cleaning, link detection, etc.

```python
def test_clean_html_removes_scripts():
    html = "<html><script>alert('test')</script><p>Content</p></html>"
    cleaned = clean_html(html)
    assert "<script>" not in cleaned
    assert "<p>Content</p>" in cleaned
```

### 2. Integration Tests

For testing the full extraction pipeline.

```python
def test_process_documentation_integration():
    url = "https://docs.example.com"
    result = process_documentation([url], temp_dir)
    assert url in result
    assert len(result[url]) > 0
```

### 3. Transparent Verification Tests

For human verification of complex extraction results.

```python
def test_transparent_extraction():
    # Download docs
    html_file = download_docs(output_dir)
    
    # Process docs
    blocks = process_docs(html_file, output_dir)
    
    # Create HTML report for human verification
    create_html_summary(html_file, blocks, output_dir)
    
    # Basic automated checks
    assert len(blocks) > 0
    assert any(b["type"] == "doc_section" for b in blocks)
```

## Verification Process

When running transparent tests, follow this verification process:

1. Examine the HTML summary report
2. Check that all expected block types are present
3. Verify that sections match their original HTML counterparts
4. Confirm that code blocks, tables, and other special elements are correctly extracted
5. Check hierarchical relationships between blocks

## CI Integration

For CI/CD pipelines:

1. Run unit and integration tests with automated assertions
2. Run transparent tests to generate artifacts
3. Store artifacts as build artifacts
4. Link to artifacts in PR comments for human review

## Dependency and Import Considerations

When working with the fetch_docs module, be aware of these important requirements:

1. **Required Dependencies**:
   - **BeautifulSoup (bs4)**: For HTML parsing and manipulation
   - **lxml**: The HTML parser used by BeautifulSoup in our module
   - **loguru**: For structured logging throughout the module
   - **spacy**: For natural language processing (with `en_core_web_sm` model)
   
2. **Import Path Considerations**:
   - Always use absolute imports: `from agent_tools.fetch_docs.clean_html import clean_html`
   - Avoid relative imports that go outside the current directory (no `from ..module import x`)
   - Ensure PYTHONPATH includes both the `src` and `tests` directories

3. **Common Pitfalls**:
   - The `extract_sections_from_html` function expects a `Path` object, not a string, for the `file_path` parameter
   - The `process_documentation` function returns a specific dictionary structure that must be maintained
   - Custom functions must match the expected interface when integrating with other modules

## Recommended Tools

- **BeautifulSoup**: For parsing and manipulating HTML
- **jq**: For inspecting JSON output files
- **diff-so-fancy**: For improved diff visualization
- **pytest**: For running all test types

## Example Commands

```bash
# Run all tests
pytest -xvs tests/

# Run only transparent tests
pytest -xvs tests/test_*_transparent.py

# Run all tests and generate coverage report
pytest --cov=agent_tools.fetch_docs tests/
```