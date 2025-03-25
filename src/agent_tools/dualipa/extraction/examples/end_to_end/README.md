# DuaLipa Documentation Integration

This module integrates external documentation sources with the DuaLipa extraction pipeline. It automatically detects documentation links in repositories, downloads and processes the documentation content, and integrates it with code extraction in a format compatible with the QA system.

## Features

- Automatic detection of documentation links in markdown files
- Support for ReadTheDocs and ArangoDB documentation formats
- HTML cleaning and section extraction
- Hierarchical content structure preservation
- Format conversion to DuaLipa-compatible blocks
- Seamless integration with code extraction

## Usage

### Basic Usage

```python
from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks

# Extract code and documentation blocks
blocks = extract_all_blocks("/path/to/repository")

# Documentation blocks are automatically included in the output
```

### Direct Documentation Integration

```python
from agent_tools.dualipa.fetch_docs_integration import integrate_docs_with_extraction

# Get code blocks from extraction
code_blocks = [...]  # Your extracted code blocks

# Enhance with documentation
all_blocks = integrate_docs_with_extraction("/path/to/repository", code_blocks)
```

## Testing

Run the test suite to validate the integration:

```bash
# General integration test
python test_fetch_docs_integration.py

# ArangoDB documentation blind test
python blind_test.py --arangodb-docs-only

# ArangoDB AQL main page specific test
python blind_test.py --aql-main-page-only

# Run all blind tests
python blind_test.py

# Run transparent tests with Playwright support for JavaScript-rendered sites
python run_transparent_tests.py --playwright
```

### JavaScript-Rendered Websites

For JavaScript-heavy websites where content is dynamically generated on the client side, use the `--playwright` flag to enable browser-based extraction:

```bash
# Test ArangoDB extraction with Playwright
python test_arangodb_extraction_transparent.py --playwright

# Test all documentation extraction with Playwright
python run_transparent_tests.py --playwright

# Test a specific URL with Playwright
python test_playwright_fetch.py https://example.com/docs --output-dir test_output
```

This uses a headless browser to properly render JavaScript content before extraction. See `PLAYWRIGHT_SUPPORT.md` for detailed information.

### Validation and Verification

For validating extraction results and ensuring quality:

```bash
# Validate extraction against expected format
python test_validation_framework.py --extraction output.json --expected expected_format.json

# Run all validation tests
python validate_all_tests.py --output-dir validation_results
```

For more information on validation approaches:
- See `VALIDATION_FRAMEWORK.md` for the technical validation framework
- See `FRICTIONLESS_VALIDATION.md` for best practices on verification and collaboration

### AQL-Specific Testing

The system includes specialized testing for the ArangoDB AQL main documentation page 
(https://docs.arangodb.com/stable/aql/), which is a critical documentation resource. 
This test validates that:

1. The main AQL page is correctly identified and extracted
2. Code blocks containing AQL examples are properly structured
3. Tables of AQL operations are correctly formatted
4. Parent-child relationships between documentation elements are maintained

For more details, see `AQL_INTEGRATION.md`.

## Architecture

1. **Link Detection**: Scans repository files for documentation links
2. **Documentation Download**: Downloads and saves documentation pages
3. **HTML Processing**: Cleans HTML and extracts sections
4. **Format Conversion**: Converts to DuaLipa-compatible format
5. **Integration**: Merges documentation blocks with code blocks

## Supported Documentation Sources

- ReadTheDocs (`*.readthedocs.io`, `readthedocs.org`)
- ArangoDB Documentation (`docs.arangodb.com`)
  - General ArangoDB docs
  - AQL main documentation page (https://docs.arangodb.com/stable/aql/)
  - AQL operations (https://docs.arangodb.com/stable/aql/operations/)
  - Indexing documentation (https://docs.arangodb.com/stable/indexing/)

## Format

Documentation blocks follow this structure:

```json
{
  "uuid": "<unique-id>",
  "id": "docs_<section-name>",
  "name": "Documentation: <section-title>",
  "type": "documentation",
  "language": "html",
  "content": "<processed-content>",
  "file_path": "<relative-path>",
  "source_url": "<original-url>",
  "child_uuids": ["<child-section-uuids>"],
  "metadata": {
    "language": "html",
    "source_url": "<original-url>",
    "doc_type": "readthedocs|arangodb",
    "section_hierarchy": ["<parent>", "<current>"]
  }
}
```