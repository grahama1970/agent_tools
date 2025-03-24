# Documentation Extraction Verification

This document explains how to run the transparent verification tests for the fetch_docs integration with dualipa. These tests download HTML documentation from websites, process it through the extraction pipeline, and create human-readable verification artifacts.

## Key Features

- Downloads and saves the original HTML from documentation sites
- Extracts and saves the processed blocks in JSON format
- Creates HTML reports with side-by-side comparisons of input and output
- Shows statistics on extracted blocks (sections, code blocks, tables, etc.)
- Provides shell commands for further inspection of results

## Available Test Scripts

### Combined Test Runner

The `run_transparent_tests.py` script runs all tests in sequence and creates a single summary report:

```bash
python run_transparent_tests.py --output-dir test_results
```

This will:
1. Run both ArangoDB and ReadTheDocs extraction tests
2. Create a combined summary report
3. Save all results in one organized directory
4. Automatically open the summary in your browser

### Individual Test Scripts

You can also run tests individually:

#### ArangoDB Test

Tests extraction from ArangoDB's AQL documentation:

```bash
python test_arangodb_extraction_transparent.py --output-dir test_results/arangodb
```

#### ReadTheDocs Test

Tests extraction from Python's ReadTheDocs documentation:

```bash
python test_readthedocs_extraction_transparent.py --output-dir test_results/readthedocs
```

## Verification Process

The test output includes several artifacts for human verification:

1. **Original HTML:** The raw HTML downloaded from the documentation site
2. **Extracted Blocks:** The JSON output from the extraction pipeline
3. **HTML Summary:** A visual report showing:
   - Statistics on extracted blocks
   - Side-by-side comparisons of HTML input and JSON output
   - Links to all generated files
   - Sample commands for further inspection

## Understanding the Results

The HTML reports are designed to help you verify that:

1. **All block types are extracted:** Documentation site, pages, sections, code blocks, and tables
2. **Hierarchical relationships are preserved:** Parent-child relationships between blocks
3. **Content is properly extracted:** Headers, code samples, tables, etc.

## Example Commands for Inspection

After running a test, you can use these commands to inspect the results:

```bash
# Count blocks by type
cat test_results/arangodb/arangodb_blocks.json | grep "type" | sort | uniq -c

# Check all code blocks
cat test_results/arangodb/arangodb_blocks.json | jq '.[] | select(.type == "code_block")'

# Examine section hierarchy
cat test_results/arangodb/arangodb_blocks.json | jq '.[] | select(.type == "doc_section") | {name: .name, header_level: .metadata.header_level}'
```

## Requirements

- Python 3.8+
- BeautifulSoup4 (for HTML parsing)
- Dependencies from agent_tools.fetch_docs and agent_tools.dualipa modules

## Troubleshooting

If downloads fail, the tests will create fallback HTML files with minimal content to allow testing to continue.

If you encounter the error "Could not import download_site function", make sure the fetch_docs module is in your Python path, or use the local download_site_patch.py which is included as a fallback.