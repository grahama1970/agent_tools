# End-to-End Examples

This directory contains end-to-end examples for the extraction module, including comprehensive blind tests for code and documentation extraction.

## Directory Structure

- **blind_test.py**: Main test script for running blind tests on various file types
- **repository_test.py**: Repository-specific testing module for testing extraction on real-world repositories
- **extraction_blocks.py**: Core functionality for extracting blocks from various file types
- **arangodb_validator.py**: Validation script for ArangoDB documentation extraction
- **arangodb_aql_test.py**: Test script for ArangoDB AQL documentation extraction

## Supported File Types

- **Code Files**: Python (.py), JavaScript (.js), TypeScript (.ts, .tsx)
- **Documentation**: HTML documentation pages from various sources
- **Markdown**: Structured markdown files with sections and elements (tables, code blocks, etc.)

## Running Tests

To run all blind tests:
```bash
python blind_test.py
```

To run only the markdown extraction tests:
```bash
python blind_test.py --markdown-only
```

This will extract and validate two markdown files from the ArangoDB repository:
- ERROR_LEVELS.md: Documentation of ArangoDB's log message error levels
- README.md: Main repository README

To run tests on the specific markdown extraction functionality:
```bash
python blind_test.py --test-markdown-extraction
```

This test checks the extraction of sections and other elements (text blocks, code blocks, tables, images) from the sample MARKDOWN_EXTRACTION.md file.
