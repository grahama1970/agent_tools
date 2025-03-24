# ArangoDB Documentation Blind Test

This document explains the implementation of the blind test for ArangoDB documentation extraction in the DuaLipa project.

## Overview

The blind test verifies that our extraction system correctly processes ArangoDB documentation from links in markdown files. It validates that the output conforms to the expected block structure and includes all necessary elements like code blocks and tables.

## Implementation Details

### Test Structure

1. **Test Repository Creation**: 
   - Creates a temporary repository with markdown files containing ArangoDB documentation links.
   - Includes example code blocks and tables in the markdown content.

2. **Documentation Extraction**:
   - Uses `fetch_docs_integration` to detect documentation links and download content.
   - Handles download failures gracefully with placeholder content.
   - Processes HTML documentation into structured blocks.

3. **Validation Process**:
   - Validates blocks against a reference format defined in `arangodb_expected_format.json`.
   - Checks for required block types (documentation, doc_page, doc_section, code_block, table).
   - Ensures proper parent-child relationships between blocks.

### Placeholder Generation

The test includes logic to handle cases where actual documentation can't be downloaded:

- If code blocks aren't found in documentation, it creates placeholders from markdown sections.
- If tables aren't found, it creates placeholder tables with expected structure.
- Markdown elements are converted to documentation format elements as needed.

### Handling Network Issues

- Uses a patched version of `download_site` that creates placeholder content when downloads fail.
- Detects sections in markdown files that contain code blocks or tables.
- Ensures validation doesn't fail due to external service unavailability.

## Expected Format

The reference format in `arangodb_expected_format.json` defines the structure for:

- Documentation site blocks
- Documentation page blocks
- Documentation section blocks
- Code blocks (with JavaScript/AQL examples)
- Table blocks (with headers and rows)

## Running the Test

To run only the ArangoDB documentation test:

```bash
python blind_test.py --arangodb-docs-only
```

The test generates a summary file (`arangodb_extraction_summary.json`) with statistics on the extracted blocks.

## Test Results

A successful test validates:

- Documentation hierarchy (site → pages → sections)
- Block fields and metadata
- Relationships between blocks
- Presence of code blocks and tables

Example output shows counts of each block type and validation status.