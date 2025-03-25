# Changelog

## 2025-03-21

### Added
- Import and export extraction for JavaScript/TypeScript files:
  - Tree-sitter based parsing for JS/TS import/export statements
  - Regex fallback for JS/TS import/export extraction when tree-sitter fails
  - Support for React component imports/exports
- Functions for extracting imports/exports in tree_sitter_utils.py:
  - extract_js_ts_imports_exports - primary tree-sitter based extractor
  - _extract_js_ts_imports_exports_regex - fallback regex extractor

### Fixed
- Test failures in test_block_verification.py:
  - Fixed test_block_extraction to ensure blocks have imports/exports
  - Updated code_extractor.py to use js_ts_extractor.py correctly
  - Fixed imports handling in Python blocks
- Tree-sitter initialization in code_extractor.py

### Changed
- Updated JS/TS extraction to properly handle React components
- Enhanced output_formatter.py to display imports and exports in all formats
- Improved error handling in tree_sitter parser initialization

### Pending
- Other test import updates (some tests still reference old module paths)
- Test organization by module type
- More comprehensive test coverage

## 2025-03-20

### Added
- Output formatting utilities for multiple formats:
  - JSON format for machine consumption
  - Markdown format for human reading
  - HTML format for web display
- Block field standardization for QA module compatibility:
  - Consistent field names (uuid/id, source_file/path/file)
  - Automatic block validation and correction
  - Support for different metadata formats
- Improved file output for extraction results:
  - Generated blocks.json for aggregate results
  - Individual block files for easy browsing
  - File structure compatible with QA module

### Fixed
- Backward compatibility with tests using legacy parameters
- Handling of Path objects in JSON serialization
- Field name inconsistencies across extractors
- Test failures in output_examples tests

### Changed
- Extraction format documentation to reflect standardized fields
- Block validation to ensure QA-compatible output
- Repository extraction to support both file and directory inputs