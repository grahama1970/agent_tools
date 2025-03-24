# Changelog

## [0.2.0] - 2025-03-24

### Added
- Finalized conversion script (`convert_for_validation.py`) for transforming raw extraction outputs to validation format
- Added document type detection with `detect_doc_type.py` for automatic template selection
- Created specialized templates for different documentation formats:
  - `arangodb_expected_format.json` for ArangoDB API documentation
  - `html_docs_expected_format.json` for HTML-based documentation
  - `markdown_docs_expected_format.json` for Markdown-based documentation
- Enhanced validation tools with additional options:
  - Automatic document type detection with `--auto-detect`
  - Format conversion with `--convert`
  - Processing directories with `--input-dir`
- Improved validation summary reporting with document type statistics
- Added comprehensive documentation in `VALIDATION_FRAMEWORK.md`

### Changed
- Enhanced conversion process to better handle hierarchical relationships
- Improved table and code block extraction from content
- Updated validation output format with more detailed error reporting
- Enhanced template detection to intelligently select the appropriate format

### Fixed
- Resolved issues with nested hierarchical structures in conversion
- Fixed table content parsing for different content formats
- Improved error handling in validation functions

## [0.1.0] - 2025-03-22

### Added
- Initial validation framework with core functions in `validation.py`
- Basic extraction format validation
- Hierarchical structure validation
- Content validation against expected formats
- Format consistency validation
- Test utilities for running validation
- Expected format templates for LENGTH and ARRAY_INTERSECTION functions
- Basic conversion tool for extraction outputs

### Changed
- Updated test cases to use the validation framework
- Implemented TDD approach for documentation extraction

### Fixed
- Addressed inconsistencies in extraction structure
- Fixed parent-child relationship validation