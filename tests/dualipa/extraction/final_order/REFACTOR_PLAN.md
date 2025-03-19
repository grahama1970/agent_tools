# Test Refactoring Plan

## Current Issues
- Duplicate tests across files
- Tests depending on functionality tested in later files
- Integration tests before component tests
- Mixed concerns in test files
- **Missing format validation tests**
- **Missing hierarchy validation**

## New Test Structure

### Level 1: Core Infrastructure (01-19)
- `test_01_simple.py` - Basic smoke test ✓
- `test_02_import.py` - Import verification ✓
- `test_05_stats_consistency.py` - Stats validation ✓
- `test_10_github_utils.py` - GitHub utilities ✓
- `test_15_language_detection.py` - Language detection (moved from 45)
- `test_17_format_validation.py` - **NEW: Validates extraction format schema**

### Level 2: Block Structure (20-29)
- `test_20_block_verification.py` - Block structure verification
  - UUID generation and linking
  - Breadcrumb validation
  - Hierarchical depth checking
  - Content flags validation
- `test_25_tree_sitter_hierarchy.py` - Tree-sitter parsing and AST validation

### Level 3: Individual Extractors (30-49)
- `test_30_python_extraction.py` - Python extraction
  - Function/class extraction
  - Docstring handling
  - Type hint preservation
  - Q&A generation for Python blocks
- `test_35_js_ts_extraction.py` - JS/TS extraction
  - Function/class/interface extraction
  - JSX/TSX support
  - Type definition handling
  - Q&A generation for JS/TS blocks
- `test_40_markdown_extraction.py` - Markdown extraction
  - Section hierarchy
  - Code block extraction
  - Content flag detection
  - Q&A generation for documentation
- `test_45_generic_extraction.py` - Generic file extraction
  - Basic text extraction
  - Binary file handling
  - Unknown format handling

### Level 4: Metadata and Cross-References (50-59)
- `test_50_metadata_validation.py` - **NEW: Validates block metadata**
  - Content flags
  - Extraction focus
  - Summary instructions
  - Test coverage info
  - Version history
- `test_55_cross_references.py` - **NEW: Tests cross-reference handling**
  - Parent-child relationships
  - Dependency tracking
  - Cross-language implementations
  - Protocol conformance

### Level 5: Integration (60-89)
- `test_60_extractor_integration.py` - Main extractor integration
  - Full extraction pipeline
  - Format compliance
  - Hierarchy preservation
- `test_70_multilang_extractor.py` - Multi-language scenarios ✓
- `test_80_output_examples.py` - Output validation ✓

### Level 6: Repository Operations (90-99)
- `test_90_repo_operations.py` - Full repository operations ✓

## Migration Steps
1. Create new format validation tests first
2. Move language detection tests
3. Update block verification to include format requirements
4. Add metadata validation tests
5. Update extractor tests to verify format compliance
6. Consolidate and clean up integration tests

## Files to Delete After Migration
- `test_40_code_extractor.py` (after splitting)
- `test_41_python_extraction.py` (after consolidation)
- `test_41_sample_block_extraction.py` (after consolidation)
- `test_42_js_ts_extraction.py` (after consolidation)
- `test_42_realworld_block_extraction.py` (merge into integration)
- `test_43_markdown_extraction.py` (after consolidation)
- `test_44_generic_extraction.py` (after moving)
- `test_45_language_detection.py` (after moving)
- `test_46_repository_integration.py` (merge into 90)

## Critical Format Requirements
1. Every extracted block must have:
   - Unique UUID
   - Proper parent/child relationships
   - Complete breadcrumb path
   - Accurate depth information
   - Content flags
   - Q&A generation support

2. Language-specific blocks must include:
   - Proper language identification
   - Language-specific parsing rules
   - Test coverage information
   - Version history
   - Dependencies

3. Documentation blocks must include:
   - Section role
   - Extraction focus
   - Summary instructions
   - Content flags
   - TOC format 