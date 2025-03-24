# fetch_docs Task Overview

## Project Goal

Create a robust tool for downloading, processing, and extracting content from documentation websites. The tool should transform HTML documentation into structured JSON data that preserves hierarchical organization and enriches the content with metadata, while also being easily integrable with other modules like DuaLipa.

## Key Requirements

1. **Download Capability**: Recursively download documentation pages from websites while preserving the site structure.
2. **HTML Processing**: Clean and sanitize HTML content, removing unwanted elements while preserving structural meaning.
3. **Section Extraction**: Identify headers and content sections to create a hierarchical representation of documentation.
4. **Metadata Enrichment**: Add token counts, hierarchy information, and other useful metadata to sections.
5. **Special Element Detection**: Extract code blocks, tables, and images as separate entities with appropriate formatting.
6. **Integration Support**: Provide clean interfaces for integration with other tools.
7. **Test Validation**: Ensure all functionality is validated with real-world documentation sources.

## Tasks and Sub-Tasks

### 1. Recursive Download & File Hierarchy Preservation

- [x] **Download Pages:**
  - [x] Use wget to recursively download entire documentation sites.
  - [x] Ensure downloaded files replicate the site's directory structure.
- [x] **File Metadata:**
  - [x] Use Python's `Path` to generate relative file paths and extract file names.
  - [x] Store file hierarchy in JSON field for downstream processing.
- [x] **Tests:**
  - [x] Verify expected files exist in the correct directory structure.
  - [x] Validate that file hierarchy metadata matches downloaded structure.

### 2. HTML Content Extraction & Minimal Cleanup

- [x] **HTML Parsing:**
  - [x] Use BeautifulSoup to parse each downloaded HTML file.
  - [x] Remove unwanted elements without damaging structural tags.
- [x] **Final Cleanup:**
  - [x] Apply sanitization before converting HTML to markdown.
  - [x] Preserve headers and structural elements during conversion.
- [x] **Tests:**
  - [x] Compare cleaned HTML against expected output.
  - [x] Validate that content remains intact after cleanup.

### 3. Section & Content Chunking

- [x] **Header Identification:**
  - [x] Identify header tags to create sections.
  - [x] Map each header to its corresponding content.
- [x] **Internal Hierarchy:**
  - [x] Record section hierarchy information.
  - [x] Maintain proper parent-child relationships.
- [ ] **Merge Short Sections:**
  - [ ] Identify and merge sections below a token threshold.
- [x] **Tests:**
  - [x] Validate section hierarchy extraction.
  - [ ] Verify section merging logic.

### 4. Metadata Enrichment

- [x] **Token Counting:**
  - [x] Count tokens for each section.
  - [x] Attach token counts to each JSON chunk.
- [x] **File & Header Metadata:**
  - [x] Include file hierarchy and header hierarchy in each section.
- [x] **Tests:**
  - [x] Confirm token counts are accurate.
  - [x] Verify hierarchy metadata is correct.

### 5. End-to-End Pipeline Integration

- [x] **Pipeline Integration:**
  - [x] Integrate all components into a unified pipeline.
- [x] **Real-World Testing:**
  - [x] Test with actual documentation sites.
- [x] **Tests:**
  - [x] End-to-end validation of pipeline.
  - [x] Verify JSON output is properly structured.

### 6. DuaLipa Integration

- [ ] **Integration Design:**
  - [ ] Design clean interface for integration with DuaLipa.
  - [ ] Create adapter to convert fetch_docs output to DuaLipa format.
- [ ] **Link Detection:**
  - [ ] Implement detection of documentation links in repositories.
- [ ] **Combined Extraction:**
  - [ ] Support combining documentation blocks with code blocks.
- [ ] **Tests:**
  - [ ] Validate integration works with DuaLipa's extraction pipeline.
  - [ ] Verify combined extraction produces correct output.

## Progress and Status

The core functionality of the module has been implemented, including downloading, HTML processing, section extraction, and metadata enrichment. The module is functioning as a standalone tool.

Current focus is on:
1. Finalizing DuaLipa integration
2. Completing automated test suite
3. Improving documentation and examples

## Next Steps

1. Complete the DuaLipa integration by implementing the integration adapter
2. Implement section merging functionality
3. Add support for additional documentation sources
4. Create comprehensive documentation

## Completion Criteria

The module will be considered complete when:
1. All above tasks are implemented and tested
2. Integration with DuaLipa is functioning properly
3. Documentation is comprehensive
4. Test coverage is adequate