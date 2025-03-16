# Task.md: Fetch Page Project – First-Pass Extraction MVP

## Overview
This project creates a pipeline that:
- Recursively downloads documentation pages (using wget or similar) while preserving the file structure.
- Extracts and cleans HTML content with minimal processing.
- Converts HTML to markdown (using tools like markdownify) while maintaining header and section hierarchies.
- Chunks the page content into ordered JSON objects enriched with metadata (file hierarchy, header hierarchy, token counts).
- Uses real-world, blind tests (i.e., tests based on actual extracted results) to verify functionality.

**Key Principles:**
- Start simple with an MVP; add complexity iteratively.
- Use tests to guide functionality and understanding (non-mocked, blind checks).
- Always reference official documentation (links must be included in source files and tests).

---

## Tasks and Sub-Tasks

### 1. Recursive Download & File Hierarchy Preservation
- [ ] **Download Pages:**
  - [ ] Use wget (or an equivalent tool) to recursively download the entire documentation.
  - [ ] Ensure the downloaded files replicate the site's directory structure.
- [ ] **File Metadata:**
  - [ ] Use Python's `Path` to generate relative file paths and extract file names (`Path.stem`).
  - [ ] Store the file hierarchy in a JSON field (e.g., `"file_hierarchy": [parent_path, child_path, current_file]`).
- [ ] **Tests:**
  - [ ] Verify that expected files (by known file names) exist in the correct directory structure.
  - [ ] Blind-check that the generated file hierarchy metadata matches the downloaded structure.

---

### 2. HTML Content Extraction & Minimal Cleanup
- [ ] **HTML Parsing:**
  - [ ] Use BeautifulSoup to parse each downloaded HTML file.
  - [ ] Remove unwanted elements (scripts, styles, navigation, etc.) without damaging structural tags.
- [ ] **Final Cleanup:**
  - [ ] Apply Bleach (or a similar tool) only right before converting HTML to markdown.
  - [ ] Use markdownify to convert cleaned HTML into markdown while preserving headers.
- [ ] **Tests:**
  - [ ] Compare the cleaned HTML against a known expected output (blind check).
  - [ ] Validate that extraneous elements are removed but key content remains intact.

---

### 3. Section & Content Chunking
- [ ] **Header Identification:**
  - [ ] Identify header tags (h1, h2, …) in order to create sections.
  - [ ] Each header should start a new section with its header text.
- [ ] **Internal Hierarchy:**
  - [ ] Record the internal section hierarchy as an ordered list (e.g., `["# Parent", "## Child", "### Current"]`).
  - [ ] Ensure that when a new parent section (e.g., new h1) is encountered, it starts a fresh hierarchy (children from previous sections are not inherited).
- [ ] **Merge Short Sections:**
  - [ ] Use SpaCy or a sentence tokenizer to count sentences in each section.
  - [ ] Merge any section with content under 2 sentences (or under a predetermined token threshold).
- [ ] **Tests:**
  - [ ] Validate the extracted section hierarchy from a sample HTML page against an expected structure.
  - [ ] Blind-check that sections below the threshold are merged properly and empty sections are not stored.

---

### 4. Metadata Enrichment
- [ ] **Token Counting:**
  - [ ] Use SpaCy to count tokens for each section.
  - [ ] Attach token counts to each JSON chunk.
- [ ] **File & Header Metadata:**
  - [ ] Ensure each chunk includes file hierarchy metadata (from Task 1) and internal header hierarchy.
- [ ] **Tests:**
  - [ ] Confirm that token counts for sections match expected values from a sample document.
  - [ ] Verify that file and header hierarchies are correctly embedded in the JSON output.

---

### 5. End-to-End Integration Testing (Blind Check)
- [ ] **Pipeline Integration:**
  - [ ] Integrate all components (download, extraction, cleanup, chunking, metadata enrichment) into one pipeline.
- [ ] **Real-World Run:**
  - [ ] Run the pipeline on a known documentation site (e.g., Requests' documentation).
- [ ] **Tests:**
  - [ ] Perform an end-to-end blind check by comparing key properties of the resulting JSON output (e.g., order of sections, file paths, header hierarchies, token counts) against known values.
  - [ ] Verify that the overall JSON list object is highly ordered and complete.

---

### 6. Documentation & Usage Example
- [ ] **Code Documentation:**
  - [ ] Include official documentation links at the top of every file (e.g., BeautifulSoup, Bleach, etc.).
- [ ] **Usage Example:**
  - [ ] Implement an `if __name__ == "__main__"` block to run a demo.
  - [ ] Provide a clear usage example that demonstrates the pipeline on a small subset of pages.
- [ ] **Tests:**
  - [ ] Verify that running the demo produces output files with the expected JSON structure.
  - [ ] Ensure that these tests serve as blind checks for actual functionality rather than relying on mocks.

---

## Additional Considerations
- [ ] Ensure tests are written to exercise real functionality (not mocked) to validate the true output.
- [ ] Use tests to help reason about file interdependencies and why functions might break.
- [ ] Keep the MVP simple and add advanced features (like table enrichment with vision models) in later passes.
- [ ] Continuously update this document with any new requirements or insights during iterative development.

---

*This Task.md document will guide the development process by ensuring each step is verifiable with real data and that the extraction pipeline remains aligned with the lessons learned from previous implementations.*
