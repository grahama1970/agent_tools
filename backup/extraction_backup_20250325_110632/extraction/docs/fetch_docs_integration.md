# DuaLipa Fetch Docs Integration

This document outlines the integration of fetch_docs with the DuaLipa extraction module. It covers the architecture, data flow, and testing approach for documentation extraction.

## Overview

The `fetch_docs` integration allows DuaLipa to:

1. Automatically detect documentation links in a repository
2. Download documentation pages from external sites
3. Process HTML content into a structured format
4. Extract code blocks, tables, and images from documentation
5. Create a hierarchical structure with proper parent-child relationships
6. Integrate documentation blocks with code blocks in the extraction output
7. Validate the extracted documentation against expected formats
8. Visualize the hierarchical structure for QA and debugging

## Architecture

The integration follows a modular architecture:

```
fetch_docs_integration.py   - Integration module
  │
  ├─ detect_doc_links()     - Scans repository for documentation links
  ├─ download_docs()        - Downloads documentation from links
  ├─ process_docs()         - Processes HTML content into structured format
  ├─ detect_special_elements() - Extracts code blocks, tables, images
  ├─ convert_to_dualipa_format() - Converts to DuaLipa block format
  └─ integrate_docs_with_extraction() - Merges with code extraction
```

The `extraction_blocks.py` module calls `integrate_docs_with_extraction()` at the end of its extraction process to enhance code blocks with documentation blocks.

## Data Flow

1. Repository scanning:
   - Scan markdown files in repository for documentation links
   - Extract URLs for ReadTheDocs, ArangoDB, and other documentation sites

2. Download and processing:
   - Download HTML content from documentation sites
   - Clean HTML to remove navigation, ads, and other non-content elements
   - Extract sections based on heading structure
   - Identify code blocks, tables, and images within sections

3. Conversion to DuaLipa format:
   - Create block hierarchy: documentation → doc_page → doc_section → elements
   - Generate UUIDs and establish parent-child relationships
   - Add metadata for each block type
   - Create blocks for special elements (code blocks, tables, images)

4. Integration with code extraction:
   - Combine documentation blocks with code blocks
   - Ensure proper parent-child relationships
   - Maintain bidirectional references

## Block Hierarchy

Documentation extraction creates a hierarchy of blocks:

1. **documentation** (Level 0):
   - Represents a documentation site (e.g., Python ReadTheDocs)
   - Contains doc_page blocks as children

2. **doc_page** (Level 1):
   - Represents an individual HTML page
   - Contains doc_section blocks as children

3. **doc_section** (Level 2):
   - Represents a section within a page (based on headings)
   - Contains special element blocks as children

4. **code_block**, **table**, **image** (Level 3):
   - Represent special elements within sections
   - No further children

This hierarchy maintains proper parent-child relationships and is compatible with the existing code block structure.

## Block Types and Fields

Each block type has specific fields:

### documentation

```json
{
  "uuid": "unique-id",
  "id": "docs_site_name",
  "name": "Documentation: site_name",
  "type": "documentation",
  "language": "html",
  "content": "Documentation site: URL",
  "file_path": "path_to_repo",
  "source_url": "documentation_url",
  "child_uuids": ["page1_uuid", "page2_uuid"],
  "metadata": {
    "language": "html",
    "source_url": "documentation_url",
    "doc_type": "readthedocs|arangodb"
  }
}
```

### doc_page

```json
{
  "uuid": "unique-id",
  "id": "docs_site_name_page_name",
  "name": "page_name",
  "type": "doc_page",
  "language": "html",
  "content": "Documentation page content",
  "file_path": "path_to_html",
  "parent_uuid": "documentation_uuid",
  "child_uuids": ["section1_uuid", "section2_uuid"],
  "metadata": {
    "language": "html",
    "source_url": "page_url",
    "relative_path": "relative/path/to/page",
    "doc_type": "readthedocs|arangodb"
  }
}
```

### doc_section

```json
{
  "uuid": "unique-id",
  "id": "docs_site_name_page_name_section_index",
  "name": "Section Title",
  "type": "doc_section",
  "language": "html",
  "content": "Section content in HTML",
  "file_path": "path_to_html",
  "parent_uuid": "page_uuid or parent_section_uuid",
  "child_uuids": ["code1_uuid", "table1_uuid"],
  "metadata": {
    "language": "html",
    "source_url": "documentation_url",
    "position": 0,
    "doc_type": "readthedocs|arangodb",
    "header_level": 2,
    "token_count": 150,
    "section_hierarchy": ["Page Title", "Section Title"],
    "has_code": true,
    "has_tables": true,
    "has_images": false
  }
}
```

### code_block, table, image

```json
{
  "uuid": "unique-id",
  "id": "docs_site_name_page_name_section_index_element_type_index",
  "name": "Element Name",
  "type": "code_block|table|image",
  "language": "detected_language|html",
  "content": "Element content",
  "file_path": "path_to_html",
  "parent_uuid": "section_uuid",
  "child_uuids": [],
  "metadata": {
    "language": "detected_language|html",
    "source_url": "documentation_url",
    "position": 0,
    "doc_type": "readthedocs|arangodb",
    "element_type": "code_block|table|image",
    "is_embedded": true,
    "section_hierarchy": ["Page Title", "Section Title"]
  }
}
```

## Testing

### Transparent Testing

The fetch_docs integration includes "transparent testing" that:

1. Downloads real documentation pages
2. Processes them through the extraction pipeline
3. Validates the extraction output against expected formats
4. Visualizes the hierarchical structure for verification

Test files:
- `test_readthedocs_extraction_transparent.py`: Tests ReadTheDocs extraction
- `test_arangodb_extraction_transparent.py`: Tests ArangoDB documentation extraction

### Validation Framework

The validation framework verifies:

1. **Structure**: Checks that block types follow the correct hierarchy
2. **Content**: Verifies that blocks have the required fields
3. **Relationships**: Ensures proper parent-child bidirectional references
4. **Format**: Validates compatibility with the QA system

### Visualization

The HTML visualization shows:
- The hierarchical structure of blocks
- Parent-child relationships
- Block content and metadata
- Validation results

## Docker Integration

The tests can be run in Docker to ensure consistent testing environments:

```bash
# Run all transparent tests in Docker
docker-compose -f docker-compose.yml up test-transparent

# View visualization in browser
docker-compose -f docker-compose.yml up server
# Access at http://localhost:8000/test_results_dashboard/summary.html
```

## Implementation Details

### Link Detection

The system looks for documentation links in repository markdown files using regular expressions:

```python
DOC_PATTERNS = [
    # Read the Docs
    r'https?://[a-zA-Z0-9-]+\.readthedocs\.io/[^\s)"\']+',
    
    # ArangoDB Documentation
    r'https?://(www\.)?arangodb\.com/docs/[^\s)"\']+',
    
    # Generic documentation patterns
    r'https?://docs\.[a-zA-Z0-9-]+\.[a-zA-Z]+/[^\s)"\']+',
]
```

### HTML Processing

1. **Cleaning**: Removes navigation, ads, scripts, and other non-content elements
2. **Section Extraction**: Identifies sections based on heading tags (h1-h6)
3. **Special Element Detection**: Extracts code blocks, tables, and images
4. **Hierarchical Structure**: Maintains nested section relationships

### Integration with Extraction Pipeline

The `extract_all_blocks` function in `extraction_blocks.py` calls `integrate_docs_with_extraction` to enhance the extraction output with documentation blocks:

```python
# Try to enhance with documentation from fetch_docs
try:
    from agent_tools.dualipa.fetch_docs_integration import integrate_docs_with_extraction
    logger.info("Enhancing extraction with documentation from fetch_docs")
    all_blocks = integrate_docs_with_extraction(source_dir, all_blocks)
    logger.info(f"Enhanced extraction complete: {len(all_blocks)} blocks total (including documentation)")
except ImportError:
    logger.info("fetch_docs_integration module not found, skipping documentation enhancement")
except Exception as e:
    logger.error(f"Error enhancing extraction with documentation: {e}")
```

## Future Work

1. Implement dedicated HTML extractors in the extraction/extractors directory
2. Add support for more documentation types (MDN, JavaDocs, etc.)
3. Enhance HTML cleaning for better extraction quality
4. Implement semantic chunking for improved section detection
5. Add support for API documentation formats (OpenAPI, Swagger)
6. Implement automatic documentation updates based on repository changes