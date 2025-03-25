# Documentation Integration Task

## Overview

Enhance the DuaLipa extraction system by integrating external documentation sources like ReadTheDocs and ArangoDB documentation. This integration will allow the system to automatically detect documentation links in repositories, download and process the documentation, and incorporate it into the extraction output in a format compatible with the QA system.

## Requirements

1. **Link Detection**: Automatically detect documentation links in repository markdown files
2. **Documentation Download**: Download documentation pages with proper error handling
3. **HTML Processing**: Clean HTML and extract hierarchical section structure
4. **Format Conversion**: Convert documentation to DuaLipa-compatible block format
5. **Integration**: Seamlessly integrate documentation blocks with code extraction
6. **Validation**: Ensure format compatibility with QA system

## Implementation Details

### Link Pattern Detection

Implement pattern matching for:
- ReadTheDocs links (`*.readthedocs.io`, `readthedocs.org`)
- ArangoDB documentation (`docs.arangodb.com`)
- Other common documentation sites (expandable)

### Documentation Download Process

- Use multi-strategy approach for website download:
  - Primary: wget for static websites
  - Fallback: Playwright for JavaScript-heavy websites
- Handle redirects and errors gracefully
- Respect rate limits and robots.txt
- Implement retry logic for transient errors
- Support JavaScript rendering for modern web applications
- Download CSS/JS resources for proper rendering

### HTML Cleaning and Processing

- Remove navigation elements, footers, and other non-content
- Extract section headers and structure
- Process tables, code blocks, and images
- Preserve hierarchical relationships between sections

### DuaLipa Format Integration

- Create documentation blocks with proper UUIDs
- Establish parent-child relationships between sections
- Include metadata (source URL, section type, etc.)
- Format content for QA compatibility

### Testing Strategy

- Unit test each component separately
- Integration test the full extraction pipeline
- Blind test with real-world repositories
- Validate output with QA system
- Implement frictionless validation utilities
- Create quick extract examples for validation
- Enable collaborative validation patterns

## Expected Output Format

Documentation blocks should follow this structure:
```json
{
  "uuid": "<unique-id>",
  "type": "doc_section",
  "name": "<section-title>",
  "content": "<processed-content>",
  "language": "html",
  "file_path": "<relative-path>",
  "parent_uuid": "<parent-section-uuid>",
  "child_uuids": ["<child-section-uuids>"],
  "metadata": {
    "doc_type": "readthedocs|arangodb",
    "header_level": 3,
    "section_hierarchy": ["<parent>", "<current>"],
    "breadcrumb": ["<top-level>", "<second-level>", "<current>"],
    "source_url": "<original-url>",
    "extraction_method": "wget|playwright"
  }
}
```

All blocks MUST include the following fields:
- `uuid`: Unique identifier for the block
- `type`: Type of block (doc_section, doc_code, etc.)
- `name`: Human-readable title/name 
- `content`: The actual content
- `language`: Content language (html, markdown, python, etc.)
- `file_path`: Path to source file
- `metadata`: Additional contextual information

For hierarchical relationships:
- `parent_uuid`: Reference to parent block (except for root blocks)
- `child_uuids`: Array of child block UUIDs (for container blocks)

The metadata must include:
- `doc_type`: Documentation source type
- `section_hierarchy`: Array showing the hierarchical path
- `breadcrumb`: Navigation breadcrumb path
- `source_url`: Original URL of the content

## Success Criteria

1. All documentation from ReadTheDocs and ArangoDB links is properly extracted
2. Documentation hierarchy is preserved
3. Output format is compatible with DuaLipa QA system
4. System handles errors gracefully
5. All tests pass
6. Documentation is comprehensive and clear
7. Successful rendering of JavaScript-heavy sites using Playwright
8. All extraction outputs pass frictionless validation checks
9. Complete block structure with all required fields
10. Proper parent-child relationships maintained
11. Proper validation utilities available for testing

## Timeline

1. Link detection implementation (1 day)
2. Download and HTML processing (2 days)
3. Format conversion and integration (2 days)
4. Playwright integration and JavaScript support (2 days)
5. Validation framework implementation (1 day)
6. Testing and validation (1 day)
7. Documentation and cleanup (1 day)

## Dependencies

- fetch_docs module
- DuaLipa extraction system
- wget for website download
- Playwright for JavaScript-rendered sites
- HTML processing libraries
- Validation utilities for extraction verification
- JSON schema validation tools