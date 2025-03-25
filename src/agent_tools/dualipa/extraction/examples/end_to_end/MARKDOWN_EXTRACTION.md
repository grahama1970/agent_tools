# Markdown Extraction

This document describes the markdown extraction functionality in the DuaLipa extraction module.

## Overview

The markdown extraction system in DuaLipa processes markdown files and extracts structured information such as:

1. Hierarchical sections based on heading levels
2. Text blocks 
3. Code blocks with their language specifiers
4. Tables with rows and columns
5. Images with alt text and URLs

## Implementation Details

### Section Extraction

Sections are extracted based on heading levels (# for h1, ## for h2, etc.) with proper hierarchical relationships maintained.

```python
def extract_markdown_sections(content: str, file_path: str, parent_uuid: str) -> List[Dict[str, Any]]:
    """Extract sections from markdown files based on headings."""
    # Implementation details...
    return sections
```

### Element Extraction

Special elements like code blocks, tables, and images are extracted with their metadata.

| Element Type | Detection Method | Metadata Extracted |
|--------------|------------------|-------------------|
| Code Blocks | Triple backtick | Language, content |
| Tables | Pipe character rows | Row count, column count |
| Images | ![alt](url) syntax | Alt text, URL |

## Testing

The extraction process is tested using both unit tests and integration tests:

1. Unit tests verify the individual extraction functions
2. Integration tests process real-world markdown files
3. Blind tests validate the extraction on unseen files

![Extraction Process](https://example.com/extraction.png)

## Performance Considerations

- Efficient regex patterns for section matching
- Single-pass processing for optimal performance
- Proper handling of large files

## Future Improvements

* Better handling of nested formatting
* More robust table extraction with cell merging
* Support for more markdown extensions