# Code Extraction with Documentation Integration

This document describes the code extraction process with documentation integration implemented in DuaLipa.

## Overview

The extraction pipeline processes source code files and optionally enhances them with external documentation. This integration provides a more comprehensive understanding of the codebase by including official documentation from sources like ReadTheDocs and ArangoDB documentation sites.

## Extraction Process

1. **Source File Discovery**: Find all source files in the repository
2. **Language Detection**: Determine the language of each file
3. **Block Extraction**: Extract code blocks (functions, classes, methods)
4. **Documentation Detection**: Find documentation links in markdown files
5. **Documentation Download**: Download and process external documentation
6. **Format Integration**: Combine code and documentation blocks
7. **Format Conversion**: Convert to QA-compatible format

## Documentation Integration

The system automatically detects documentation links in markdown files and enhances the extraction with external documentation. This includes:

- Reading documentation from ReadTheDocs (`*.readthedocs.io`)
- Processing ArangoDB documentation (`docs.arangodb.com`)
- Supporting other documentation sites with similar structure

### Documentation Block Structure

Documentation blocks follow this hierarchical structure:

1. **Documentation Site**: Parent block representing the entire documentation site
2. **Documentation Page**: Individual HTML pages from the site
3. **Documentation Section**: Content sections based on headers
4. **Special Elements**: Code blocks, tables, and images within sections

Each block has proper metadata including:
- Source URL
- Documentation type (readthedocs, arangodb)
- Section hierarchy
- Position information
- Special element indicators

### Integration Process

```python
# Main integration function
def integrate_docs_with_extraction(repo_path, output_blocks):
    # 1. Detect documentation links
    doc_links = detect_doc_links(repo_path)
    
    # 2. Download and process documentation
    downloaded_sites = download_docs(doc_links, docs_dir)
    processed_docs = process_docs(downloaded_sites)
    
    # 3. Convert to DuaLipa format
    doc_blocks = convert_to_dualipa_format(processed_docs, repo_path)
    
    # 4. Append to extraction output
    output_blocks.extend(doc_blocks)
    
    return output_blocks
```

## Format Validation

The extraction output is validated to ensure compatibility with the QA system. This includes:

- Checking required fields and formats
- Validating special elements (code blocks, tables, images)
- Ensuring proper parent-child relationships
- Verifying section hierarchy

## Usage Example

```python
from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks

# Extract code and documentation blocks
blocks = extract_all_blocks("/path/to/repository")

# Blocks now include both code and documentation
```