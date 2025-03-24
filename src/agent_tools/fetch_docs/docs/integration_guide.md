# Integration Guide for fetch_docs

This document provides guidance on integrating the fetch_docs module with other tools and systems, particularly DuaLipa's extraction pipeline.

## Integration with DuaLipa

### Overview

The integration between fetch_docs and DuaLipa allows for:
1. Detecting documentation links in repositories
2. Downloading and processing those documentation pages
3. Converting documentation into DuaLipa-compatible blocks
4. Merging documentation blocks with code blocks in a unified extraction result

### Integration Steps

#### 1. Create Integration Module in DuaLipa

Create an integration module (e.g., `docs_integration.py`) in the DuaLipa extraction package:

```python
"""
docs_integration.py

This module provides integration between fetch_docs and DuaLipa's extraction pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any

def extract_all_blocks_with_docs(repo_path: Path) -> List[Dict[str, Any]]:
    """
    Enhanced extraction function that includes documentation.
    
    Args:
        repo_path: Directory to extract from
        
    Returns:
        List of extracted blocks including documentation
    """
    # Import DuaLipa extraction
    from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
    
    # Regular extraction
    code_blocks = extract_all_blocks(repo_path)
    
    # Enhance with documentation
    enhanced_blocks = integrate_docs_with_extraction(repo_path, code_blocks)
    
    return enhanced_blocks
```

#### 2. Implement Integration Function

Implement the `integrate_docs_with_extraction` function:

```python
def integrate_docs_with_extraction(repo_path: Path, output_blocks: List[Dict]) -> List[Dict]:
    """
    Main integration function to detect docs, download, and merge with extraction output.
    
    Args:
        repo_path: Path to the repository
        output_blocks: Existing extraction blocks from DuaLipa
        
    Returns:
        Enhanced list of blocks including documentation
    """
    try:
        # Import functions from fetch_docs
        from agent_tools.fetch_docs.link_detector import detect_documentation_links
        from agent_tools.fetch_docs.processor import process_documentation
        
        # Detect documentation links
        doc_links = detect_documentation_links(repo_path)
        
        if not doc_links:
            return output_blocks
            
        # Process documentation
        docs_dir = repo_path / ".dualipa_docs"
        docs_dir.mkdir(exist_ok=True)
        
        doc_data = process_documentation(doc_links, docs_dir)
        
        # Convert to DuaLipa format
        doc_blocks = convert_to_dualipa_format(doc_data, repo_path)
        
        # Append to output
        output_blocks.extend(doc_blocks)
        
    except ImportError:
        # Log error but continue with code blocks only
        pass
        
    return output_blocks
```

#### 3. Implement Format Conversion

Implement the conversion from fetch_docs format to DuaLipa blocks:

```python
def convert_to_dualipa_format(doc_data: Dict, repo_path: Path) -> List[Dict]:
    """
    Convert fetch_docs processed documentation into DuaLipa extraction format.
    
    Args:
        doc_data: Dictionary of processed documentation data
        repo_path: Path to the repository (for reference)
        
    Returns:
        List of DuaLipa-compatible blocks
    """
    import uuid
    
    dualipa_blocks = []
    
    # Implementation details...
    
    return dualipa_blocks
```

### Usage Example

```python
from agent_tools.dualipa.extraction.docs_integration import extract_all_blocks_with_docs
from pathlib import Path
import json

# Extract blocks including documentation
blocks = extract_all_blocks_with_docs(Path("/path/to/repo"))

# Write to file
with open("extraction_with_docs.json", "w") as f:
    json.dump(blocks, f, indent=2)
```

## Integration with Other Systems

### General Integration Pattern

When integrating fetch_docs with other systems:

1. **Link Detection**: Use `detect_documentation_links` to find documentation URLs
2. **Documentation Processing**: Process links using `process_documentation`
3. **Format Conversion**: Convert to the target system's format

### Integration with Vector Databases

For integrating with vector databases:

1. Process documentation using fetch_docs
2. Generate embeddings for each section
3. Store in the vector database with appropriate metadata

### Integration with QA Systems

For question-answering systems:

1. Process documentation using fetch_docs
2. Convert sections to appropriate context format
3. Use sections as context for LLM-based QA

## API Reference

### Key Functions for Integration

1. `detect_documentation_links(repo_path: Path) -> List[str]`
   - Detects documentation links in a repository
   - Returns a list of URLs

2. `process_documentation(urls: List[str], output_dir: Path) -> Dict`
   - Downloads and processes documentation from URLs
   - Returns structured documentation data

3. `clean_html(html_content: str) -> str`
   - Cleans HTML content for further processing
   - Returns sanitized HTML

4. `extract_sections_from_html(html_content: str, file_path: str) -> List[Dict]`
   - Extracts sections from HTML content
   - Returns list of section dictionaries with metadata