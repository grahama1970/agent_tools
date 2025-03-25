# Fetch Docs Integration

This document provides detailed information on integrating with the fetch_docs module.

## Overview

The fetch_docs module provides functionality for downloading and processing HTML documentation from websites. The extraction module integrates with fetch_docs to download documentation and convert it to the extraction format.

## Components

### DocumentationDownloader

The `DocumentationDownloader` class provides a simplified interface for downloading documentation using either wget or Playwright.

```python
from agent_tools.dualipa.extraction.integration import DocumentationDownloader

# Initialize the downloader
downloader = DocumentationDownloader("output_dir", use_playwright=True)

# Download documentation
success = downloader.download("https://docs.arangodb.com/stable/aql/")

# Download with Playwright directly
stats = downloader.download_with_playwright(
    "https://docs.arangodb.com/stable/aql/",
    recursive=True,
    max_depth=2
)
```

### HTMLProcessor

The `HTMLProcessor` class provides methods for extracting structured content from HTML files.

```python
from agent_tools.dualipa.extraction.integration import HTMLProcessor

# Initialize the processor
processor = HTMLProcessor("output_dir")

# Extract sections from a specific file
sections = processor.extract_sections("output_dir/docs.arangodb.com/stable/aql/index.html")

# Process an entire directory
processed_docs = processor.process_directory(doc_type="arangodb")
```

## Integration with Extraction

After downloading and processing documentation, you can convert it to the extraction format:

```python
from agent_tools.dualipa.extraction.docs_integration import convert_to_dualipa_format

# Convert processed docs to extraction format
blocks = convert_to_dualipa_format(processed_docs, "output_dir")
```

## Configuration Options

You can customize the behavior of the fetch_docs integration by modifying the configuration options in `fetch_docs_config.py`:

```python
# Default options for documentation download
DEFAULT_DOWNLOAD_OPTIONS = {
    "recursive": True,
    "max_depth": 2,
    "use_playwright": False,
    "timeout": 30000  # 30 seconds
}

# Doc type detection mapping
DOC_TYPE_MAPPING = {
    "readthedocs.io": "readthedocs",
    "readthedocs.org": "readthedocs",
    "arangodb.com": "arangodb",
    "docs.python.org": "python",
    "developer.mozilla.org": "mdn"
}

# HTML processing options
HTML_PROCESSING_OPTIONS = {
    "extract_code_blocks": True,
    "extract_tables": True,
    "extract_images": True,
    "min_section_length": 50,  # Minimum content length for a section to be extracted
    "max_section_length": 10000  # Maximum content length for a section
}
```
