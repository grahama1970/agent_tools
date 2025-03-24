# fetch_docs Module

## Overview

The `fetch_docs` module is a standalone tool for downloading, processing, and extracting content from documentation websites. It transforms HTML documentation into structured JSON data that preserves the hierarchical organization of the content and enriches it with metadata.

## Key Features

- **Documentation Download**: Recursively downloads documentation pages with site structure preservation
- **HTML Processing**: Cleans HTML content by removing unwanted elements while preserving structure
- **Section Extraction**: Identifies headers and sections to create a hierarchical representation
- **Metadata Enrichment**: Adds token counts, hierarchy information, and other useful metadata
- **Special Element Detection**: Extracts code blocks, tables, and images as separate entities
- **Integration Support**: Provides interfaces for integration with other tools (like DuaLipa)

## Module Structure

The `fetch_docs` module is organized into the following components:

```
fetch_docs/
├── __init__.py
├── clean_html.py          # HTML cleaning utilities
├── download_site.py       # Documentation site download
├── extract_sections.py    # Section extraction and hierarchy
├── main.py                # Pipeline integration
├── page_downloader.py     # Page download utilities
├── docs/                  # Documentation and guides
├── db/                    # Database integration
├── embedding/             # Embedding utilities
├── llm/                   # LLM integration
├── tests/                 # Tests and verification
├── utils/                 # Helper utilities
└── scripts/               # Example scripts
```

## Integration with Other Tools

The `fetch_docs` module is designed to work both as a standalone tool and as a component that can be integrated with other tools in the agent_tools ecosystem. The primary integration point is currently with the `dualipa` module, where fetch_docs provides documentation extraction capabilities to complement dualipa's code extraction.

## Related Documentation

- [Task Overview](task.md): Description of the module's goals and requirements
- [TDD Strategy](tdd_strategy.md): Test-driven development approach for the module
- [Module Relationships](module_relationships.md): How fetch_docs relates to other modules
- [Integration Guide](integration_guide.md): How to integrate fetch_docs with other tools

## Usage Example

```python
from agent_tools.fetch_docs.download_site import download_site
from agent_tools.fetch_docs.main import process_directory
from pathlib import Path
import json

# Download a documentation site
output_dir = Path("./docs_output")
download_site("https://docs.example.com", str(output_dir))

# Process the downloaded site
processed_data = process_directory(output_dir)

# Save the processed data
with open("processed_docs.json", 'w') as f:
    json.dump(processed_data, f, indent=2)
```