"""
Documentation extraction module for DuaLipa.

This module provides functionality for extracting content from documentation sources,
including HTML pages downloaded from documentation sites like ReadTheDocs and ArangoDB.

Key Features:
1. HTML cleaning and conversion
2. Documentation link detection
3. Section extraction and hierarchy preservation 
4. Integration with DuaLipa extraction pipeline

Dependencies:
- BeautifulSoup: For HTML parsing
- markdownify: For HTML to markdown conversion
- loguru: For logging
"""

from .docs_extractor import (
    extract_documentation,
    detect_doc_links,
    download_docs,
    process_docs,
    convert_to_dualipa_format,
    integrate_docs_with_extraction
)

__all__ = [
    'extract_documentation',
    'detect_doc_links',
    'download_docs',
    'process_docs',
    'convert_to_dualipa_format',
    'integrate_docs_with_extraction'
]