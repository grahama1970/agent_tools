"""
Configuration for fetch_docs integration.

This module contains configuration options for the fetch_docs integration.
"""

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
