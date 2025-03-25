"""
Markdown extraction module for DuaLipa.

This module provides functionality for extracting and processing markdown content,
including section parsing and hierarchy management.

Key Features:
1. Markdown parsing and token processing
2. Section hierarchy extraction
3. Code block extraction
4. Content organization and metadata

Dependencies:
- markdown-it-py: For markdown parsing
- loguru: For logging

Related Files:
- parser.py: Core markdown parsing
- hierarchy.py: Section hierarchy handling
- extractor.py: Content extraction
"""

from .markdown_extractor import extract_markdown_blocks

__all__ = ['extract_markdown_blocks']
