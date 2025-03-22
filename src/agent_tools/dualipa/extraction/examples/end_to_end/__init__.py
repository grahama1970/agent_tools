"""
End-to-End Extraction Example package for DuaLipa.

This package demonstrates the complete extraction pipeline from code files
to QA-compatible output format, split into modular components.

Modules:
- extraction_blocks.py: Finding and extracting code blocks
- hierarchy_analyzer.py: Hierarchy analysis functions
- qa_formatter.py: Converting to QA-compatible format
- validation.py: Validation functions
- main.py: Main entry point and orchestration
"""

from .extraction_blocks import find_source_files, extract_all_blocks
from .hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
from .qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
from .validation import validate_qa_output
from .main import main

__all__ = [
    'find_source_files',
    'extract_all_blocks',
    'analyze_hierarchies',
    'enrich_blocks_with_hierarchy',
    'create_qa_compatible_blocks',
    'create_qa_compatible_output',
    'validate_qa_output',
    'main'
]