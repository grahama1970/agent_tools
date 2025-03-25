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
from .validation import validate_qa_output, validate_extraction
from .main import main, extract_repository, extract_file, extract_markdown, extract_html, analyze_hierarchy

# For backwards compatibility with old import paths
try:
    from .validation import extract_and_validate, generate_validation_report
    from .qa_formatter import format_for_qa_system
except ImportError:
    # These functions might not exist yet, so define placeholders
    def extract_and_validate(*args, **kwargs):
        """Placeholder for extract_and_validate."""
        raise NotImplementedError("extract_and_validate not implemented yet")
        
    def generate_validation_report(*args, **kwargs):
        """Placeholder for generate_validation_report."""
        raise NotImplementedError("generate_validation_report not implemented yet")
        
    def format_for_qa_system(*args, **kwargs):
        """Placeholder for format_for_qa_system."""
        raise NotImplementedError("format_for_qa_system not implemented yet")

__all__ = [
    'find_source_files',
    'extract_all_blocks',
    'analyze_hierarchies',
    'enrich_blocks_with_hierarchy',
    'create_qa_compatible_blocks',
    'create_qa_compatible_output',
    'validate_qa_output',
    'validate_extraction',
    'main',
    'extract_repository',
    'extract_file',
    'extract_markdown', 
    'extract_html',
    'analyze_hierarchy',
    'extract_and_validate',
    'generate_validation_report',
    'format_for_qa_system'
]