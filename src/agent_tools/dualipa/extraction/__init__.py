"""
DuaLipa extraction module.

This module provides functionality for extracting content from repositories,
including code blocks, markdown sections, and repository analysis.

Key Features:
1. Code block extraction
2. Markdown content extraction
3. Repository analysis
4. Statistics tracking
5. Output formatting

Dependencies:
- ast: For Python parsing
- tree-sitter: For JS/TS parsing
- markdown-it-py: For markdown parsing
- loguru: For logging

Related Files:
- docs/extraction_format.md: Block format specification
- docs/module_relationships.md: Module organization
"""

from .extractors.code import (
    extract_code_blocks,
    extract_python_blocks,
    extract_js_ts_blocks,
    extract_generic_blocks,
    validate_block,
    verify_block,
    analyze_code_hierarchy
)

from .extractors.markdown import extract_markdown_blocks

from .extractors.github import (
    clone_repository,
    verify_repo_structure,
    extract_repository
)

from .extractors.utils import (
    detect_language,
    get_language_info,
    is_supported_language,
    init_stats,
    update_stats,
    merge_stats,
    format_stats,
    format_output_as_json,
    format_output_as_md,
    format_output_as_html,
    format_output
)

__all__ = [
    'extract_code_blocks',
    'extract_python_blocks',
    'extract_js_ts_blocks',
    'extract_generic_blocks',
    'validate_block',
    'verify_block',
    'analyze_code_hierarchy',
    'extract_markdown_blocks',
    'clone_repository',
    'verify_repo_structure',
    'extract_repository',
    'detect_language',
    'get_language_info',
    'is_supported_language',
    'init_stats',
    'update_stats',
    'merge_stats',
    'format_stats',
    'format_output_as_json',
    'format_output_as_md',
    'format_output_as_html',
    'format_output'
] 