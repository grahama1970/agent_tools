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
6. State management for extraction processes
7. Memory management for AI operations

Dependencies:
- ast: For Python parsing
- tree-sitter: For JS/TS parsing
- markdown-it-py: For markdown parsing
- loguru: For logging
- sqlite3: For state management

Related Files:
- docs/extraction_format.md: Block format specification
- docs/module_relationships.md: Module organization
- docs/state_management.md: State management documentation
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

# State management and memory tools
from .test_state_manager import (
    TestStateManager,
    get_state_manager,
    what_am_i_doing,
    remember_context,
    add_docs,
    get_docs,
    verify_extraction_completeness
)

# Memory management for AI operations
from .memory import (
    remember,
    recall, 
    save_docs,
    find_docs,
    load_project_docs,
    get_verification_summary,
    log_error,
    suggest_recovery,
    think,
    remind_me,
    note,
    recall_thought
)

__all__ = [
    # Extraction functionality
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
    'format_output',
    
    # State management
    'TestStateManager',
    'get_state_manager',
    'what_am_i_doing',
    'remember_context',
    'add_docs',
    'get_docs',
    'verify_extraction_completeness',
    
    # Memory management
    'remember',
    'recall', 
    'save_docs',
    'find_docs',
    'load_project_docs',
    'get_verification_summary',
    'log_error',
    'suggest_recovery',
    'think',
    'remind_me',
    'note',
    'recall_thought'
] 