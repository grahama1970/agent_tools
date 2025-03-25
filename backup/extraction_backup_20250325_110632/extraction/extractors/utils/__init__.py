"""
Utility modules for DuaLipa extraction.

This module provides common utilities used across the extraction process,
including language detection, statistics tracking, and error handling.

Key Features:
1. Language detection and mapping
2. Block format validation
3. Code block verification
4. Statistics tracking
5. Output formatting
6. Common file operations

Dependencies:
- loguru: For logging

Related Files:
- language_utils.py: Language detection and mapping
- validation_utils.py: Block format validation
- verification_utils.py: Code block verification
- stats_utils.py: Statistics tracking
- output_formatter.py: Output formatting utilities
"""

from .language_utils import (
    detect_language,
    get_language_info,
    is_supported_language,
    get_comment_pattern,
    get_block_comment_patterns
)

from .validation_utils import validate_block_format, validate_metadata
from .verification_utils import verify_code_block, verify_block_syntax
from .stats_utils import (
    init_stats,
    update_stats,
    merge_stats,
    format_stats
)

from .block_metadata import (
    initialize_stats_dict,
    verify_block_metadata,
    create_block_metadata,
)

from .output_formatter import (
    format_output_as_json,
    format_output_as_md,
    format_output_as_html,
    format_output
)

__all__ = [
    'detect_language',
    'get_language_info',
    'is_supported_language',
    'get_comment_pattern',
    'get_block_comment_patterns',
    'validate_block_format',
    'validate_metadata',
    'verify_code_block',
    'verify_block_syntax',
    'init_stats',
    'update_stats',
    'merge_stats',
    'format_stats',
    'initialize_stats_dict',
    'verify_block_metadata',
    'create_block_metadata',
    'format_output_as_json',
    'format_output_as_md',
    'format_output_as_html',
    'format_output'
]
