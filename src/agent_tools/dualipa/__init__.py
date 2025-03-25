"""
DuaLipa module for code extraction and analysis.
"""

__version__ = "0.1.0"

# Re-export code_extractor functions from their new location
from agent_tools.dualipa.extraction.extractors.code.code_extractor import (
    extract_code_blocks,
    extract_python_blocks,
    extract_js_ts_blocks,
    extract_generic_blocks,
    validate_block,
    verify_block,
    extract_repository,
    _extract_python_blocks,
    _extract_js_ts_blocks,
    _extract_generic_blocks
)

# For statistics tracking
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import (
    initialize_stats_dict,
    init_stats,
    update_stats,
    merge_stats,
    format_stats
)

# For GitHub operations
from agent_tools.dualipa.github_utils import (
    parse_github_url,
    discover_files,
    clone_repository,
    verify_repo_structure
)

# For code hierarchy
try:
    from agent_tools.dualipa.extraction.extractors.code.hierarchy import (
        analyze_code_hierarchy
    )
except ImportError:
    # Define a stub if import fails
    def analyze_code_hierarchy(*args, **kwargs):
        """Stub for analyze_code_hierarchy."""
        return {"error": "Hierarchy module not available"}

# Make these available at the package level
__all__ = [
    # Code extraction
    'extract_code_blocks',
    'extract_python_blocks',
    'extract_js_ts_blocks',
    'extract_generic_blocks',
    'validate_block',
    'verify_block',
    'extract_repository',
    '_extract_python_blocks',
    '_extract_js_ts_blocks',
    '_extract_generic_blocks',
    
    # Statistics
    'initialize_stats_dict',
    'init_stats',
    'update_stats',
    'merge_stats',
    'format_stats',
    
    # GitHub
    'parse_github_url',
    'discover_files',
    'clone_repository',
    'verify_repo_structure',
    
    # Hierarchy
    'analyze_code_hierarchy'
]
