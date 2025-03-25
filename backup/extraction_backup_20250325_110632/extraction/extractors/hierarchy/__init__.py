"""Code hierarchy analysis for DuaLipa.

This package handles code structure analysis and hierarchy extraction,
including class relationships, function dependencies, and module organization.

Key Features:
1. Class hierarchy analysis
2. Function dependency tracking
3. Module relationship mapping
4. Import graph generation

Submodules:
- python: Python-specific code hierarchy analysis
- js_ts: JavaScript/TypeScript-specific code hierarchy analysis
- generic: Generic language code hierarchy analysis
- core: Core hierarchy functionality and common utilities
"""

from .core import analyze_code_hierarchy, build_code_hierarchy
from .python.parser import analyze_python_hierarchy
from .js_ts.parser import analyze_js_ts_hierarchy
from .generic.parser import analyze_generic_hierarchy
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict

__all__ = [
    'analyze_code_hierarchy', 
    'build_code_hierarchy',
    'analyze_python_hierarchy',
    'analyze_js_ts_hierarchy',
    'analyze_generic_hierarchy',
    'initialize_stats_dict'
]