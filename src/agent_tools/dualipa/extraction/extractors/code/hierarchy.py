"""
Code hierarchy analysis for DuaLipa.

This module is a shim for backward compatibility, redirecting to the new
hierarchy analysis modules in the extraction.extractors.hierarchy package.

Dependencies:
- agent_tools.dualipa.extraction.extractors.hierarchy: New hierarchy modules

Documentation Links:
- https://docs.python.org/3/library/typing.html
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import directly from the specific submodules for backward compatibility
from agent_tools.dualipa.extraction.extractors.hierarchy.core import analyze_code_hierarchy, build_code_hierarchy
from agent_tools.dualipa.extraction.extractors.hierarchy.python.parser import analyze_python_hierarchy
from agent_tools.dualipa.extraction.extractors.hierarchy.js_ts.parser import analyze_js_ts_hierarchy
from agent_tools.dualipa.extraction.extractors.hierarchy.generic.parser import analyze_generic_hierarchy
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict

# Import stats_utils for init_stats
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats

# Re-export all the functions for backward compatibility
__all__ = [
    'analyze_code_hierarchy', 
    'build_code_hierarchy', 
    'initialize_stats_dict',
    'analyze_python_hierarchy',
    'analyze_js_ts_hierarchy',
    'analyze_generic_hierarchy',
    'init_stats'
]