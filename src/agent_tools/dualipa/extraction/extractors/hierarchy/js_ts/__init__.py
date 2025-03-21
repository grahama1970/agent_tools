"""JavaScript/TypeScript-specific code hierarchy analysis.

This module handles JS/TS code structure analysis using tree-sitter.
"""

from .parser import analyze_js_ts_hierarchy

__all__ = ['analyze_js_ts_hierarchy']
