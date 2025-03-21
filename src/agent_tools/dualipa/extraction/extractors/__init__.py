"""
Extractors module for DuaLipa.

This module contains the core extraction functionality for different
file types and languages, organized into focused submodules.

Submodules:
- code/: Language-specific code extraction
- markdown/: Markdown parsing and extraction
- github/: Repository operations
- utils/: Common utilities

Each submodule is self-contained and focused on a specific task,
with clear dependencies and interfaces.
"""

from . import code
from . import markdown
from . import github
from . import utils

__all__ = [
    'code',
    'markdown',
    'github',
    'utils'
]
