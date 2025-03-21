# src/agent_tools/dualipa/__init__.py
__version__ = "0.1.0"  # Or whatever version you want

# Re-export code_extractor functions from their new location
from agent_tools.dualipa.extraction.extractors.code.code_extractor import (
    extract_code_blocks,
    extract_python_blocks,
    extract_js_ts_blocks,
    extract_generic_blocks,
    validate_block,
    verify_block
)

from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks
from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks

# Make these available at the package level
__all__ = [
    'extract_code_blocks',
    'extract_python_blocks',
    'extract_js_ts_blocks',
    'extract_generic_blocks',
    'validate_block',
    'verify_block'
]
