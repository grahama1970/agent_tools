"""
Configuration file for pytest.

This file sets up the Python path and other configurations for tests.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add src directory to the Python path
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Print debug information
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}") 