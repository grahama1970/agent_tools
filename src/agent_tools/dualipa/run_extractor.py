#!/usr/bin/env python
"""
Simple script to run the code extractor directly.

This script handles import path setup to make code_extractor.py work
even when run outside of the package context.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    from agent_tools.dualipa.code_extractor import main, demo_code_extractor
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_code_extractor()
    else:
        main() 