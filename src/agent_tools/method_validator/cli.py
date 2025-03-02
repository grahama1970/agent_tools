#!/usr/bin/env python3
"""
Method Validator wrapper script.
"""

import os
import sys

# Add the agent_tools package to Python path
agent_tools_path = "/Users/robert/Documents/dev/workspace/agent_tools/src"
if agent_tools_path not in sys.path:
    sys.path.insert(0, agent_tools_path)

from agent_tools.method_validator.method_validator import main

if __name__ == "__main__":
    main()
