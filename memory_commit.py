#!/usr/bin/env python3
"""
Memory commitment script for AST extraction.
This script commits new patterns to the memory system.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.agent_tools.dualipa.extraction.extraction_memory import (
    init_extraction_memory,
    save_extraction_knowledge
)

def main():
    """Main function to commit memory."""
    # Initialize memory
    init_extraction_memory("extraction_memory.db")
    
    # Save new extraction knowledge
    save_extraction_knowledge(
        "ast_memory_integration",
        "# AST Memory Integration\n\n"
        "The AST extractor has been successfully integrated with the memory system.\n\n"
        "## Key Features\n\n"
        "- Extraction of nested classes\n"
        "- Extraction of inheritance hierarchies\n"
        "- Detection of decorator patterns\n"
        "- Persistent memory across extractions\n\n"
        "This integration improves extraction quality over time as the system learns from previous extractions.",
        summary="Tree-sitter AST extractor with memory integration for improved code structure extraction",
        tags=["ast", "memory", "tree-sitter", "extraction"]
    )
    
    print("Memory commitment complete.")

if __name__ == "__main__":
    main()