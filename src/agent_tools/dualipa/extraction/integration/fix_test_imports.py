#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix import paths in test files after reorganization.

This script updates import paths in test files to match the new 
directory structure after reorganization.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Import mappings - old import path to new import path
IMPORT_MAPPINGS = {
    # Code extraction imports
    "from agent_tools.dualipa.code_extractor": "from agent_tools.dualipa.extraction.extractors.code.code_extractor",
    "import agent_tools.dualipa.code_extractor": "import agent_tools.dualipa.extraction.extractors.code.code_extractor",
    
    # GitHub utils imports
    "from agent_tools.dualipa.github_utils": "from agent_tools.dualipa.extraction.extractors.github.repo_utils",
    "import agent_tools.dualipa.github_utils": "import agent_tools.dualipa.extraction.extractors.github.repo_utils",
    
    # Extraction repo imports
    "from agent_tools.dualipa.extract_repo": "from agent_tools.dualipa.extraction.extractors.github.repo_utils",
    "import agent_tools.dualipa.extract_repo": "import agent_tools.dualipa.extraction.extractors.github.repo_utils",
    
    # Markdown parser imports
    "from agent_tools.dualipa.markdown_parser": "from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor",
    "import agent_tools.dualipa.markdown_parser": "import agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor",
    
    # Old import for code/hierarchy.py
    "from agent_tools.dualipa.code_hierarchy": "from agent_tools.dualipa.extraction.extractors.code.hierarchy",
    "import agent_tools.dualipa.code_hierarchy": "import agent_tools.dualipa.extraction.extractors.code.hierarchy",
    
    # Old imports for end_to_end
    "import end_to_end_extraction": "import agent_tools.dualipa.extraction.examples.end_to_end.main as end_to_end_extraction",
    "from end_to_end_extraction": "from agent_tools.dualipa.extraction.examples.end_to_end.main",
}

def fix_imports_in_file(file_path: str) -> int:
    """
    Fix import paths in a single file.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        Number of imports replaced
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Track replacements
        replacements = 0
        
        # Apply replacements
        for old_import, new_import in IMPORT_MAPPINGS.items():
            pattern = re.escape(old_import) + r"\b"
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, new_import, content)
                replacements += len(matches)
                
        # Write updated content
        if replacements > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return replacements
        
    except Exception as e:
        print(f"Error fixing imports in {file_path}: {e}")
        return 0

def fix_imports_in_directory(directory: str) -> Dict[str, int]:
    """
    Fix import paths in all Python files in a directory.
    
    Args:
        directory: Directory to search for Python files
        
    Returns:
        Dictionary mapping file paths to number of replacements
    """
    results = {}
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    replacements = fix_imports_in_file(file_path)
                    if replacements > 0:
                        results[file_path] = replacements
                        
        return results
        
    except Exception as e:
        print(f"Error fixing imports in directory {directory}: {e}")
        return results

def main():
    """Main entry point."""
    # Determine base directory
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    tests_dir = project_root / "tests"
    
    print(f"Fixing imports in {tests_dir}...")
    results = fix_imports_in_directory(str(tests_dir))
    
    # Print summary
    total_files = len(results)
    total_replacements = sum(results.values())
    print(f"\nFixed {total_replacements} imports across {total_files} files.")
    
    # Print details for modified files
    if results:
        print("\nModified files:")
        for file_path, replacements in results.items():
            rel_path = os.path.relpath(file_path, start=str(project_root))
            print(f"  {rel_path}: {replacements} replacements")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())