#!/usr/bin/env python3
"""
Script to move extracted files to the dualipa data directory.

This script uses the code_extractor.py module to move files
from the extracted_repo directory to the dualipa/data/files directory.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from src.agent_tools.dualipa.code_extractor import move_extracted_files

def main():
    """
    Move extracted files to the dualipa data directory.
    
    This function moves the files from the extracted_repo directory
    to the dualipa/data/files directory, organizing them for further
    processing by the dualipa pipeline.
    """
    # Source directory
    source_dir = os.path.join(project_dir, "extracted_repo")
    
    # Target directory (will default to dualipa/data/files if not specified)
    target_dir = None  # Let the function use the default
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return 1
    
    print(f"Moving files from {source_dir} to dualipa data directory...")
    
    # Move the files
    num_files = move_extracted_files(source_dir, target_dir)
    
    print(f"Done! {num_files} files were moved successfully.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 