#!/usr/bin/env python
"""
Simple script to extract code blocks from a single file.
This demonstrates the basic usage of the code_extractor module with individual files.

Example usage:
    python extract_single_file.py ../../code_extractor.py ./output
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from agent_tools.dualipa.code_extractor import extract_repository
except ImportError:
    # Fall back to direct import for standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from code_extractor import extract_repository

def main():
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_single_file.py <source_file> [output_dir]")
        return 1
    
    source_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if the file exists
    if not os.path.isfile(source_file):
        print(f"Error: File not found: {source_file}")
        return 1
    
    print(f"Extracting code from: {source_file}")
    print(f"Output directory: {output_dir or 'Default (data/files)'}")
    
    # Extract the file
    stats = extract_repository(
        source=source_file,
        output_path=output_dir,
        extract_documentation=True,
        extract_code=True,
        extract_blocks=True
    )
    
    # Print the results
    print("\nExtraction completed!")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Code files: {stats['code_files']}")
    print(f"Documentation files: {stats['documentation_files']}")
    print(f"Code blocks extracted: {stats['code_blocks']}")
    
    # Save the stats to a JSON file in the output directory
    if output_dir:
        stats_file = Path(output_dir) / "extraction_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {stats_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 