#!/usr/bin/env python
"""
Script to extract code blocks from markdown files.
This demonstrates using the code_extractor module for processing documentation files.

Example usage:
    python extract_markdown_blocks.py ../task.md ./output
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from agent_tools.dualipa.code_extractor import extract_repository
    from agent_tools.dualipa.markdown_parser import extract_code_blocks
except ImportError:
    # Fall back to direct import for standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from code_extractor import extract_repository
    from markdown_parser import extract_code_blocks

def main():
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_markdown_blocks.py <source_md_file> [output_dir]")
        return 1
    
    source_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if the file exists
    if not os.path.isfile(source_file):
        print(f"Error: File not found: {source_file}")
        return 1
    
    # Check if it's a markdown file
    if not source_file.lower().endswith(('.md', '.markdown')):
        print(f"Warning: File does not appear to be a markdown file: {source_file}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    print(f"Extracting from markdown file: {source_file}")
    print(f"Output directory: {output_dir or 'Default (data/files)'}")
    
    # Method 1: Use extract_repository for full processing
    print("\nMethod 1: Using extract_repository:")
    stats = extract_repository(
        source=source_file,
        output_path=output_dir,
        extract_documentation=True,
        extract_code=False,  # We only want to process the markdown
        extract_blocks=True
    )
    
    # Print the results of Method 1
    print("\nExtraction completed!")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Documentation files: {stats['documentation_files']}")
    print(f"Documentation blocks extracted: {stats['doc_blocks']}")
    
    # Method 2: Use markdown_parser directly for just the code blocks
    print("\nMethod 2: Using markdown_parser directly for code blocks:")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        code_blocks = extract_code_blocks(content)
        
        # Create an output directory for the extracted code blocks
        if output_dir:
            blocks_dir = Path(output_dir) / "direct_blocks"
            blocks_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each code block to a file
            for i, block in enumerate(code_blocks):
                language = block.get('language', 'txt')
                code = block.get('code', '')
                
                block_file = blocks_dir / f"block_{i+1}.{language}"
                with open(block_file, 'w', encoding='utf-8') as f:
                    f.write(code)
            
            print(f"Extracted {len(code_blocks)} code blocks directly")
            print(f"Saved to: {blocks_dir}")
        else:
            # Just print the code blocks if no output directory
            for i, block in enumerate(code_blocks):
                language = block.get('language', 'txt')
                code = block.get('code', '')
                print(f"\nBlock {i+1} ({language}):")
                print("```")
                print(code[:200] + "..." if len(code) > 200 else code)
                print("```")
    
    except Exception as e:
        print(f"Error extracting code blocks directly: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 