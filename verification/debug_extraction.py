#!/usr/bin/env python3
"""
Debug script to diagnose extraction issues.

This script analyzes the extraction process to understand why blocks
are counted but not written to output files.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from agent_tools.dualipa.code_extractor import extract_repository
    from agent_tools.dualipa.github_utils import download_github_repo
    
    # Import internal functions for debugging
    try:
        from agent_tools.dualipa.code_extractor import _extract_python_blocks, _extract_js_ts_blocks, _extract_markdown_blocks
        print("Successfully imported internal block extraction functions")
    except ImportError:
        print("Warning: Could not import internal block extraction functions")
    
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def debug_extraction():
    """Debug the extraction process."""
    print_section("Debugging Repository Extraction")
    
    # Use a small GitHub repository for testing
    repo_url = "https://github.com/psf/requests"
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        repo_dir = temp_path / "repo"
        output_dir = temp_path / "output"
        stats_dict = {}  # Dictionary to collect statistics
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created temporary directories:")
        print(f"  - Repository: {repo_dir}")
        print(f"  - Output: {output_dir}")
        
        # Download repository
        print("\nDownloading repository...")
        repo_path = download_github_repo(repo_url, str(repo_dir))
        print(f"Repository downloaded to: {repo_path}")
        
        # Find a sample Python file to extract blocks from
        print("\nFinding sample files...")
        python_file = None
        markdown_file = None
        js_file = None
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py") and not python_file:
                    python_file = os.path.join(root, file)
                elif file.endswith(".md") and not markdown_file:
                    markdown_file = os.path.join(root, file)
                elif file.endswith(".js") and not js_file:
                    js_file = os.path.join(root, file)
                
                if python_file and markdown_file and js_file:
                    break
            
            if python_file and markdown_file and js_file:
                break
        
        # Try extracting blocks directly from files
        if python_file:
            print(f"\nFound Python file: {python_file}")
            try:
                if '_extract_python_blocks' in globals():
                    with open(python_file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    blocks = _extract_python_blocks(content, str(python_file), stats_dict)
                    print(f"Extracted {len(blocks)} Python blocks directly from file")
                else:
                    print("Cannot extract Python blocks directly (function not available)")
            except Exception as e:
                print(f"Error extracting Python blocks: {e}")
        
        if markdown_file:
            print(f"\nFound Markdown file: {markdown_file}")
            try:
                if '_extract_markdown_blocks' in globals():
                    with open(markdown_file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    blocks = _extract_markdown_blocks(content, str(markdown_file), stats_dict)
                    print(f"Extracted {len(blocks)} Markdown blocks directly from file")
                else:
                    print("Cannot extract Markdown blocks directly (function not available)")
            except Exception as e:
                print(f"Error extracting Markdown blocks: {e}")
        
        # Debug repository extraction
        print("\nExtracting blocks from repository...")
        try:
            stats = extract_repository(
                source=repo_path,
                output_path=str(output_dir),
                max_files=10,  # Limit to 10 files for debugging
                extract_documentation=True,
                extract_code=True,
                extract_blocks=True
            )
            
            print(f"\nRepository extraction statistics:")
            print(f"  Files processed: {stats.get('total_files', 0)}")
            print(f"  Files with code: {stats.get('code_files', 0)}")
            print(f"  Files with documentation: {stats.get('documentation_files', 0)}")
            print(f"  Code blocks: {stats.get('code_blocks', 0)}")
            
            # Load and inspect the extraction_stats.json file
            stats_file = output_dir / "extraction_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                print(f"\nExtraction stats file contains:")
                print(f"  Total files: {stats_data.get('total_files', 0)}")
                print(f"  Code blocks: {stats_data.get('code_blocks', 0)}")
                
                # Check if it has blocks collection
                if "all_blocks" in stats_data:
                    print(f"  all_blocks collection: {len(stats_data['all_blocks'])} items")
            
            # Check output files
            blocks_file = output_dir / "blocks.json"
            code_file = output_dir / "code.json"
            docs_file = output_dir / "documentation.json"
            
            print("\nExamining output files:")
            for file_path in [blocks_file, code_file, docs_file]:
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"  {file_path.name}: {size / 1024:.2f} KB")
                    
                    # Read and check content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        data = json.loads(content) if content else []
                        
                    print(f"    Items: {len(data)}")
                    print(f"    Empty: {'Yes' if len(data) == 0 else 'No'}")
                else:
                    print(f"  {file_path.name}: File not found")
            
            # Look for other JSON files in the output directory
            print("\nLooking for other JSON files in output directory...")
            json_files = list(output_dir.glob("*.json"))
            for file_path in json_files:
                if file_path not in [blocks_file, code_file, docs_file, stats_file]:
                    size = file_path.stat().st_size
                    print(f"  {file_path.name}: {size / 1024:.2f} KB")
                    
                    # Read the file to see if it contains blocks
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            data = json.loads(content) if content else []
                            if isinstance(data, list) and len(data) > 0:
                                first_item = data[0]
                                if isinstance(first_item, dict) and "type" in first_item and "content" in first_item:
                                    print(f"    This file appears to contain code blocks! ({len(data)} items)")
                                    print(f"    First block type: {first_item.get('type')}")
                    except Exception as e:
                        print(f"    Error reading file: {e}")
            
            # Check if there are any files in subdirectories
            print("\nChecking for files in subdirectories...")
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob("*.*"))
                    if files:
                        print(f"  {subdir.name}/: {len(files)} file(s)")
                        for file in files[:5]:  # Show only first 5 files
                            size = file.stat().st_size
                            print(f"    {file.name}: {size / 1024:.2f} KB")
                            
                            # If it's a JSON file, check if it contains blocks
                            if file.name.endswith(".json"):
                                try:
                                    with open(file, "r", encoding="utf-8") as f:
                                        content = f.read()
                                        data = json.loads(content) if content else []
                                        if isinstance(data, list) and len(data) > 0:
                                            first_item = data[0]
                                            if isinstance(first_item, dict) and "type" in first_item and "content" in first_item:
                                                print(f"      This file appears to contain code blocks! ({len(data)} items)")
                                                print(f"      First block type: {first_item.get('type')}")
                                except Exception as e:
                                    pass
        
        except Exception as e:
            print(f"Error in repository extraction: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    debug_extraction() 