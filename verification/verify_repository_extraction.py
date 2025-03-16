#!/usr/bin/env python3
"""
Verify repository extraction functionality.

This script tests the repository extraction functionality in the DuaLipa library
by extracting code from specified repositories and reporting statistics.
"""

import os
import sys
import tempfile
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Check if required modules are available
try:
    from tqdm import tqdm
    from rich.console import Console
    from rich.table import Table
    print("Successfully imported required external modules")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required modules: pip install tqdm rich")
    sys.exit(1)

# Import the required modules from DuaLipa
try:
    from agent_tools.dualipa.code_extractor import extract_repository
    from agent_tools.dualipa.github_utils import is_github_url, download_github_repo
    print("Successfully imported DuaLipa modules")
except ImportError as e:
    print(f"Error importing DuaLipa modules: {e}")
    sys.exit(1)

# Initialize console for rich output
console = Console()

# Define repositories to test
REPOSITORIES = [
    {
        "name": "Requests",
        "url": "https://github.com/psf/requests",
        "description": "A popular HTTP library for Python",
        "max_files": 500
    },
    {
        "name": "Flask",
        "url": "https://github.com/pallets/flask",
        "description": "A micro web framework for Python",
        "max_files": 500
    },
    {
        "name": "OpenWebUI",
        "url": "https://github.com/open-webui/open-webui",
        "description": "An open-source web UI framework",
        "max_files": 1000
    },
    {
        "name": "ArangoDB",
        "url": "https://github.com/arangodb/arangodb",
        "description": "Multi-model database",
        "max_files": 1000
    }
]

def print_header(text, underline='='):
    """Print a header with underline."""
    console.print(f"\n{text}", style="bold cyan")
    console.print(underline * len(text), style="cyan")

def find_repo_config(repo_name):
    """Find repository configuration by name."""
    for repo in REPOSITORIES:
        if repo["name"].lower() == repo_name.lower():
            return repo
    return None

def test_repository_extraction(repo_config):
    """Test repository extraction for a single repository."""
    print_header(f"Testing extraction for {repo_config['name']}", "-")
    console.print(f"[italic]{repo_config['description']}[/italic]")
    console.print(f"Repository URL: [link={repo_config['url']}]{repo_config['url']}[/link]")
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        repo_dir = temp_path / "repo"
        output_dir = temp_path / "output"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get repository name and owner from URL
            if is_github_url(repo_config["url"]):
                # Download the repository
                console.print("\nDownloading repository...", style="bold yellow")
                repo_path = download_github_repo(repo_config["url"], str(repo_dir))
                console.print(f"Repository downloaded to: {repo_path}", style="green")
            else:
                console.print(f"Invalid GitHub URL: {repo_config['url']}", style="bold red")
                return False
            
            # Set up extraction parameters
            max_files = repo_config.get("max_files", 500)
            include_patterns = repo_config.get("include_patterns", ["*.*"])
            exclude_patterns = repo_config.get("exclude_patterns", [])
            
            # Extract from the repository
            console.print("\nExtracting code from repository...", style="bold yellow")
            start_time = time.time()
            
            statistics = extract_repository(
                source=repo_path,
                output_path=str(output_dir),
                max_files=max_files,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                extract_documentation=True,
                extract_code=True,
                extract_blocks=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Display extraction statistics
            print_header("Extraction Statistics", "-")
            
            # Create a rich table for statistics
            table = Table(title=f"Extraction Results for {repo_config['name']}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            # Add statistics to the table
            table.add_row("Files Processed", str(statistics.get("files_processed", 0)))
            table.add_row("Files with Code", str(statistics.get("files_with_code", 0)))
            table.add_row("Files with Documentation", str(statistics.get("files_with_documentation", 0)))
            table.add_row("Code Blocks Extracted", str(statistics.get("code_blocks", 0)))
            table.add_row("Processing Time", f"{duration:.2f} seconds")
            
            # Calculate success rate
            if statistics.get("files_processed", 0) > 0:
                success_rate = (statistics.get("files_processed", 0) - statistics.get("errors", 0)) / statistics.get("files_processed", 0) * 100
                table.add_row("Success Rate", f"{success_rate:.2f}%")
            
            console.print(table)
            
            # Check for output files
            blocks_file = output_dir / "blocks.json"
            code_file = output_dir / "code.json"
            docs_file = output_dir / "documentation.json"
            
            files_exist = blocks_file.exists() and code_file.exists() and docs_file.exists()
            
            if files_exist:
                console.print("\nOutput files created successfully:", style="bold green")
                console.print(f"  - {blocks_file.name}: {blocks_file.stat().st_size / 1024:.2f} KB")
                console.print(f"  - {code_file.name}: {code_file.stat().st_size / 1024:.2f} KB")
                console.print(f"  - {docs_file.name}: {docs_file.stat().st_size / 1024:.2f} KB")
            else:
                console.print("\nSome output files are missing!", style="bold red")
                console.print(f"  - blocks.json: {'✅' if blocks_file.exists() else '❌'}")
                console.print(f"  - code.json: {'✅' if code_file.exists() else '❌'}")
                console.print(f"  - documentation.json: {'✅' if docs_file.exists() else '❌'}")
            
            return files_exist and statistics.get("code_blocks", 0) > 0
            
        except Exception as e:
            console.print(f"\nError during repository extraction: {str(e)}", style="bold red")
            return False

def main():
    """Run repository extraction verification."""
    print_header("Repository Extraction Verification")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify repository extraction")
    parser.add_argument("--repo", help="Specific repository to test (by name)")
    parser.add_argument("--all", action="store_true", help="Test all repositories")
    args = parser.parse_args()
    
    # Check if required modules are available
    try:
        import tqdm
        import rich
    except ImportError as e:
        console.print(f"Error: Required modules not found: {e}", style="bold red")
        console.print("Please install required modules: pip install tqdm rich", style="yellow")
        return 1
    
    # Test specific repository or all
    results = {}
    
    if args.repo:
        repo_config = find_repo_config(args.repo)
        if repo_config:
            success = test_repository_extraction(repo_config)
            results[repo_config["name"]] = success
        else:
            console.print(f"Repository '{args.repo}' not found in the configuration.", style="bold red")
            console.print("Available repositories:", style="yellow")
            for repo in REPOSITORIES:
                console.print(f"  - {repo['name']}: {repo['description']}")
            return 1
    elif args.all:
        for repo_config in tqdm(REPOSITORIES, desc="Testing repositories"):
            success = test_repository_extraction(repo_config)
            results[repo_config["name"]] = success
    else:
        # Default to the first repository if none specified
        repo_config = REPOSITORIES[0]
        success = test_repository_extraction(repo_config)
        results[repo_config["name"]] = success
    
    # Print summary
    print_header("Verification Summary", "=")
    
    summary_table = Table(title="Repository Extraction Results")
    summary_table.add_column("Repository", style="cyan")
    summary_table.add_column("Result", style="green")
    
    all_success = True
    for repo_name, success in results.items():
        summary_table.add_row(
            repo_name, 
            "✅ Passed" if success else "❌ Failed",
            style=None if success else "red"
        )
        all_success = all_success and success
    
    console.print(summary_table)
    console.print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}", 
                  style="bold green" if all_success else "bold red")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 