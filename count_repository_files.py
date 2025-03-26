#!/usr/bin/env python3
"""
Repository File Analysis

This script scans a repository and provides counts of files by extension,
which should be run BEFORE attempting any extraction to understand the data.
"""

import os
import sys
from collections import Counter
import json
from pathlib import Path
import argparse

def count_files_by_extension(repo_path, exclude_dirs=None):
    """
    Count files by extension in a repository.
    
    Args:
        repo_path: Path to the repository
        exclude_dirs: List of directories to exclude
        
    Returns:
        Counter object with counts by extension
    """
    if exclude_dirs is None:
        exclude_dirs = ['3rdParty', 'node_modules', '.git']
    
    extension_counts = Counter()
    total_files = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            total_files += 1
            _, ext = os.path.splitext(file)
            if ext:
                extension_counts[ext] += 1
            else:
                extension_counts['no extension'] += 1
    
    return extension_counts, total_files

def analyze_repository(repo_path):
    """
    Analyze a repository and print file statistics.
    
    Args:
        repo_path: Path to the repository
    """
    print(f"Analyzing repository: {repo_path}")
    
    # Count files by extension
    extension_counts, total_files = count_files_by_extension(repo_path)
    
    # Calculate percentages and sort by count
    percentages = {ext: (count / total_files) * 100 for ext, count in extension_counts.items()}
    sorted_extensions = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\nRepository contains {total_files} files")
    print("\nFiles by extension:")
    print(f"{'Extension':<15} {'Count':<8} {'Percentage':<10}")
    print("-" * 35)
    
    for ext, count in sorted_extensions:
        print(f"{ext:<15} {count:<8} {percentages[ext]:.2f}%")
    
    # Specific language counts
    print("\nSpecific language files:")
    print(f"Python files (.py): {extension_counts.get('.py', 0)}")
    print(f"C++ files (.cpp, .h, .cc): {extension_counts.get('.cpp', 0) + extension_counts.get('.h', 0) + extension_counts.get('.cc', 0)}")
    print(f"JavaScript files (.js): {extension_counts.get('.js', 0)}")
    print(f"TypeScript files (.ts): {extension_counts.get('.ts', 0)}")
    print(f"Markdown files (.md): {extension_counts.get('.md', 0)}")
    
    # Save results to JSON
    results = {
        "total_files": total_files,
        "extension_counts": dict(extension_counts),
        "percentages": {ext: round(pct, 2) for ext, pct in percentages.items()}
    }
    
    output_file = os.path.join(os.path.dirname(repo_path), "repository_analysis.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze repository file types")
    parser.add_argument("--repo-path", type=str, default="/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb",
                        help="Path to repository")
    args = parser.parse_args()
    
    analyze_repository(args.repo_path)
    
if __name__ == "__main__":
    main()