#!/usr/bin/env python3
"""
Extraction Workflow Checklist

This script implements a verification checklist for file extraction workflows.
It ensures proper analysis is done before extraction, verifies coverage,
and validates output format.

Usage:
    python extraction_checklist.py --repo-path /path/to/repo
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path
import time

def log_step(step_name):
    """Log a step in the checklist with timing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 50}")
            print(f"STEP: {step_name}")
            print(f"{'=' * 50}")
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"✓ Completed in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator

@log_step("Repository Analysis")
def analyze_repository(repo_path, exclude_dirs=None):
    """
    Analyze a repository to understand its structure before extraction.
    
    Args:
        repo_path: Path to the repository
        exclude_dirs: Directories to exclude
        
    Returns:
        Dictionary with file counts and statistics
    """
    if exclude_dirs is None:
        exclude_dirs = ['3rdParty', 'node_modules', '.git']
    
    print(f"Analyzing repository: {repo_path}")
    
    # Initialize counters
    file_counts = Counter()  # By extension
    language_counts = {}     # Mapped to common languages
    total_files = 0
    
    # Extension to language mapping
    ext_to_lang = {
        '.py': 'python',
        '.cpp': 'cpp', '.h': 'cpp', '.hpp': 'cpp', '.cc': 'cpp',
        '.js': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.md': 'markdown', '.markdown': 'markdown',
        '.html': 'html', '.htm': 'html',
        '.css': 'css', '.scss': 'css',
        '.json': 'json',
        '.sh': 'bash', '.bash': 'bash'
    }
    
    # Count files
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        rel_path = os.path.relpath(root, repo_path)
        if any(excl in rel_path.split(os.sep) for excl in exclude_dirs):
            continue
            
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            file_counts[ext] += 1
            
            # Map to language
            if ext in ext_to_lang:
                lang = ext_to_lang[ext]
                language_counts[lang] = language_counts.get(lang, 0) + 1
    
    # Calculate percentages
    percentages = {ext: (count / total_files) * 100 for ext, count in file_counts.items()}
    
    # Print results
    print(f"\nRepository contains {total_files} files")
    
    print("\nLanguage distribution:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {lang}: {count} files ({count/total_files*100:.1f}%)")
    
    print("\nTop 10 file extensions:")
    for ext, count in file_counts.most_common(10):
        print(f"- {ext or 'no extension'}: {count} files ({percentages[ext]:.1f}%)")
    
    # Prepare results
    results = {
        "total_files": total_files,
        "file_counts": dict(file_counts),
        "language_counts": language_counts,
        "percentages": {ext: round(pct, 2) for ext, pct in percentages.items()}
    }
    
    # Save results
    analysis_file = os.path.join(os.path.dirname(repo_path), "repository_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis saved to {analysis_file}")
    return results

@log_step("Extraction Planning")
def plan_extraction(analysis, target_extensions=None):
    """
    Plan the extraction based on repository analysis.
    
    Args:
        analysis: Repository analysis results
        target_extensions: Extensions to extract (default: all)
        
    Returns:
        Extraction plan
    """
    if target_extensions is None:
        # Default to common code and documentation files
        target_extensions = ['.py', '.js', '.ts', '.cpp', '.h', '.md']
    
    # Calculate total files to extract
    total_target_files = sum(analysis["file_counts"].get(ext, 0) for ext in target_extensions)
    
    # Create extraction plan
    plan = {
        "target_extensions": target_extensions,
        "total_target_files": total_target_files,
        "extension_counts": {ext: analysis["file_counts"].get(ext, 0) for ext in target_extensions}
    }
    
    # Print plan
    print("Extraction Plan:")
    print(f"- Total files to extract: {total_target_files}")
    
    for ext in target_extensions:
        count = analysis["file_counts"].get(ext, 0)
        print(f"- {ext}: {count} files")
    
    # Check if any target extension has 0 files
    zero_exts = [ext for ext in target_extensions if analysis["file_counts"].get(ext, 0) == 0]
    if zero_exts:
        print(f"\nWARNING: No files found with extensions: {', '.join(zero_exts)}")
    
    return plan

@log_step("Extraction Simulation")
def simulate_extraction(repo_path, plan):
    """
    Simulate extraction to verify file discovery.
    
    Args:
        repo_path: Path to repository
        plan: Extraction plan
        
    Returns:
        List of files that would be extracted
    """
    files_to_extract = []
    
    for ext in plan["target_extensions"]:
        ext_files = []
        for root, _, files in os.walk(repo_path):
            if "3rdParty" in root or "node_modules" in root or ".git" in root:
                continue
                
            for file in files:
                if file.endswith(ext):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_path)
                    ext_files.append(rel_path)
        
        # Check if we found expected number
        expected = plan["extension_counts"][ext]
        actual = len(ext_files)
        
        if actual == expected:
            print(f"✓ Found {actual}/{expected} files with extension {ext}")
        else:
            print(f"⚠ Found {actual}/{expected} files with extension {ext}")
            
        files_to_extract.extend(ext_files)
        
        # Print sample of files for each extension
        if ext_files:
            print(f"\nSample files with extension {ext}:")
            for sample_file in ext_files[:3]:
                print(f"- {sample_file}")
            if len(ext_files) > 3:
                print(f"- ... and {len(ext_files)-3} more")
    
    # Check for specific important files
    important_files = [
        "tests/js/common/test-data/search/docs/generate_ii_sa_dataset.py",
        "utils/gantt.py",
        "scripts/toolbox/modules/HotBackup.py"
    ]
    
    print("\nChecking for important files:")
    for imp_file in important_files:
        if imp_file in files_to_extract:
            print(f"✓ Found important file: {imp_file}")
        else:
            print(f"❌ MISSING important file: {imp_file}")
            # Try to find if it exists but wasn't discovered
            full_path = os.path.join(repo_path, imp_file)
            if os.path.exists(full_path):
                print(f"  File exists but wasn't included in extraction!")
    
    print(f"\nTotal files to extract: {len(files_to_extract)}")
    return files_to_extract

@log_step("Output Format Validation")
def validate_output_format(output_schema):
    """
    Validate the output format schema.
    
    Args:
        output_schema: Dictionary describing output format
        
    Returns:
        True if valid, False otherwise
    """
    # Check required top-level fields
    required_fields = ["sections", "section_relationships", "extraction_metadata"]
    
    missing = [field for field in required_fields if field not in output_schema]
    if missing:
        print(f"❌ Output schema missing required fields: {missing}")
        return False
    
    # Check section required fields
    if "section_fields" not in output_schema:
        print("❌ Output schema missing section_fields definition")
        return False
        
    section_required = ["uuid", "id", "type", "name", "content", "language"]
    missing = [field for field in section_required if field not in output_schema["section_fields"]]
    if missing:
        print(f"❌ Section schema missing required fields: {missing}")
        return False
    
    print("✓ Output format validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Extraction Workflow Checklist")
    parser.add_argument("--repo-path", type=str, default="/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb",
                      help="Path to the repository")
    parser.add_argument("--target-exts", type=str, nargs="+", default=[".py", ".js", ".cpp", ".h", ".md"],
                      help="File extensions to extract")
    args = parser.parse_args()
    
    # Step 1: Repository analysis
    analysis = analyze_repository(args.repo_path)
    
    # Step 2: Plan extraction
    plan = plan_extraction(analysis, args.target_exts)
    
    # Step 3: Simulate extraction
    files = simulate_extraction(args.repo_path, plan)
    
    # Step 4: Validate output format
    output_schema = {
        "sections": [],
        "section_relationships": {},
        "extraction_metadata": {},
        "section_fields": [
            "uuid", "id", "type", "name", "content", "language",
            "extraction_focus", "summary_instructions", "breadcrumb",
            "parent_uuid", "child_uuids"
        ]
    }
    validate_output_format(output_schema)
    
    print("\nExtraction checklist completed!")
    print(f"- Repository analyzed: {args.repo_path}")
    print(f"- Files to extract: {len(files)}")
    print("- Output format validated")
    
    # Recommendations
    print("\nRecommendations:")
    print("1. Run extraction with verified file discovery approach")
    print("2. Validate extraction results against expected file counts")
    print("3. Check for important files in the extraction output")
    print("4. Ensure all required fields are present in the output")

if __name__ == "__main__":
    main()