#!/usr/bin/env python3
"""
End-to-End JSON Extraction Demo

This script performs a complete extraction of JavaScript/Python code structures
from a repository and generates a compatible JSON output with QA integration.
It leverages the AI memory system for tracking state and handling errors.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("end_to_end_extraction.log")
    ]
)
logger = logging.getLogger("end_to_end_extraction")

# Import memory helpers
try:
    from src.agent_tools.dualipa.extraction.extraction_memory import (
        init_extraction_memory,
        track_extraction_start,
        track_extraction_progress,
        track_extraction_completion,
        record_extraction_error,
        find_similar_errors,
        save_extraction_knowledge,
        find_extraction_knowledge,
        get_extraction_context
    )
    MEMORY_AVAILABLE = True
except ImportError:
    logger.error("Failed to import extraction memory modules. Continuing without memory.")
    MEMORY_AVAILABLE = False

# Regular output directory for extracted data
OUTPUT_DIR = Path("extraction_output")

# Function to find source code files in a directory
def find_source_files(repo_path: str, extensions: List[str]) -> Dict[str, List[str]]:
    """
    Find all source files in a repository with the given extensions.
    
    Args:
        repo_path: Path to the repository
        extensions: List of file extensions to search for
        
    Returns:
        Dictionary mapping extensions to lists of file paths
    """
    repo_dir = Path(repo_path)
    result = {ext: [] for ext in extensions}
    
    # Walk through the repository
    for path in repo_dir.glob('**/*'):
        if path.is_file():
            ext = path.suffix.lower()
            if ext in extensions:
                result[ext].append(str(path))
    
    return result

# Function to extract code structures from a Python file
def extract_python_structures(file_path: str) -> Dict[str, Any]:
    """
    Extract classes, functions, and other structures from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with extracted structure information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize extraction result
        result = {
            "file_path": file_path,
            "language": "python",
            "classes": [],
            "functions": [],
            "imports": []
        }
        
        # Process the content line by line for basic extraction
        lines = content.split('\n')
        current_class = None
        current_function = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract imports
            if line.startswith('import ') or line.startswith('from '):
                result["imports"].append({
                    "line": i + 1,
                    "text": line
                })
            
            # Extract class definitions
            elif line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].strip(':')
                current_class = {
                    "name": class_name,
                    "line": i + 1,
                    "methods": [],
                    "attributes": []
                }
                result["classes"].append(current_class)
            
            # Extract function definitions
            elif line.startswith('def '):
                function_name = line.split('def ')[1].split('(')[0]
                
                if current_class is not None:
                    # Method in a class
                    method = {
                        "name": function_name,
                        "line": i + 1
                    }
                    current_class["methods"].append(method)
                else:
                    # Standalone function
                    function = {
                        "name": function_name,
                        "line": i + 1
                    }
                    result["functions"].append(function)
        
        return result
        
    except Exception as e:
        if MEMORY_AVAILABLE:
            record_extraction_error(
                "python_extraction_error",
                f"Error extracting Python structures: {str(e)}",
                file_path,
                severity=6
            )
        logger.error(f"Error extracting Python file {file_path}: {str(e)}")
        return {
            "file_path": file_path,
            "language": "python",
            "error": str(e)
        }

# Function to extract code structures from a JavaScript/TypeScript file
def extract_js_structures(file_path: str) -> Dict[str, Any]:
    """
    Extract classes, functions, and other structures from a JavaScript file.
    
    Args:
        file_path: Path to the JavaScript file
        
    Returns:
        Dictionary with extracted structure information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize extraction result
        result = {
            "file_path": file_path,
            "language": "javascript",
            "classes": [],
            "functions": [],
            "imports": []
        }
        
        # Process the content line by line for basic extraction
        lines = content.split('\n')
        current_class = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract imports
            if line.startswith('import ') or line.startswith('const') and 'require(' in line:
                result["imports"].append({
                    "line": i + 1,
                    "text": line
                })
            
            # Extract class definitions
            elif line.startswith('class '):
                class_name = line.split('class ')[1].split(' {')[0].split(' extends')[0]
                current_class = {
                    "name": class_name,
                    "line": i + 1,
                    "methods": []
                }
                result["classes"].append(current_class)
            
            # Extract constructor or method
            elif current_class is not None and ('constructor(' in line or 
                                               (line.strip() and '(' in line and not line.startswith(('if', 'for', 'while')))):
                method_name = line.split('(')[0].strip()
                method = {
                    "name": method_name,
                    "line": i + 1
                }
                current_class["methods"].append(method)
            
            # Extract standalone functions
            elif (line.startswith('function ') or 
                 ('const ' in line and ' = function(' in line) or
                 ('const ' in line and ' = (' in line and ') =>' in line)):
                
                # Extract function name
                if line.startswith('function '):
                    function_name = line.split('function ')[1].split('(')[0]
                elif ' = function(' in line:
                    function_name = line.split('const ')[1].split(' = ')[0]
                else:
                    function_name = line.split('const ')[1].split(' = ')[0]
                
                function = {
                    "name": function_name,
                    "line": i + 1
                }
                result["functions"].append(function)
        
        return result
        
    except Exception as e:
        if MEMORY_AVAILABLE:
            record_extraction_error(
                "js_extraction_error",
                f"Error extracting JavaScript structures: {str(e)}",
                file_path,
                severity=6
            )
        logger.error(f"Error extracting JavaScript file {file_path}: {str(e)}")
        return {
            "file_path": file_path,
            "language": "javascript",
            "error": str(e)
        }

# Main extraction function
def extract_repository(repo_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract code structures from all source files in a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Tuple of (statistics, extracted_data)
    """
    if MEMORY_AVAILABLE:
        repo_name = os.path.basename(repo_path)
        track_extraction_start(repo_name, "code")
    
    start_time = time.time()
    
    # Statistics
    stats = {
        "python_files": 0,
        "js_files": 0,
        "extracted_files": 0,
        "errors": 0,
        "start_time": start_time
    }
    
    # Find source files
    logger.info(f"Scanning repository {repo_path} for source files...")
    files = find_source_files(repo_path, ['.py', '.js', '.ts'])
    
    stats["python_files"] = len(files.get('.py', []))
    stats["js_files"] = len(files.get('.js', [])) + len(files.get('.ts', []))
    total_files = stats["python_files"] + stats["js_files"]
    
    logger.info(f"Found {stats['python_files']} Python files and {stats['js_files']} JavaScript/TypeScript files")
    
    if MEMORY_AVAILABLE:
        track_extraction_progress(
            repo_name,
            "scanning",
            f"Found {total_files} source files",
            "Extract code structures from files",
            stats
        )
    
    # Extract structures from each file
    extracted_data = []
    
    # Process Python files
    for file_path in files.get('.py', []):
        logger.info(f"Extracting Python structures from {file_path}...")
        result = extract_python_structures(file_path)
        
        extracted_data.append(result)
        
        if "error" in result:
            stats["errors"] += 1
        else:
            stats["extracted_files"] += 1
    
    # Update progress
    if MEMORY_AVAILABLE and stats["python_files"] > 0:
        track_extraction_progress(
            repo_name,
            "python_extraction",
            f"Extracted {stats['extracted_files']} Python files with {stats['errors']} errors",
            "Extract JavaScript structures",
            stats
        )
    
    # Process JavaScript files
    for ext in ['.js', '.ts']:
        for file_path in files.get(ext, []):
            logger.info(f"Extracting JavaScript structures from {file_path}...")
            result = extract_js_structures(file_path)
            
            extracted_data.append(result)
            
            if "error" in result:
                stats["errors"] += 1
            else:
                stats["extracted_files"] += 1
    
    # Finalize statistics
    end_time = time.time()
    stats["end_time"] = end_time
    stats["duration_seconds"] = end_time - start_time
    
    logger.info(f"Extraction completed: {stats['extracted_files']} files processed with {stats['errors']} errors")
    logger.info(f"Total extraction time: {stats['duration_seconds']:.2f} seconds")
    
    if MEMORY_AVAILABLE:
        track_extraction_completion(
            repo_name,
            f"Extracted {stats['extracted_files']} files with {stats['errors']} errors",
            stats
        )
    
    return stats, extracted_data

# Format extraction results as QA-compatible JSON
def format_qa_compatible(extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format extracted data into a structure compatible with QA generation.
    
    Args:
        extracted_data: List of extracted structures
        
    Returns:
        QA-compatible JSON structure
    """
    result = {
        "version": "1.0",
        "timestamp": time.time(),
        "extraction_type": "code_structures",
        "structures": []
    }
    
    # Process each extracted file
    for data in extracted_data:
        # Skip files with errors
        if "error" in data:
            continue
        
        file_path = data["file_path"]
        language = data["language"]
        
        # Process classes
        for cls in data.get("classes", []):
            structure = {
                "type": "class",
                "name": cls["name"],
                "language": language,
                "file_path": file_path,
                "line": cls["line"],
                "components": []
            }
            
            # Add methods as components
            for method in cls.get("methods", []):
                component = {
                    "type": "method",
                    "name": method["name"],
                    "line": method["line"]
                }
                structure["components"].append(component)
            
            # Add attributes as components (Python only)
            for attr in cls.get("attributes", []):
                component = {
                    "type": "attribute",
                    "name": attr["name"],
                    "line": attr["line"]
                }
                structure["components"].append(component)
            
            result["structures"].append(structure)
        
        # Process standalone functions
        for func in data.get("functions", []):
            structure = {
                "type": "function",
                "name": func["name"],
                "language": language,
                "file_path": file_path,
                "line": func["line"]
            }
            result["structures"].append(structure)
    
    return result

# Main function
def main():
    """Main function to run the end-to-end extraction."""
    parser = argparse.ArgumentParser(description="End-to-end extraction to QA-compatible JSON")
    
    parser.add_argument(
        "--repo-path",
        help="Path to repository to extract",
        required=True
    )
    
    parser.add_argument(
        "--output-file",
        help="Path to output file",
        default="qa_compatible_output.json"
    )
    
    parser.add_argument(
        "--memory-db",
        help="Path to memory database",
        default="extraction_memory.db"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize memory system if available
    if MEMORY_AVAILABLE:
        logger.info(f"Initializing memory system using database {args.memory_db}")
        init_extraction_memory(args.memory_db)
    
    # Run extraction
    logger.info(f"Starting extraction from {args.repo_path}")
    stats, extracted_data = extract_repository(args.repo_path)
    
    # Format output
    logger.info("Formatting extraction results for QA compatibility")
    qa_compatible = format_qa_compatible(extracted_data)
    
    # Write output
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_compatible, f, indent=2)
    
    logger.info(f"Extraction results written to {args.output_file}")
    
    # Print statistics
    print("\nExtraction Statistics:")
    print(f"Total Python files: {stats['python_files']}")
    print(f"Total JavaScript/TypeScript files: {stats['js_files']}")
    print(f"Successfully extracted files: {stats['extracted_files']}")
    print(f"Files with errors: {stats['errors']}")
    print(f"Total extraction time: {stats['duration_seconds']:.2f} seconds")
    
    # Show context from memory if available
    if MEMORY_AVAILABLE:
        repo_name = os.path.basename(args.repo_path)
        context = get_extraction_context(repo_name)
        print("\nExtraction Memory Context:")
        print(f"Task: {context.get('task', 'None')}")
        print(f"Progress: {context.get('progress', 'None')}")
        print(f"Next steps: {context.get('next_steps', 'None')}")
        
        # Get any tree-sitter knowledge
        knowledge = find_extraction_knowledge("tree-sitter")
        if isinstance(knowledge, list) and knowledge:
            print("\nExtraction Knowledge:")
            for item in knowledge[:1]:  # Just show the first item
                print(f"Topic: {item.get('topic', 'Unknown')}")
                print(f"Summary: {item.get('summary', 'No summary')}")

if __name__ == "__main__":
    main()