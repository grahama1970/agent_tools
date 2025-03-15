"""
Code extraction module for DuaLipa.

This module provides functions to extract code from repositories,
parse files, and generate datasets for LoRA fine-tuning.

Official Documentation References:
- gitpython: https://gitpython.readthedocs.io/en/stable/tutorial.html
- loguru: https://loguru.readthedocs.io/en/stable/
- tqdm: https://tqdm.github.io/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

import os
import json 
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple, Any
import re
import time
from loguru import logger
from tqdm import tqdm
import shutil
import ast
from collections import defaultdict
import textwrap

# Import local modules
try:
    # When running as part of the package
    from .github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
    from .language_detection import detect_language
    from .markdown_parser import extract_code_blocks
    from .utils import format_string
except ImportError:
    # Handle case where this module is run standalone
    logger.warning("Running in standalone mode, attempting relative imports")
    try:
        from github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
        from language_detection import detect_language
        from markdown_parser import extract_code_blocks
        from utils import format_string
    except ImportError:
        # If that fails, define the format_string function locally for standalone use
        def format_string(text: str, **kwargs: Any) -> str:
            """Format a string with optional formatting parameters."""
            if not kwargs:
                return text.strip()
            try:
                return text.format(**kwargs).strip()
            except Exception as e:
                return f"{text.strip()} (Error: {str(e)})"

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Common extensions for code and documentation files
CODE_FILE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', 
    '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala'
}

DOCUMENTATION_FILE_EXTENSIONS = {
    '.md', '.rst', '.txt', '.ipynb', '.tex'
}

# Files to ignore
IGNORED_FILES = {
    'LICENSE', '.gitignore', '.gitattributes', '.gitmodules',
    'setup.py', 'requirements.txt', 'package.json', 'package-lock.json',
    'yarn.lock', 'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'poetry.lock',
    'Cargo.toml', 'Cargo.lock', 'Gemfile', 'Gemfile.lock'
}

# Directories to ignore
IGNORED_DIRECTORIES = {
    '.git', '.github', '.vscode', '.idea', '__pycache__', 
    'node_modules', 'venv', 'env', '.env', 'build', 'dist', 
    'target', 'out', 'bin', 'obj', 'tmp', 'temp', 'tests'
}

# Output directories structure
OUTPUT_DIRS = {
    "CODE_FILES": "code",
    "DOC_FILES": "docs",
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs"
}

def _is_code_file(filename: str) -> bool:
    """Check if a file is a code file based on its extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in CODE_FILE_EXTENSIONS


def _is_documentation_file(filename: str) -> bool:
    """Check if a file is a documentation file based on its extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in DOCUMENTATION_FILE_EXTENSIONS


def _should_process_file(
    file_path: str, 
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> bool:
    """
    Determine if a file should be processed based on include/exclude patterns.
    
    Args:
        file_path: Path to the file
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        Boolean indicating whether the file should be processed
    """
    import fnmatch
    
    # Get the filename
    filename = os.path.basename(file_path)
    
    # Skip ignored files
    if filename in IGNORED_FILES:
        return False
    
    # Check exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return False
    
    # Check include patterns
    if include_patterns:
        for pattern in include_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        # If include patterns are specified but none match, skip this file
        return False
    
    # If no include patterns, process the file
    return True


# ---------- Block Extraction Functions ----------

def _extract_python_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> None:
    """
    Extract functions and classes from Python code using AST.
    
    Args:
        file_path: Path to the Python file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
    """
    try:
        # Parse the Python code
        tree = ast.parse(content)
        
        # Create output directory for Python blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / "python"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract blocks from the AST
        block_count = 0
        lines = content.splitlines()
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block_count += 1
                
                # Get line numbers
                start = node.lineno - 1
                end = getattr(node, "end_lineno", None)
                if end is None:
                    # For Python < 3.8 that doesn't have end_lineno
                    # Find the last line by iterating through children
                    end = start
                    for child in ast.walk(node):
                        if hasattr(child, "lineno"):
                            end = max(end, getattr(child, "lineno", 0))
                
                # Extract the block code
                block_lines = lines[start:end]
                block_code = "\n".join(block_lines)
                
                # Get block metadata
                docstring = ast.get_docstring(node) or "No docstring available."
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                
                # Create block header with metadata
                header = f"# Original file: {file_path}\n"
                header += f"# Block type: {node_type}\n"
                header += f"# Name: {node.name}\n"
                header += f"# Docstring: {docstring}\n"
                
                # Add parameter info for functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [arg.arg for arg in node.args.args]
                    param_info = f"# Parameters: {', '.join(params)}" if params else "# No parameters."
                    header += f"{param_info}\n"
                
                header += "\n"
                
                # Combine header and code
                block_content = header + block_code
                
                # Save the block to a file
                output_file = blocks_dir / f"{file_path.stem}_{node.name}_{block_count}.py"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(block_content)
                
                # Update statistics
                stats["code_blocks"] += 1
                
        # If no blocks were extracted, log a message
        if block_count == 0:
            logger.debug(f"No blocks extracted from Python file: {file_path}")
            
        return block_count
        
    except SyntaxError:
        # Handle Python files that can't be parsed with AST
        logger.warning(f"SyntaxError parsing Python file: {file_path}, falling back to generic splitting")
        return _extract_generic_blocks(file_path, content, output_dir, stats, "python")
        
    except Exception as e:
        error_msg = f"Error extracting Python blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _extract_markdown_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> int:
    """
    Extract sections from markdown based on headers.
    
    Args:
        file_path: Path to the markdown file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for markdown blocks
        blocks_dir = output_dir / OUTPUT_DIRS["DOC_BLOCKS"] / "markdown"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Split on headers (lines starting with #)
        sections = re.split(r"(?m)(?=^# )", content)
        
        # Process each section
        block_count = 0
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Get the section header for naming
            header_match = re.match(r"^(# .+)", section)
            if header_match:
                header_line = header_match.group(1)
                section_title = re.sub(r"\W+", "_", header_line.strip("# ").strip()).strip("_")
            else:
                section_title = f"section{i}"
                
            # Create metadata header
            block_header = f"<!-- Original file: {file_path} -->\n"
            block_header += f"<!-- Section: {section_title} -->\n\n"
                
            # Combine header and content
            block_content = block_header + section
            
            # Save the section to a file
            output_file = blocks_dir / f"{file_path.stem}_{section_title}_{i}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["doc_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting markdown blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _extract_js_ts_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> int:
    """
    Extract functions and classes from JavaScript/TypeScript.
    Uses a simplified regex approach since Tree-sitter is complex to integrate.
    
    Args:
        file_path: Path to the JS/TS file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier (js, ts, etc.)
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for JS/TS blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Using regex to find function and class definitions
        # This is a simplified approach - a full parser would be better
        function_pattern = r"(function\s+\w+\s*\([^)]*\)\s*{[\s\S]*?})"
        arrow_func_pattern = r"(const\s+\w+\s*=\s*(?:\([^)]*\)|[^=]*)\s*=>\s*{[\s\S]*?})"
        class_pattern = r"(class\s+\w+[\s\S]*?{[\s\S]*?})"
        
        # Combine patterns
        patterns = [function_pattern, arrow_func_pattern, class_pattern]
        
        # Extract blocks
        blocks = []
        for pattern in patterns:
            blocks.extend(re.findall(pattern, content))
            
        # If no blocks found with regex, fall back to splitting by double newlines
        if not blocks:
            return _extract_generic_blocks(file_path, content, output_dir, stats, language)
            
        # Process each block
        block_count = 0
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
                
            # Create metadata header
            block_header = f"// Original file: {file_path}\n"
            block_header += f"// Block index: {i}\n\n"
            
            # Combine header and block
            block_content = block_header + block
            
            # Save the block to a file
            output_file = blocks_dir / f"{file_path.stem}_block_{i}.{language}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["code_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting JS/TS blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _extract_generic_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> int:
    """
    Extract blocks from generic code files by splitting on double newlines.
    
    Args:
        file_path: Path to the code file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Split by double newlines
        blocks = re.split(r"\n\s*\n", content)
        
        # Process each block
        block_count = 0
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block or len(block.split('\n')) < 3:  # Skip very small blocks
                continue
                
            # Create metadata header
            if language in ["python", "py"]:
                block_header = f"# Original file: {file_path}\n# Block index: {i}\n\n"
            elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "cs", "go", "rust", "swift", "kotlin"]:
                block_header = f"// Original file: {file_path}\n// Block index: {i}\n\n"
            elif language in ["html", "xml"]:
                block_header = f"<!-- Original file: {file_path} -->\n<!-- Block index: {i} -->\n\n"
            else:
                block_header = f"# Original file: {file_path}\n# Block index: {i}\n\n"
                
            # Combine header and block
            block_content = block_header + block
            
            # Save the block to a file
            file_ext = file_path.suffix if file_path.suffix else f".{language}"
            output_file = blocks_dir / f"{file_path.stem}_chunk_{i}{file_ext}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["code_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting generic blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _process_code_file(
    file_path: Path, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: Optional[str] = None,
    extract_blocks: bool = True
) -> None:
    """
    Process a code file and update statistics.
    
    Args:
        file_path: Path to the code file
        output_dir: Output directory to save processed files
        stats: Statistics dictionary to update
        language: Language of the code file (if known)
        extract_blocks: Whether to extract code blocks from the file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Detect language if not provided
        if not language:
            language = detect_language(file_path)
        
        # Update statistics
        stats["code_files"] += 1
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for code files
        code_dir = output_dir / OUTPUT_DIRS["CODE_FILES"]
        code_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory for the language if it doesn't exist
        lang_dir = code_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        comment_marker = ""
        if language in ["python", "py"]:
            comment_marker = "# "
        elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "cs", "go", "rust", "swift", "kotlin"]:
            comment_marker = "// "
        elif language in ["html", "xml"]:
            comment_marker = "<!-- "
        else:
            comment_marker = "# "  # Default to # for unknown languages
            
        path_comment = f"{comment_marker}Original file path: {file_path}\n\n"
        content_with_path = path_comment + content
        
        # Save the file to the output directory with a unique name
        # Add the original path as part of the filename to maintain context
        path_hash = hash(str(file_path)) % 10000
        rel_path = str(file_path).replace('/', '_').replace('\\', '_')
        output_file = lang_dir / f"{rel_path}_{path_hash:04d}{ext}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_with_path)
        
        # Extract blocks if requested
        if extract_blocks:
            blocks_extracted = 0
            if language in ["python", "py"]:
                blocks_extracted = _extract_python_blocks(file_path, content, output_dir, stats)
            elif language in ["javascript", "js"]:
                blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, "javascript")
            elif language in ["typescript", "ts"]:
                blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, "typescript")
            else:
                # Use generic block extraction for other languages
                blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, language)
                
            logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
        
        logger.debug(f"Processed code file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing code file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)


def _process_documentation_file(
    file_path: Path, 
    output_dir: Path, 
    stats: Dict[str, Any],
    extract_blocks: bool = True
) -> None:
    """
    Process a documentation file and update statistics.
    
    Args:
        file_path: Path to the documentation file
        output_dir: Output directory to save processed files
        stats: Statistics dictionary to update
        extract_blocks: Whether to extract blocks from the documentation file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Update statistics
        stats["documentation_files"] += 1
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for documentation files
        docs_dir = output_dir / OUTPUT_DIRS["DOC_FILES"]
        docs_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory based on the file type (e.g., markdown)
        doc_type = ext.replace('.', '')
        if not doc_type:
            doc_type = "text"
        type_dir = docs_dir / doc_type
        type_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        path_comment = ""
        if ext.lower() in ['.md', '.markdown']:
            path_comment = f"<!-- Original file path: {file_path} -->\n\n"
        else:
            path_comment = f"# Original file path: {file_path}\n\n"
            
        content_with_path = path_comment + content
        
        # Save the file to the output directory with a unique name
        # Add the original path as part of the filename to maintain context
        path_hash = hash(str(file_path)) % 10000
        rel_path = str(file_path).replace('/', '_').replace('\\', '_')
        output_file = type_dir / f"{rel_path}_{path_hash:04d}{ext}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_with_path)
        
        # Extract blocks if requested
        if extract_blocks:
            blocks_extracted = 0
            if ext.lower() in ['.md', '.markdown']:
                blocks_extracted = _extract_markdown_blocks(file_path, content, output_dir, stats)
            else:
                # For other doc types, use a simple splitting strategy
                blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, doc_type)
                
            logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
        
        logger.debug(f"Processed documentation file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing documentation file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)


def _process_repository(
    repo_dir: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    max_files: int = 1000,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    extract_documentation: bool = True,
    extract_code: bool = True,
    extract_blocks: bool = True
) -> Dict[str, Any]:
    """
    Process all files in a repository and extract code/documentation.
    
    Args:
        repo_dir: Path to the repository directory
        output_dir: Output directory to save processed files
        stats: Statistics dictionary to update
        max_files: Maximum number of files to process
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        extract_documentation: Whether to extract documentation files
        extract_code: Whether to extract code files
        extract_blocks: Whether to extract blocks from files for QA pair generation
        
    Returns:
        Updated statistics dictionary
    """
    try:
        # Walk through the repository
        processed_files = 0
        
        for root, dirs, files in os.walk(repo_dir):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
            
            # Process files
            for file in tqdm(files, desc="Processing files", leave=False):
                # Check if we've reached the maximum number of files
                if processed_files >= max_files:
                    logger.info(f"Reached maximum number of files ({max_files}), stopping.")
                    break
                
                file_path = os.path.join(root, file)
                
                # Skip files that don't match the include/exclude patterns
                if not _should_process_file(file_path, include_patterns, exclude_patterns):
                    continue
                
                # Update statistics
                stats["total_files"] += 1
                processed_files += 1
                
                # Process the file based on its type
                path_obj = Path(file_path)
                try:
                    if _is_code_file(file) and extract_code:
                        language = detect_language(path_obj)
                        _process_code_file(path_obj, output_dir, stats, language, extract_blocks)
                    elif _is_documentation_file(file) and extract_documentation:
                        _process_documentation_file(path_obj, output_dir, stats, extract_blocks)
                    else:
                        logger.debug(f"Skipping unsupported file: {file_path}")
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            
            # Check if we've reached the maximum number of files
            if processed_files >= max_files:
                break
        
        return stats
        
    except Exception as e:
        error_msg = f"Error processing repository: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return stats

def extract_repository(
    source: str, 
    output_path: str = None,
    max_files: int = 1000,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    extract_documentation: bool = True,
    extract_code: bool = True,
    extract_blocks: bool = True
) -> Dict[str, Any]:
    """
    Extract code and documentation from a repository.
    
    Args:
        source: Repository URL or local path
        output_path: Path to save the extracted data (defaults to data/files if None)
        max_files: Maximum number of files to extract
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        extract_documentation: Whether to extract documentation files
        extract_code: Whether to extract code files
        extract_blocks: Whether to extract blocks from files for QA pair generation
        
    Returns:
        Dictionary with statistics about the extraction process
    """
    # If no output path specified, use the default data/files path
    if output_path is None:
        # Get the path to the dualipa module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(module_dir, "data", "files")
    
    stats = {
        "total_files": 0,
        "code_files": 0,
        "documentation_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "languages": {},
        "file_types": {},
        "errors": []
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle GitHub repository URL
    if is_github_url(source):
        logger.info(f"Cloning GitHub repository: {source}")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_info = parse_github_url(source)
                clone_url = get_clone_url(repo_info['owner'], repo_info['repo'])
                
                # Clone the repository
                local_path = clone_github_repo(clone_url, temp_dir)
                if not local_path:
                    error_msg = f"Failed to clone repository: {source}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    return stats
                
                # If a specific path in the repo was provided, use it
                repo_dir = os.path.join(local_path, repo_info['path']) if repo_info['path'] else local_path
                return _process_repository(
                    repo_dir, 
                    output_dir, 
                    stats, 
                    max_files,
                    include_patterns,
                    exclude_patterns,
                    extract_documentation,
                    extract_code,
                    extract_blocks
                )
        except Exception as e:
            error_msg = f"Error processing GitHub repository: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
    
    # Handle local directory
    elif os.path.isdir(source):
        logger.info(f"Processing local directory: {source}")
        return _process_repository(
            source, 
            output_dir, 
            stats, 
            max_files,
            include_patterns,
            exclude_patterns,
            extract_documentation,
            extract_code,
            extract_blocks
        )
    
    # Handle local file
    elif os.path.isfile(source):
        logger.info(f"Processing single file: {source}")
        try:
            file_path = Path(source)
            language = detect_language(file_path)
            
            # Process the file based on its type
            if _is_code_file(file_path.name) and extract_code:
                _process_code_file(file_path, output_dir, stats, language, extract_blocks)
            elif _is_documentation_file(file_path.name) and extract_documentation:
                _process_documentation_file(file_path, output_dir, stats, extract_blocks)
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                
            return stats
        except Exception as e:
            error_msg = f"Error processing file {source}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
    
    # Invalid source
    else:
        error_msg = f"Invalid source: {source}. Must be a GitHub URL, local directory, or file."
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return stats

def move_extracted_files(source_dir: str, target_dir: str = None) -> int:
    """
    Move extracted files from a source directory to the target directory.
    
    Args:
        source_dir: Path to the source directory containing extracted files
        target_dir: Path to the target directory (defaults to data/files if None)
        
    Returns:
        Number of files moved
    """
    # If no target directory specified, use the default data/files path
    if target_dir is None:
        # Get the path to the dualipa module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(module_dir, "data", "files")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Track files moved
    files_moved = 0
    
    # Copy the directory structure and files
    for root, dirs, files in os.walk(source_path):
        rel_path = os.path.relpath(root, source_path)
        if rel_path == '.':
            rel_path = ''
        
        # Create corresponding directories in the target
        target_dir_path = target_path / rel_path
        target_dir_path.mkdir(exist_ok=True)
        
        # Copy files
        for file in files:
            source_file = Path(root) / file
            target_file = target_dir_path / file
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            files_moved += 1
            
    logger.info(f"Moved {files_moved} files from {source_dir} to {target_dir}")
    return files_moved

def load_template(filename: str) -> str:
    """Load a template file from the resources directory.
    
    Args:
        filename: Name of the template file to load
        
    Returns:
        String content of the template file
    """
    template_dir = Path(__file__).parent / "resources" / "templates"
    template_path = template_dir / filename
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading template {filename}: {str(e)}")
        return f"Error: Could not load template {filename}"

def demo_code_extractor() -> None:
    """Demonstrate the code extractor functionality with examples.
    
    This function shows how to use the main components of the code extractor:
    1. Extracting code from a Python file
    2. Extracting code blocks from a Markdown file
    3. Extracting a repository (using a local example)
    4. Using the default data directory
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("Code Extractor Demo")
        logger.info("===================")
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Create a sample Python file
            logger.info("\n1. Extracting code from a Python file:")
            python_file = temp_path / "sample.py"
            with open(python_file, "w") as f:
                f.write(load_template("sample_python.py"))
            
            # 2. Create a sample Markdown file
            logger.info("\n2. Extracting code blocks from a Markdown file:")
            md_file = temp_path / "sample.md"
            with open(md_file, "w") as f:
                f.write(load_template("sample_markdown.md"))
            
            # 3. Create a simple repo structure
            logger.info("\n3. Extracting a repository:")
            repo_dir = temp_path / "sample_repo"
            repo_dir.mkdir(exist_ok=True)
            (repo_dir / "src").mkdir(exist_ok=True)
            
            # Create some files in the repo
            with open(repo_dir / "src" / "main.py", "w") as f:
                f.write(load_template("repo_main.py"))
                
            with open(repo_dir / "README.md", "w") as f:
                f.write(load_template("repo_readme.md"))
            
            # 4. Extract each example
            output_dir = temp_path / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Extract Python file
            python_stats = extract_repository(
                str(python_file),
                output_path=str(output_dir / "python_example"),
                extract_documentation=False
            )
            logger.info(f"Python extraction stats: {json.dumps(python_stats, indent=2)}")
            
            # Extract Markdown file
            md_stats = extract_repository(
                str(md_file),
                output_path=str(output_dir / "markdown_example"),
                extract_code=False
            )
            logger.info(f"Markdown extraction stats: {json.dumps(md_stats, indent=2)}")
            
            # Extract repository
            repo_stats = extract_repository(
                str(repo_dir),
                output_path=str(output_dir / "repo_example"),
                max_files=10
            )
            logger.info(f"Repository extraction stats: {json.dumps(repo_stats, indent=2)}")
            
            logger.info("\nDemo completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())




def main():
    """Main function."""
    message = format_string("Hello, World!")
    print(message)
    

if __name__ == "__main__":
    main()
