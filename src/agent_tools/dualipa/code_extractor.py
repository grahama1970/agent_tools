"""
DuaLiPA - Dual Language Processing Agent for Code and Documentation Extraction

This module provides tools for extracting and processing code from repositories,
generating structured representations for use with language models.

The module handles:
1. Code extraction from local or GitHub repositories
2. Code block identification and extraction
3. Language detection
4. Documentation extraction
5. Structured output for analysis

Main functions:
- extract_repository: Extract content from a repository
- extract_single_file: Extract content from a single file
- format_output_as_json: Format extracted data as JSON
- format_output_as_md: Format extracted data as Markdown
- format_output_as_html: Format extracted data as HTML
"""

import os
import re
import sys
import time
import json
import tempfile
import glob
import shutil
import ast
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from loguru import logger

from .language_detection import detect_language

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def count_tokens(text: str) -> int:
    """
    Simple token counter - splits on whitespace and punctuation.
    
    Args:
        text: Text to count tokens in
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)

# JSON serializer helper function for handling Path objects
def _json_serializer(obj):
    """Custom JSON serializer to handle Path objects and other non-serializable types."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):  # Handle datetime objects
        return obj.isoformat()
    return str(obj)  # Fall back to string representation for other types

# Language detection patterns
# ... rest of the file ...

# Add this new function for stats standardization
def initialize_stats_dict(source: Union[str, Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Initialize a standardized stats dictionary with all required fields.
    
    Args:
        source: Source repository or file path
        output_dir: Output directory path
        
    Returns:
        Stats dictionary with all required fields initialized
    """
    now = datetime.now()
    return {
        # Source and output information
        "source": str(source) if source else "",
        "repo_url": str(source) if source else "",  # For backward compatibility 
        "output_path": str(output_dir) if output_dir else "",
        
        # Timing information
        "start_time": now.isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        
        # File and block counts
        "total_files": 0,
        "documentation_files": 0,
        "code_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "skipped_files": 0,
        "error_files": 0,
        
        # Categorization
        "languages": {},
        "file_types": {},
        
        # Error tracking
        "errors": [],
        
        # Block storage
        "file_blocks": {}  # Dictionary to collect blocks from each file
    }

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
    Extract content from a repository source (local path or GitHub URL).
    
    Args:
        source: Path to the repository or GitHub URL
        output_path: Path to the output directory (default: temp directory)
        max_files: Maximum number of files to process (default: 1000)
        include_patterns: List of glob patterns to include files
        exclude_patterns: List of glob patterns to exclude files
        extract_documentation: Whether to extract documentation files (default: True)
        extract_code: Whether to extract code files (default: True)
        extract_blocks: Whether to extract blocks from files (default: True)
        
    Returns:
        Statistics dictionary
    """
    # Check if source is a GitHub URL
    from agent_tools.dualipa.github_utils import is_github_url, download_github_repo
    is_github_repo = is_github_url(source)
    
    # Set up output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(tempfile.gettempdir(), f"extracted_{timestamp}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics with standardized structure
    stats = initialize_stats_dict(source, output_dir)
    
    repo_dir = None
    
    try:
        # Handle GitHub repositories
        if is_github_repo:
            logger.info(f"Cloning GitHub repository: {source}")
            try:
                # Set up temporary directory for cloning
                repo_dir = tempfile.mkdtemp(prefix="github_repo_")
                logger.debug(f"Created temporary directory: {repo_dir}")
                
                # Clone the repository
                download_github_repo(source, repo_dir)
                
                # Now process the local repository directory
                source = repo_dir  # Update source to the local path
                # Continue with the local directory processing below
            except Exception as e:
                error_msg = f"Error cloning GitHub repository: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                
                # Update end time and duration
                end_time = datetime.now()
                stats["end_time"] = end_time.isoformat()
                stats["duration_seconds"] = (end_time - datetime.fromisoformat(stats["start_time"])).total_seconds()
                
                if repo_dir and os.path.exists(repo_dir):
                    shutil.rmtree(repo_dir, ignore_errors=True)
                
                return stats
        
        # Process local repository (or GitHub repo that was just cloned)
        if os.path.isdir(source):
            logger.info(f"Processing repository: {source}")
            repo_dir = Path(source)
            
            if not repo_dir.exists():
                error_msg = f"Repository directory does not exist: {repo_dir}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                return stats
            
            # Get all files to process
            all_files = []
            total_files = 0
            
            # Apply include patterns
            if include_patterns:
                for pattern in include_patterns:
                    pattern_path = os.path.join(source, pattern)
                    matched_files = glob.glob(pattern_path, recursive=True)
                    all_files.extend([f for f in matched_files if os.path.isfile(f)])
                    logger.info(f"Include pattern '{pattern}' matched {len(matched_files)} files")
            else:
                # Default: process all files recursively
                for root, _, files in os.walk(source):
                    for file in files:
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
                logger.info(f"Found {len(all_files)} files in repository")
            
            # Apply exclude patterns
            if exclude_patterns:
                excluded = []
                for pattern in exclude_patterns:
                    for file_path in list(all_files):
                        if re.search(pattern, file_path):
                            all_files.remove(file_path)
                            excluded.append(file_path)
                logger.info(f"Excluded {len(excluded)} files based on patterns")
            
            # Process files (limit to max_files)
            stats["total_files"] = len(all_files[:max_files])
            
            # Create a progress bar for file processing
            from tqdm import tqdm
            with tqdm(total=min(len(all_files), max_files), desc="Processing files") as pbar:
                for file_path in all_files[:max_files]:
                    try:
                        file_path = Path(file_path)
                        language = detect_language(file_path)
                        
                        # Process based on file type
                        if _is_code_file(file_path.name) and extract_code:
                            _process_code_file(file_path, output_dir, stats, language, extract_blocks=extract_blocks)
                        
                        elif _is_documentation_file(file_path.name) and extract_documentation:
                            _process_documentation_file(file_path, output_dir, stats, extract_blocks=extract_blocks)
                        
                        else:
                            stats["skipped_files"] = stats.get("skipped_files", 0) + 1
                            logger.debug(f"Skipping unsupported file: {file_path}")
                    
                    except Exception as e:
                        error_msg = f"Error processing file {file_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
                        stats["error_files"] = stats.get("error_files", 0) + 1
                    
                    pbar.update(1)
        
        # Handle single file
        elif os.path.isfile(source):
            logger.info(f"Processing single file: {source}")
            try:
                file_path = Path(source)
                language = detect_language(file_path)
                
                # Ensure total_files is set for single files
                stats["total_files"] = 1
                
                # Process the file based on its type
                if _is_code_file(file_path.name) and extract_code:
                    _process_code_file(file_path, output_dir, stats, language, extract_blocks=extract_blocks)
                elif _is_documentation_file(file_path.name) and extract_documentation:
                    _process_documentation_file(file_path, output_dir, stats, extract_blocks=extract_blocks)
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    stats["skipped_files"] = stats.get("skipped_files", 0) + 1
            
            except Exception as e:
                error_msg = f"Error processing file {source}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                stats["error_files"] = stats.get("error_files", 0) + 1
        
        # Invalid source
        else:
            error_msg = f"Invalid source: {source}. Must be a GitHub URL, local directory, or file."
            logger.error(error_msg)
            stats["errors"].append(error_msg)
        
        # Create blocks.json with extracted blocks
        blocks_dir = output_dir / "blocks"
        blocks_dir.mkdir(exist_ok=True)
        
        blocks_file = blocks_dir / "blocks.json"
        if stats["file_blocks"]:
            with open(blocks_file, "w", encoding="utf-8") as f:
                json.dump(stats["file_blocks"], f, indent=2, default=_json_serializer)
            logger.info(f"Saved {sum(len(blocks) for blocks in stats['file_blocks'].values())} blocks to blocks.json")
        
        # Save code.json with extracted code files
        code_files = []
        for fp, blocks in stats.get("file_blocks", {}).items():
            fp = Path(fp)
            if _is_code_file(fp.name):
                code_files.extend(blocks)
        
        code_file = blocks_dir / "code.json"
        with open(code_file, "w", encoding="utf-8") as f:
            json.dump(code_files, f, indent=2, default=_json_serializer)
        logger.info(f"Saved {len(code_files)} entries to code.json file")
        
        # Save documentation.json with extracted documentation files
        doc_files = []
        for fp, blocks in stats.get("file_blocks", {}).items():
            fp = Path(fp)
            if _is_documentation_file(fp.name):
                doc_files.extend(blocks)
        
        doc_file = blocks_dir / "documentation.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc_files, f, indent=2, default=_json_serializer)
        logger.info(f"Saved {len(doc_files)} entries to documentation.json file")
        
        # Update end time and duration
        end_time = datetime.now()
        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = (end_time - datetime.fromisoformat(stats["start_time"])).total_seconds()
        
        # Save extraction stats
        stats_file = output_dir / "extraction_stats.json"
        _save_stats_to_json(stats, stats_file, source)
        logger.info(f"Extraction completed. Statistics saved to {stats_file}")
        
        return stats
    
    except Exception as e:
        error_msg = f"Error during extraction: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        
        # Update end time and duration even on error
        end_time = datetime.now()
        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = (end_time - datetime.fromisoformat(stats["start_time"])).total_seconds()
        
        # Try to save stats even on error
        try:
            stats_file = output_dir / "extraction_stats.json"
            _save_stats_to_json(stats, stats_file, source)
        except Exception as stats_error:
            logger.error(f"Failed to save statistics: {stats_error}")
        
        # Clean up temporary directory if we created one
        if is_github_repo and repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        
        return stats

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
        language: Optional language identifier (auto-detected if None)
        extract_blocks: Whether to extract blocks from the code file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # If language is not provided, detect it
        if language is None:
            language = detect_language(file_path)
        
        # Update statistics
        stats["code_files"] += 1
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for code files
        code_dir = output_dir / "code_files"
        code_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory based on the language
        lang_dir = code_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        # Use appropriate comment syntax for the language
        lang_comment = {
            'python': '#',
            'javascript': '//',
            'typescript': '//',
            'java': '//',
            'c': '//',
            'cpp': '//',
            'go': '//',
            'rust': '//',
            'ruby': '#',
            'php': '//',
            'shell': '#',
            'bash': '#'
        }
        
        comment_char = lang_comment.get(language, '#')
        path_comment = f"{comment_char} Original file path: {file_path}\n\n"
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
            try:
                blocks_extracted = 0
                
                # Use the appropriate block extraction function based on language
                if language == 'python':
                    blocks_extracted = _extract_python_blocks(file_path, content, output_dir, stats)
                elif language in ['javascript', 'typescript', 'js', 'ts']:
                    blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, language)
                else:
                    # Skip block extraction for unsupported languages
                    logger.debug(f"Skipping block extraction for unsupported language: {language}")
                
                logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
            except Exception as e:
                error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                
        logger.debug(f"Processed code file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing code file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

def _extract_python_blocks(
    file_path: Path,
    content: str,
    output_dir: Path,
    stats: Dict[str, Any]
) -> int:
    """
    Extract functions and classes from Python code using AST.
    
    Args:
        file_path: Path to the Python file (must be a Path object)
        content: Content of the file as string
        output_dir: Output directory to save extracted blocks
        stats: Statistics dictionary
        
    Returns:
        Number of blocks extracted
    """
    # No need to explicitly set defaults, just ensure we're accessing keys that exist in the standard structure
    
    # Initialize file_blocks for this file if not already present
    if str(file_path) not in stats["file_blocks"]:
        stats["file_blocks"][str(file_path)] = []
    
    # Update language statistics
    stats["languages"]["python"] = stats["languages"].get("python", 0) + 1
    
    # Update file type statistics
    ext = file_path.suffix.lower()
    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    
    # Create output directory for Python blocks
    python_blocks_dir = output_dir / "blocks" / "code" / "python"
    python_blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize block counter
    blocks_extracted = 0
    
    try:
        # Parse the Python code using AST
        tree = ast.parse(content)
        
        # Extract all top-level blocks (functions and classes)
        for node in tree.body:
            try:
                # Extract functions
                if isinstance(node, ast.FunctionDef):
                    block_name = node.name
                    start_line = node.lineno
                    end_line = _find_func_end(content, start_line)
                    block_content = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    _save_python_block(
                        block_name, block_content, file_path, python_blocks_dir, 
                        start_line, end_line, stats, "function"
                    )
                    blocks_extracted += 1
                
                # Extract classes and their methods
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    start_line = node.lineno
                    end_line = _find_class_end(content, start_line)
                    class_content = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    # Save the entire class as a block
                    _save_python_block(
                        class_name, class_content, file_path, python_blocks_dir, 
                        start_line, end_line, stats, "class"
                    )
                    blocks_extracted += 1
                    
                    # Extract methods from the class
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef):
                            method_name = class_node.name
                            method_start_line = class_node.lineno
                            method_end_line = _find_func_end(content, method_start_line, indent_level=4)
                            method_content = "\n".join(content.splitlines()[method_start_line-1:method_end_line])
                            
                            _save_python_block(
                                f"{class_name}.{method_name}", method_content, file_path, 
                                python_blocks_dir, method_start_line, method_end_line, 
                                stats, "method"
                            )
                            blocks_extracted += 1
            
            except Exception as e:
                error_msg = f"Error processing Python node: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        # Update stats
        stats["code_blocks"] += blocks_extracted
        stats["code_files"] += 1
        
        return blocks_extracted
    
    except SyntaxError as e:
        error_msg = f"Python syntax error in {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0
    
    except Exception as e:
        error_msg = f"Error extracting Python blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0


def _find_func_end(content: str, start_line: int, indent_level: int = 0) -> int:
    """
    Find the end line of a function or method.
    
    Args:
        content: File content
        start_line: Line number where the function starts
        indent_level: Expected indentation level (0 for functions, 4 for methods)
        
    Returns:
        End line number of the function
    """
    lines = content.splitlines()
    # Skip the function definition line
    line_idx = start_line
    
    # Find the indentation of the function body (first line after def)
    while line_idx < len(lines) and (not lines[line_idx].strip() or lines[line_idx].strip().startswith("#")):
        line_idx += 1
    
    if line_idx >= len(lines):
        return len(lines)
    
    # Get indentation of first line of function body
    func_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    
    # Continue until we find a line with indentation <= function definition
    while line_idx < len(lines):
        # Skip empty lines and comments
        if not lines[line_idx].strip() or lines[line_idx].strip().startswith("#"):
            line_idx += 1
            continue
        
        # If we find a line with indentation <= function definition indentation, we've reached the end
        curr_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
        if curr_indent <= indent_level:
            break
            
        line_idx += 1
    
    return line_idx


def _find_class_end(content: str, start_line: int) -> int:
    """
    Find the end line of a class.
    
    Args:
        content: File content
        start_line: Line number where the class starts
        
    Returns:
        End line number of the class
    """
    return _find_func_end(content, start_line)


def _save_python_block(
    block_name: str,
    block_content: str,
    source_file: Path,
    output_dir: Path,
    start_line: int,
    end_line: int,
    stats: Dict[str, Any],
    block_type: str
) -> None:
    """
    Save a Python code block to a file and update stats.
    
    Args:
        block_name: Name of the block (function/class/method name)
        block_content: Content of the block
        source_file: Source file path
        output_dir: Output directory
        start_line: Start line of the block
        end_line: End line of the block
        stats: Statistics dictionary
        block_type: Type of block (function, class, method)
    """
    # Create a sanitized filename
    safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
    block_file = output_dir / f"{safe_name}.py"
    
    # Ensure we don't overwrite existing blocks with the same name
    if block_file.exists():
        count = 1
        while block_file.exists():
            block_file = output_dir / f"{safe_name}_{count}.py"
            count += 1
    
    # Write block to file
    with open(block_file, "w", encoding="utf-8") as f:
        f.write(block_content)
    
    # Create block metadata
    block_info = {
        "type": block_type,
        "name": block_name,
        "source_file": str(source_file),
        "source_start_line": start_line,
        "source_end_line": end_line,
        "content": block_content,
        "language": "python",
        "output_file": str(block_file),
        "extracted_at": datetime.now().isoformat()
    }
    
    # Add to stats
    stats["file_blocks"][str(source_file)].append(block_info)

def _extract_js_ts_blocks(
    file_path: Path,
    content: str,
    output_dir: Path,
    stats: Dict[str, Any],
    language: str = None
) -> int:
    """
    Extract functions, classes, and components from JavaScript and TypeScript.
    
    Args:
        file_path: Path to the JavaScript/TypeScript file (must be a Path object)
        content: Content of the file as string
        output_dir: Output directory to save extracted blocks
        stats: Statistics dictionary
        language: "javascript" or "typescript" (default: auto-detect from file extension)
        
    Returns:
        Number of blocks extracted
    """
    # No need to explicitly set defaults as stats is standardized
    
    # Auto-detect language if not provided
    if not language:
        if file_path.suffix.lower() in ['.ts', '.tsx']:
            language = 'typescript'
        else:
            language = 'javascript'
    
    # Initialize file_blocks for this file if not already present
    if str(file_path) not in stats["file_blocks"]:
        stats["file_blocks"][str(file_path)] = []
    
    # Update language statistics
    stats["languages"][language] = stats["languages"].get(language, 0) + 1
    
    # Update file type statistics
    ext = file_path.suffix.lower()
    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    
    # Create output directory for JS/TS blocks
    blocks_dir = output_dir / "blocks" / "code" / language
    blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize block counter
    blocks_extracted = 0
    
    try:
        # Define patterns for JS/TS blocks
        # Function patterns (traditional and arrow functions)
        function_patterns = [
            # Regular functions: function name()
            r'function\s+([a-zA-Z_$][\w$]*)\s*\([^)]*\)\s*{',
            # Arrow functions: const name = () => {
            r'(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>\s*{',
            # Method definitions: methodName() {
            r'(?<!\.|\'|\"|\})([a-zA-Z_$][\w$]*)\s*\([^)]*\)\s*{',
            # Class methods without the class: name() {
            r'([a-zA-Z_$][\w$]*)\s*=\s*function\s*\([^)]*\)\s*{',
        ]
        
        # Class pattern
        class_patterns = [
            # ES6 classes: class Name
            r'class\s+([a-zA-Z_$][\w$]*)',
            # React components: const Component
            r'(?:const|let|var)\s+([A-Z][a-zA-Z_$][\w$]*)\s*=\s*(?:React\.createClass|React\.Component|React\.PureComponent|\({)',
        ]
        
        # React component patterns (JSX components)
        component_patterns = [
            # React component functions: function ComponentName()
            r'function\s+([A-Z][a-zA-Z_$][\w$]*)\s*\([^)]*\)\s*{',
            # React component arrow functions: const ComponentName = ()
            r'(?:const|let|var)\s+([A-Z][a-zA-Z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>\s*{',
        ]
        
        # Extract blocks using regex patterns
        lines = content.splitlines()
        
        # Extract functions
        for pattern in function_patterns:
            blocks_extracted += _extract_blocks_with_pattern(
                content, pattern, file_path, blocks_dir, stats, 'function', language
            )
        
        # Extract classes
        for pattern in class_patterns:
            blocks_extracted += _extract_blocks_with_pattern(
                content, pattern, file_path, blocks_dir, stats, 'class', language
            )
        
        # Extract React components
        for pattern in component_patterns:
            blocks_extracted += _extract_blocks_with_pattern(
                content, pattern, file_path, blocks_dir, stats, 'component', language
            )
        
        # Handle script-level extraction (for config files like webpack.config.js)
        if blocks_extracted == 0 and _is_js_config_file(file_path.name):
            # Save the entire file as a script block
            script_name = file_path.stem
            script_file = blocks_dir / f"{script_name}_script.{file_path.suffix}"
            
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Create script block info
            script_info = {
                "type": "script",
                "name": script_name,
                "source_file": str(file_path),
                "source_start_line": 1,
                "source_end_line": len(lines),
                "content": content,
                "language": language,
                "output_file": str(script_file),
                "extracted_at": datetime.now().isoformat()
            }
            
            # Add to stats
            stats["file_blocks"][str(file_path)].append(script_info)
            blocks_extracted += 1
        
        # Update stats
        stats["code_blocks"] += blocks_extracted
        stats["code_files"] += 1
        
        return blocks_extracted
    
    except Exception as e:
        error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0


def _extract_blocks_with_pattern(
    content: str,
    pattern: str,
    file_path: Path,
    output_dir: Path,
    stats: Dict[str, Any],
    block_type: str,
    language: str
) -> int:
    """
    Extract code blocks using a regex pattern.
    
    Args:
        content: Content of the file as string
        pattern: Regex pattern to match block definitions
        file_path: Source file path
        output_dir: Output directory to save extracted blocks
        stats: Statistics dictionary
        block_type: Type of block (function, class, component)
        language: Programming language
        
    Returns:
        Number of blocks extracted
    """
    blocks_extracted = 0
    lines = content.splitlines()
    
    # Find all matches for the pattern
    matches = list(re.finditer(pattern, content))
    
    for match in matches:
        try:
            # Get block name and start position
            block_name = match.group(1).strip()
            start_pos = match.start()
            
            # Skip if name is a common keyword or starts with underscore
            if block_name in ['if', 'for', 'while', 'switch', 'catch'] or block_name.startswith('_'):
                continue
            
            # Find corresponding line number
            start_line = content[:start_pos].count('\n') + 1
            
            # Find block end (matching closing brace)
            end_line = _find_js_block_end(content, start_line)
            
            # Extract block content
            block_content = '\n'.join(lines[start_line-1:end_line])
            
            # Skip empty blocks or those too small
            if not block_content or len(block_content) < 10:
                continue
            
            # Create a sanitized filename
            safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
            file_ext = '.ts' if language == 'typescript' else '.js'
            block_file = output_dir / f"{safe_name}{file_ext}"
            
            # Ensure we don't overwrite existing blocks with the same name
            if block_file.exists():
                count = 1
                while block_file.exists():
                    block_file = output_dir / f"{safe_name}_{count}{file_ext}"
                    count += 1
            
            # Write block to file
            with open(block_file, 'w', encoding='utf-8') as f:
                f.write(block_content)
            
            # Create block metadata
            block_info = {
                "type": block_type,
                "name": block_name,
                "source_file": str(file_path),
                "source_start_line": start_line,
                "source_end_line": end_line,
                "content": block_content,
                "language": language,
                "output_file": str(block_file),
                "extracted_at": datetime.now().isoformat(),
                "block_type": block_type  # Duplicate for backward compatibility
            }
            
            # Add to stats
            stats["file_blocks"][str(file_path)].append(block_info)
            blocks_extracted += 1
            
        except Exception as e:
            logger.error(f"Error extracting {block_type} '{match.group(1)}': {str(e)}")
    
    return blocks_extracted


def _find_js_block_end(content: str, start_line: int) -> int:
    """
    Find the end line of a JS/TS block by tracking braces.
    
    Args:
        content: File content
        start_line: Line number where the block starts
        
    Returns:
        End line number of the block
    """
    lines = content.splitlines()
    # Start from the line after the definition
    line_idx = start_line
    
    # Find the opening brace
    while line_idx < len(lines) and '{' not in lines[line_idx]:
        line_idx += 1
    
    if line_idx >= len(lines):
        return len(lines)
    
    # Count braces to find matching closing brace
    brace_count = 0
    in_string = False
    string_char = None
    
    for i in range(line_idx, len(lines)):
        line = lines[i]
        
        for char in line:
            # Handle string literals to avoid counting braces inside strings
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            # Count braces outside strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    # Found matching closing brace
                    if brace_count == 0:
                        return i + 1
    
    # If no closing brace found, return end of file
    return len(lines)


def _is_js_config_file(filename: str) -> bool:
    """
    Check if a file is a JavaScript configuration file.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if the file is a JS config file, False otherwise
    """
    config_patterns = [
        r'\.config\.js$',
        r'webpack',
        r'babel',
        r'rollup',
        r'eslint',
        r'\.rc\.js$',
        r'tsconfig\.json$'
    ]
    
    for pattern in config_patterns:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    
    return False

# Output directories structure
OUTPUT_DIRS = {
    "CODE_FILES": "code",
    "DOC_FILES": "docs",
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs"
}

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
        stats: Statistics dictionary
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for markdown blocks
        blocks_dir = output_dir / OUTPUT_DIRS["DOC_BLOCKS"] / "markdown"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Update language statistics
        stats["languages"]["markdown"] = stats["languages"].get("markdown", 0) + 1
        
        # Update file type statistics
        ext = file_path.suffix.lower()
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Split on headers (lines starting with #)
        sections = re.split(r"(?m)(?=^# )", content)
        
        # Process each section
        block_count = 0
        # Initialize the list of blocks for this file in the stats dictionary
        file_blocks = []
        
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
            
            # Count tokens
            token_count = count_tokens(section)
            
            # Create block metadata and add to file_blocks
            block_data = {
                "type": "documentation",
                "language": "markdown",
                "content": section,
                "title": section_title,
                "file": str(file_path),
                "section": i,
                "output_file": str(output_file),
                "token_count": token_count,
                "metadata": {
                    "token_count": token_count,
                    "language": "markdown",
                    "section_title": section_title
                }
            }
            file_blocks.append(block_data)
                
            block_count += 1
            
        # Update statistics
        stats["doc_blocks"] += block_count
        
        # Add blocks to stats file_blocks dictionary
        if block_count > 0:
            stats["file_blocks"][str(file_path)] = file_blocks
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting markdown blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

# ... rest of the existing code ...
