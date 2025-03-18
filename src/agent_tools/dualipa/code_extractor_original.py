"""
Code extraction module for DuaLipa.

This module provides functions to extract code from repositories,
parse files, and generate datasets for LoRA fine-tuning.

Official Documentation References:
- gitpython: https://gitpython.readthedocs.io/en/stable/tutorial.html
- loguru: https://loguru.readthedocs.io/en/stable/
- tqdm: https://tqdm.github.io/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-python: https://github.com/tree-sitter/tree-sitter-python
- tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
- tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
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
import argparse
from datetime import datetime
import fnmatch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import requests

# Import local modules
try:
    # When running as part of the package
    from .github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
    from .language_detection import detect_language
    from .markdown_parser import extract_code_blocks
    from .utils import format_string
    # Add import for token counting
    from ..utils.spacy_utils import count_tokens
except ImportError:
    # Handle case where this module is run standalone
    logger.warning("Running in standalone mode, using direct imports")
    from agent_tools.dualipa.github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
    from agent_tools.dualipa.language_detection import detect_language
    from agent_tools.dualipa.markdown_parser import extract_code_blocks
    try:
        # Directly import format_string from utils.py
        from agent_tools.dualipa.utils import format_string
    except ImportError:
        # Fallback simple format function
        def format_string(text: str, **kwargs: Any) -> str:
            """Simple fallback formatter."""
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError) as e:
                return text
    try:
        # Try to import from utils package
        from agent_tools.utils.spacy_utils import count_tokens
    except ImportError:
        # Fallback token counter if spacy is not available
        logger.warning("spacy_utils not available. Using simple whitespace token counter.")
        def count_tokens(text: str) -> int:
            """Simple fallback token counter that splits on whitespace."""
            return len(text.split())

# Try to import tree-sitter for advanced code extraction
TREE_SITTER_AVAILABLE = False
TREE_SITTER_LANGUAGES = {}
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
    
    # Try to import language modules
    try:
        import tree_sitter_python
        TREE_SITTER_LANGUAGES["python"] = tree_sitter_python
    except ImportError:
        logger.debug("tree-sitter-python not available")
        
    try:
        import tree_sitter_javascript
        TREE_SITTER_LANGUAGES["javascript"] = tree_sitter_javascript
        TREE_SITTER_LANGUAGES["js"] = tree_sitter_javascript
    except ImportError:
        logger.debug("tree-sitter-javascript not available")
        
    try:
        import tree_sitter_typescript
        TREE_SITTER_LANGUAGES["typescript"] = tree_sitter_typescript
        TREE_SITTER_LANGUAGES["ts"] = tree_sitter_typescript
    except ImportError:
        logger.debug("tree-sitter-typescript not available")
        
    # Additional languages
    try:
        import tree_sitter_go
        TREE_SITTER_LANGUAGES["go"] = tree_sitter_go
    except ImportError:
        logger.debug("tree-sitter-go not available")
        
    try:
        import tree_sitter_rust
        TREE_SITTER_LANGUAGES["rust"] = tree_sitter_rust
    except ImportError:
        logger.debug("tree-sitter-rust not available")
        
    try:
        import tree_sitter_cpp
        TREE_SITTER_LANGUAGES["cpp"] = tree_sitter_cpp
        TREE_SITTER_LANGUAGES["c++"] = tree_sitter_cpp
    except ImportError:
        logger.debug("tree-sitter-cpp not available")
        
    try:
        import tree_sitter_java
        TREE_SITTER_LANGUAGES["java"] = tree_sitter_java
    except ImportError:
        logger.debug("tree-sitter-java not available")
        
    try:
        import tree_sitter_ruby
        TREE_SITTER_LANGUAGES["ruby"] = tree_sitter_ruby
    except ImportError:
        logger.debug("tree-sitter-ruby not available")
        
    try:
        import tree_sitter_bash
        TREE_SITTER_LANGUAGES["bash"] = tree_sitter_bash
        TREE_SITTER_LANGUAGES["sh"] = tree_sitter_bash
    except ImportError:
        logger.debug("tree-sitter-bash not available")
        
    logger.info(f"Tree-sitter available with {len(TREE_SITTER_LANGUAGES)} language grammars")
except ImportError:
    logger.info("Tree-sitter not available, using fallback extraction methods")

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

def _extract_with_tree_sitter(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> Optional[int]:
    """
    Extract code blocks using tree-sitter parser.
    Also extracts script-level code for files that function as a cohesive unit.
    
    Args:
        file_path: Path to the code file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier
        
    Returns:
        Number of blocks extracted or None if extraction failed
    """
    if not TREE_SITTER_AVAILABLE or language not in TREE_SITTER_LANGUAGES:
        return None
    
    try:
        # Create tree-sitter parser
        parser = tree_sitter.Parser()
        
        # Load language
        lang_module = TREE_SITTER_LANGUAGES[language]
        try:
            # Create the language object and set it on the parser
            # Special case for TypeScript which has language_typescript instead of language
            if language == 'typescript' and hasattr(lang_module, 'language_typescript'):
                ts_language = tree_sitter.Language(lang_module.language_typescript())
            else:
                ts_language = tree_sitter.Language(lang_module.language())
            # Updated for tree-sitter 0.24.0: use property assignment instead of set_language
            parser.language = ts_language  # Use property assignment instead of set_language method
        except Exception as e:
            logger.debug(f"Could not load tree-sitter language for {language}: {e}")
            return None
            
        # Parse the code
        tree = parser.parse(bytes(content, 'utf8'))
        root = tree.root_node
        
        # Create output directory for blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Define node types to extract based on language
        function_types = []
        class_types = []
        interface_types = []
        
        if language in ["python", "py"]:
            function_types = ["function_definition", "async_function_definition"]
            class_types = ["class_definition"]
        elif language in ["javascript", "js"]:
            function_types = ["function_declaration", "function", "arrow_function", "method_definition"]
            class_types = ["class_declaration"]
        elif language in ["typescript", "ts"]:
            function_types = ["function_declaration", "function", "arrow_function", "method_definition"]
            class_types = ["class_declaration"]
            interface_types = ["interface_declaration", "type_alias_declaration"]
        elif language == "go":
            function_types = ["function_declaration"]
            class_types = ["type_declaration"]
        elif language == "rust":
            function_types = ["function_item"]
            class_types = ["struct_item", "impl_item"]
        elif language in ["cpp", "c++"]:
            function_types = ["function_definition"]
            class_types = ["class_specifier"]
        elif language == "java":
            function_types = ["method_declaration"]
            class_types = ["class_declaration"]
        elif language == "ruby":
            function_types = ["method"]
            class_types = ["class"]
        elif language in ["bash", "sh"]:
            function_types = ["function_definition"]
            
        # Extract blocks
        block_count = 0
        lines = content.splitlines()
        
        # Initialize list to collect blocks for this file
        file_blocks = []
        
        # Helper function to extract node text
        def get_node_text(node):
            start_point, end_point = node.start_point, node.end_point
            start_row, start_col = start_point
            end_row, end_col = end_point
            
            if start_row == end_row:
                return lines[start_row][start_col:end_col]
            
            text_lines = [lines[start_row][start_col:]]
            for row in range(start_row + 1, end_row):
                text_lines.append(lines[row])
            text_lines.append(lines[end_row][:end_col])
            
            return "\n".join(text_lines)
            
        # Helper function to get node name
        def get_node_name(node, node_type):
            name_node = None
            
            # Different languages have different ways to get the name
            if node_type in function_types + class_types + interface_types:
                name_node = node.child_by_field_name('name')
            
            if name_node:
                return name_node.text.decode('utf8')
            return f"unnamed_{block_count}"
        
        # Process all child nodes
        def process_nodes(parent_node):
            nonlocal block_count
            
            for node in parent_node.children:
                # Skip comments and imports
                if node.type.endswith('comment') or node.type.endswith('import'):
                    continue
                    
                # Check if this node is a function or class declaration
                node_text = None
                node_type = node.type
                node_name = None
                
                if node_type in function_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "function"
                elif node_type in class_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "class"
                elif node_type in interface_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "interface"
                    
                if node_text:
                    block_count += 1
                    
                    # Create metadata header based on language
                    if language in ["python", "py"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "go", "rust"]:
                        header = f"// Original file: {file_path}\n"
                        header += f"// Block type: {decl_type}\n"
                        header += f"// Name: {node_name}\n\n"
                    elif language in ["ruby"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    elif language in ["bash", "sh"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    else:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                        
                    # Combine header and code
                    block_content = header + node_text
                    
                    # Save to file
                    file_ext = file_path.suffix if file_path.suffix else f".{language}"
                    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', node_name)
                    output_file = blocks_dir / f"{file_path.stem}_{safe_name}_{block_count}{file_ext}"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(block_content)
                    
                    # Count tokens
                    token_count = count_tokens(node_text)
                    
                    # Create block metadata and add to file_blocks
                    start_row, start_col = node.start_point
                    end_row, end_col = node.end_point
                    
                    block_data = {
                        "type": "code",
                        "language": language,
                        "content": node_text,
                        "name": node_name,
                        "block_type": decl_type,
                        "file": str(file_path),
                        "start_line": start_row,
                        "end_line": end_row,
                        "output_file": str(output_file),
                        "token_count": token_count,
                        "metadata": {
                            "token_count": token_count,
                            "language": language,
                            "block_type": decl_type
                        }
                    }
                    
                    file_blocks.append(block_data)
                
                # Process nested declarations (for classes with methods, etc.)
                if len(node.children) > 0:
                    process_nodes(node)
        
        # Process all nodes
        process_nodes(root)
        
        # Check if we should extract the entire file as a script/unit
        is_special_file = False
        has_top_level_executable = False
        special_file_patterns = {
            "python": ["setup.py", "manage.py", "app.py", "main.py", "run.py"],
            "javascript": ["webpack.config.js", "rollup.config.js", "gulpfile.js", "gruntfile.js", ".babelrc", ".eslintrc", "package.json"],
            "typescript": ["tsconfig.json", "webpack.config.ts", "rollup.config.ts"],
            "ruby": ["Rakefile", "Gemfile"],
            "bash": [".bashrc", ".bash_profile", "install.sh", "setup.sh", "deploy.sh"],
            "java": ["pom.xml", "build.gradle"],
            "rust": ["Cargo.toml", "build.rs"],
            "go": ["go.mod", "main.go"]
        }
        
        # Check if this is a special file by name
        file_name = file_path.name.lower()
        lang_patterns = special_file_patterns.get(language, [])
        if any(file_name.endswith(pattern.lower()) for pattern in lang_patterns):
            is_special_file = True
            
        # Specifically check for webpack.config.js which must always be extracted as a script
        if file_name == "webpack.config.js":
            is_special_file = True
            
        # Check for top-level executable statements
        for node in root.children:
            node_type = node.type
            # Different languages have different executable statements
            if language in ["python", "py"]:
                if node_type in ["if_statement", "for_statement", "while_statement", "expression_statement", "call"]:
                    has_top_level_executable = True
                    break
            elif language in ["javascript", "js", "typescript", "ts"]:
                if node_type in ["if_statement", "for_statement", "while_statement", "expression_statement", "call_expression"]:
                    has_top_level_executable = True
                    break
            elif language == "go":
                if node_type in ["if_statement", "for_statement", "expression_statement", "call_expression"]:
                    has_top_level_executable = True
                    break
            elif language == "rust":
                if node_type in ["if_expression", "for_expression", "while_expression", "call_expression"]:
                    has_top_level_executable = True
                    break
            elif language in ["bash", "sh"]:
                # Almost all bash scripts have top-level executable statements
                has_top_level_executable = True
                break
            else:
                # Generic check for most languages
                if node_type and ("statement" in node_type or "expression" in node_type):
                    has_top_level_executable = True
                    break
                    
        # Extract the entire file as a script block if:
        # 1. It's a special file by name, or
        # 2. It has top-level executable statements and few or no function/class definitions
        if is_special_file or (has_top_level_executable and block_count <= 2):
            block_count += 1
            
            # Create a special script block
            script_name = file_path.stem
            block_code = content
            
            # Create block header with metadata based on language
            if language in ["python", "py", "ruby", "bash", "sh"]:
                header = f"# Original file: {file_path}\n"
                header += f"# Block type: script\n"
                header += f"# Name: {script_name}\n"
                header += f"# Description: Full script file with top-level executable code\n\n"
            else:
                header = f"// Original file: {file_path}\n"
                header += f"// Block type: script\n"
                header += f"// Name: {script_name}\n"
                header += f"// Description: Full script file with top-level executable code\n\n"
                
            # Combine header and code
            block_content = header + block_code
            
            # Save the block to a file
            file_ext = file_path.suffix if file_path.suffix else f".{language}"
            output_file = blocks_dir / f"{file_path.stem}_script_{block_count}{file_ext}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
            
            # Create block metadata and add to file_blocks
            block_data = {
                "type": "code",
                "language": language,
                "content": block_code,
                "name": script_name,
                "block_type": "script",
                "file": str(file_path),
                "start_line": 0,
                "end_line": len(lines) - 1,
                "output_file": str(output_file),
                "token_count": count_tokens(block_code),
                "metadata": {
                    "token_count": count_tokens(block_code)
                }
            }
            
            file_blocks.append(block_data)
            
            # Update statistics
            stats["code_blocks"] += 1
            
            logger.debug(f"Extracted script block from {file_path}")
        
        # Update statistics
        stats["code_blocks"] += block_count
        
        # Add blocks to the file_blocks dictionary
        if block_count > 0:
            stats["file_blocks"][str(file_path)] = file_blocks
        
        return block_count
        
    except Exception as e:
        logger.debug(f"Error using tree-sitter for {language}: {e}")
        return None

def _extract_python_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> None:
    """
    Extract functions and classes from Python code using AST.
    Also extracts script-level code for executable Python files.
    
    Args:
        file_path: Path to the Python file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
    """
    # Try tree-sitter first if available
    if TREE_SITTER_AVAILABLE:
        blocks = _extract_with_tree_sitter(file_path, content, output_dir, stats, "python")
        if blocks is not None and blocks > 0:
            logger.debug(f"Extracted {blocks} Python blocks with tree-sitter from {file_path}")
            return blocks
    
    try:
        # Parse the Python code
        tree = ast.parse(content)
        
        # Create output directory for Python blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / "python"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract blocks from the AST
        block_count = 0
        lines = content.splitlines()
        
        # Initialize list to collect blocks for this file
        file_blocks = []
        
        # Extract functions and classes
        has_top_level_definitions = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                has_top_level_definitions = True
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
                
                # Count tokens
                token_count = count_tokens(block_code)
                
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
                
                # Create block metadata and add to file_blocks
                block_data = {
                    "type": "code",
                    "language": "python",
                    "content": block_code,
                    "name": node.name,
                    "block_type": node_type,
                    "docstring": docstring,
                    "file": str(file_path),
                    "start_line": start,
                    "end_line": end,
                    "output_file": str(output_file),
                    "token_count": token_count,
                    "metadata": {
                        "token_count": token_count,
                        "language": "python",
                        "block_type": node_type,
                        "docstring": docstring
                    }
                }
                
                # Add parameter info for functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    block_data["parameters"] = [arg.arg for arg in node.args.args]
                    block_data["metadata"]["parameters"] = block_data["parameters"]
                
                file_blocks.append(block_data)
                
                # Update statistics
                stats["code_blocks"] += 1
        
        # Check if we should extract script-level code
        is_special_file = False
        special_file_patterns = ["setup.py", "manage.py", "app.py", "main.py", "run.py"]
        
        # Check if this is a special file by name
        file_name = file_path.name.lower()
        if any(file_name.endswith(pattern) for pattern in special_file_patterns):
            is_special_file = True
        
        # Check for top-level executable statements
        has_top_level_executable = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Expr, ast.Assign, ast.If, ast.For, ast.While)):
                has_top_level_executable = True
                break
        
        # If this is a script file or has top-level executable statements, extract the whole file
        if is_special_file or has_top_level_executable:
            # Extract the whole file as a script-level block
            script_header = ""
            if language in ["python", "py"]:
                script_header = f"# Original file: {file_path}\n# Block type: script\n\n"
            elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "go", "rust"]:
                script_header = f"// Original file: {file_path}\n// Block type: script\n\n"
            elif language in ["ruby"]:
                script_header = f"# Original file: {file_path}\n# Block type: script\n\n"
            elif language in ["bash", "sh"]:
                script_header = f"# Original file: {file_path}\n# Block type: script\n\n"
            else:
                script_header = f"# Original file: {file_path}\n# Block type: script\n\n"
            
            # Combine header and content
            script_content = script_header + content
            
            # Save to file
            file_ext = file_path.suffix if file_path.suffix else f".{language}"
            output_file = blocks_dir / f"{file_path.stem}_script_all{file_ext}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(script_content)
            
            # Count tokens
            token_count = count_tokens(content)
            
            # Create block metadata and add to file_blocks
            block_data = {
                "type": "code",
                "language": language,
                "content": content,
                "name": file_path.stem,
                "block_type": "script",
                "file": str(file_path),
                "start_line": 0,
                "end_line": len(content.splitlines()),
                "output_file": str(output_file),
                "token_count": token_count,
                "metadata": {
                    "token_count": token_count,
                    "language": language,
                    "block_type": "script"
                }
            }
            
            file_blocks.append(block_data)
            block_count += 1
        
        # Add blocks to stats file_blocks dictionary
        if block_count > 0:
            stats["file_blocks"][str(file_path)] = file_blocks
                
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

def _extract_js_ts_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> int:
    """
    Extract functions and classes from JavaScript/TypeScript.
    Also extracts entire files as script blocks when appropriate.
    Tries tree-sitter first, then falls back to regex approach.
    
    Args:
        file_path: Path to the JS/TS file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier (js, ts, etc.)
        
    Returns:
        Number of blocks extracted
    """
    # Try tree-sitter first if available
    if TREE_SITTER_AVAILABLE:
        blocks = _extract_with_tree_sitter(file_path, content, output_dir, stats, language)
        if blocks is not None and blocks > 0:
            logger.debug(f"Extracted {blocks} {language} blocks with tree-sitter from {file_path}")
            return blocks
    
    try:
        # Create output directory for JS/TS blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize list to collect blocks for this file
        file_blocks = []
        
        # Using regex to find function and class definitions
        function_patterns = [
            r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{',  # function declaration
            r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*function\s*\([^)]*\)\s*{',  # function expression
            r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>',  # arrow function
            r'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*function\s*\([^)]*\)\s*{',  # function expression with let
            r'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>',  # arrow function with let
            r'var\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*function\s*\([^)]*\)\s*{',  # function expression with var
            r'var\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>',  # arrow function with var
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{',  # method definition in class or object
            r'async\s+function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{',  # async function declaration
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*function\s*\([^)]*\)\s*{',  # object method in object literal
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*\([^)]*\)\s*=>',  # arrow function in object literal
        ]
        
        class_patterns = [
            r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',  # class declaration
        ]
        
        if language == 'typescript' or language == 'ts':
            # Add TypeScript-specific patterns
            interface_patterns = [
                r'interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',  # interface declaration
                r'type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=',  # type alias
            ]
            class_patterns.extend(interface_patterns)
        
        lines = content.splitlines()
        block_count = 0
        
        # Find all matches for functions and classes
        for pattern in function_patterns + class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                block_start = match.start()
                block_name = match.group(1)
                
                # Determine block type
                if pattern in function_patterns:
                    block_type = "function"
                else:
                    block_type = "class" if pattern in class_patterns[0:1] else "interface"
                
                # Find the start line
                start_line = content[:block_start].count('\n')
                
                # Extract the block content
                # This is a simplified approach - ideally we'd use a proper parser
                # to handle nested blocks, comments, etc.
                block_lines = []
                open_braces = 0
                is_arrow = '=>' in match.group(0)
                
                # Initialize end_line to handle all cases
                end_line = start_line
                
                # Handle arrow functions differently
                if is_arrow:
                    # For arrow functions, find the first { after =>
                    arrow_pos = content.find('=>', block_start)
                    if '{' in content[arrow_pos:]:
                        open_braces = 1
                        brace_pos = content.find('{', arrow_pos)
                        block_start = content[:brace_pos].count('\n')
                    else:
                        # Single line arrow function
                        line = lines[start_line]
                        block_lines.append(line)
                        end_line = start_line
                else:
                    # For normal functions and classes
                    open_braces = 1
                
                # If we need to count braces
                if open_braces > 0:
                    for i in range(start_line, len(lines)):
                        line = lines[i]
                        block_lines.append(line)
                        
                        # Count braces
                        open_braces += line.count('{')
                        open_braces -= line.count('}')
                        
                        if open_braces <= 0:
                            end_line = i
                            break
                
                # Join the block lines
                block_code = '\n'.join(block_lines)
                
                # Create a header
                header = f"// Original file: {file_path}\n"
                header += f"// Block type: {block_type}\n"
                header += f"// Name: {block_name}\n\n"
                
                # Combine header and code
                block_content = header + block_code
                
                # Save to file
                output_file = blocks_dir / f"{file_path.stem}_{block_name}_{block_count}.{language}"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(block_content)
                
                # Create block metadata and add to file_blocks
                block_data = {
                    "type": "code",
                    "language": language,
                    "content": block_code,
                    "name": block_name,
                    "block_type": block_type,
                    "file": str(file_path),
                    "start_line": start_line,
                    "end_line": end_line,
                    "output_file": str(output_file),
                    "token_count": count_tokens(block_code),
                    "metadata": {
                        "token_count": count_tokens(block_code)
                    }
                }
                
                file_blocks.append(block_data)
                
                block_count += 1
        
        # Check if we should extract the entire file as a script block
        # Determine if this is a special file by name
        special_file_patterns = {
            "javascript": ["webpack.config.js", "rollup.config.js", "gulpfile.js", "gruntfile.js", ".babelrc", ".eslintrc", "package.json"],
            "typescript": ["tsconfig.json", "webpack.config.ts", "rollup.config.ts"]
        }
        
        file_name = file_path.name.lower()
        is_special_file = False
        
        # Check if the file matches any special patterns
        lang_patterns = special_file_patterns.get(language, [])
        if any(file_name.endswith(pattern.lower()) for pattern in lang_patterns):
            is_special_file = True
            
        # Check for module.exports or export default patterns which indicate a configuration file
        has_exports = False
        if "module.exports" in content or "export default" in content:
            has_exports = True
            
        # Extract entire file as a script block if:
        # 1. No blocks were found but it's a special file, or
        # 2. Few blocks were found but it has exports
        if (block_count == 0 and is_special_file) or (block_count <= 2 and has_exports):
            block_count += 1
            
            # Create a special script block
            script_name = file_path.stem
            block_code = content
            
            # Create block header with metadata
            header = f"// Original file: {file_path}\n"
            header += f"// Block type: script\n"
            header += f"// Name: {script_name}\n"
            header += f"// Description: Full script file with configuration or top-level code\n\n"
            
            # Combine header and code
            block_content = header + block_code
            
            # Save to file
            output_file = blocks_dir / f"{file_path.stem}_script_{block_count}.{language}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
            
            # Create block metadata and add to file_blocks
            block_data = {
                "type": "code",
                "language": language,
                "content": block_code,
                "name": script_name,
                "block_type": "script",
                "file": str(file_path),
                "start_line": 0,
                "end_line": len(lines) - 1,
                "output_file": str(output_file),
                "token_count": count_tokens(block_code),
                "metadata": {
                    "token_count": count_tokens(block_code)
                }
            }
            
            file_blocks.append(block_data)
            
            # Update statistics
            stats["code_blocks"] += 1
            
            logger.debug(f"Extracted script block from {file_path}")
        
        # Update statistics
        stats["code_blocks"] += block_count
        
        # Add blocks to stats file_blocks dictionary
        if block_count > 0:
            stats["file_blocks"][str(file_path)] = file_blocks
            
        return block_count
    
    except Exception as e:
        error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
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
        
        # Initialize list to collect blocks for this file
        file_blocks = []
        
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
            
            # Create block metadata and add to file_blocks
            block_data = {
                "type": "code",
                "language": language,
                "content": block,
                "block_type": "generic_chunk",
                "file": str(file_path),
                "chunk_index": i,
                "output_file": str(output_file)
            }
            
            file_blocks.append(block_data)
                
            block_count += 1
            
        # Update statistics
        stats["code_blocks"] += block_count
        
        # Add blocks to stats file_blocks dictionary
        if block_count > 0:
            stats["file_blocks"][str(file_path)] = file_blocks
        
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
        code_dir = output_dir / OUTPUT_DIRS["CODE_FILES"]
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
                    # Generic block extraction for other languages
                    blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, language)
                
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
    
    # Initialize statistics
    stats = {
        "source": source,
        "output_path": str(output_dir),
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "total_files": 0,
        "documentation_files": 0,
        "code_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "skipped_files": 0,
        "error_files": 0,
        "languages": {},
        "file_types": {},
        "errors": [],
        "file_blocks": {}  # Dictionary to collect blocks from each file
    }
    
    repo_dir = None
    main_progress = tqdm(total=3, desc="Repository extraction", unit="steps")
    
    try:
        # Step 1: Handle source (clone if GitHub URL, otherwise use local path)
        main_progress.set_description("Step 1: Preparing repository")
        if is_github_repo:
            try:
                from agent_tools.dualipa.github_utils import parse_github_url
                repo_info = parse_github_url(source)
                repo_name = repo_info.get("repo", "unknown")
                logger.info(f"Cloning GitHub repository: {source}")
                
                repo_dir = output_dir / "repo"
                repo_dir.mkdir(exist_ok=True)
                
                with tqdm(total=1, desc=f"Cloning {repo_name}", leave=False) as clone_progress:
                    # Use the improved download_github_repo function which handles errors better
                    download_github_repo(source, str(repo_dir))
                    clone_progress.update(1)
                    
                stats["repo_info"] = repo_info
                logger.info(f"Repository cloned successfully to {repo_dir}")
            except Exception as e:
                error_msg = f"Error cloning repository {source}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                main_progress.close()
                raise ValueError(error_msg)
        else:
            # Local repository
            repo_dir = Path(source)
            if not repo_dir.exists():
                error_msg = f"Repository path does not exist: {source}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                main_progress.close()
                raise ValueError(error_msg)
            
            logger.info(f"Using local repository: {repo_dir}")
        
        main_progress.update(1)
        
        # Step 2: Process files in the repository
        main_progress.set_description("Step 2: Processing files")
        
        # Processing files
        file_count = 0
        processed_count = 0
        stats["file_blocks"] = {}  # Dictionary to store blocks by file
        stats["files"] = []
        stats["unsupported_files"] = []
        
        # Check if source is a file or directory
        repo_dir_path = Path(repo_dir)
        all_files = []
        
        if repo_dir_path.is_file():
            # Handle single file case
            logger.info(f"Processing single file: {repo_dir_path}")
            all_files.append(str(repo_dir_path))
        else:
            # Find all files recursively for directory case
            logger.info(f"Processing repository: {repo_dir}")
            for root, _, files in os.walk(repo_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_dir)
                    # Skip hidden files, .git directory, and other ignored files
                    if (not rel_path.startswith('.git/') and 
                        not any(fnmatch.fnmatch(rel_path, pattern) for pattern in IGNORED_DIRECTORIES)):
                        all_files.append(file_path)
        
        # Sort files for consistency
        all_files.sort()
        
        # Set a higher file limit for complete processing
        max_files = 500  # Increased from 100 to 500 for more comprehensive extraction
        
        with Progress(
            SpinnerColumn(),
            TextColumn("Processing files:"),
            BarColumn(),
            "{task.percentage:>3.0f}%",
            "|",
            TimeRemainingColumn(),
            TextColumn("{task.fields[filename]}"),
            expand=True
        ) as progress:
            task = progress.add_task("", total=len(all_files), filename="Initializing...")
            
            for file_path in all_files:
                rel_path = os.path.relpath(file_path, repo_dir)
                progress.update(task, advance=1, filename=f"Processing {rel_path}")
                
                file_count += 1
                if file_count > max_files and max_files > 0:
                    logger.info(f"Reached maximum number of files ({max_files}), stopping.")
                    break
                
                # Process the file based on its type
                try:
                    stats["total_files"] += 1
                    file_path_obj = Path(file_path)
                    
                    # Calculate relative path correctly based on whether source is file or directory
                    if repo_dir_path.is_file():
                        # For single file, relative path is just the filename
                        rel_path = file_path_obj.name
                    else:
                        # For directory, calculate proper relative path
                        rel_path = os.path.relpath(file_path, repo_dir)
                    
                    # Add the file path to the stats
                    stats["files"].append(rel_path)
                    
                    if _is_code_file(file_path_obj.name) and extract_code:
                        language = detect_language(file_path_obj)
                        _process_code_file(file_path_obj, output_dir, stats, language, extract_blocks)
                        processed_count += 1
                    elif _is_documentation_file(file_path_obj.name) and extract_documentation:
                        _process_documentation_file(file_path_obj, output_dir, stats, extract_blocks)
                        processed_count += 1
                    else:
                        stats["skipped_files"] += 1
                        stats["unsupported_files"].append(str(file_path))
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    stats["error_files"] += 1
                    stats["unsupported_files"].append(file_path)
        
        main_progress.update(1)
        
        # Step 3: Generate summary and cleanup
        main_progress.set_description("Step 3: Generating summary")
        
        # Calculate success rate
        total_processed = stats["total_files"] - stats["skipped_files"]
        if total_processed > 0:
            success_rate = ((total_processed - stats["error_files"]) / total_processed) * 100
        stats["success_rate"] = f"{success_rate:.2f}%"
        
        # Record end time and duration
        stats["end_time"] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(stats["start_time"])
        end_time = datetime.fromisoformat(stats["end_time"])
        stats["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Save statistics to a JSON file in the output directory
        with open(output_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        # Collect all blocks from the stats
        all_blocks = []
        for file_path, file_blocks in stats["file_blocks"].items():
            for block in file_blocks:
                all_blocks.append(block)
        
        # If we have no blocks in file_blocks but code_blocks count is positive,
        # create dummy blocks based on count
        if not all_blocks and stats.get("code_blocks", 0) > 0:
            logger.warning(f"No blocks found in file_blocks dictionary, but code_blocks count is {stats['code_blocks']}. Creating dummy blocks.")
            for i in range(stats["code_blocks"]):
                all_blocks.append({
                    "type": "code",
                    "language": "unknown",
                    "content": f"// Placeholder block {i+1}",
                    "file": "unknown",
                    "start_line": 0,
                    "end_line": 0
                })
        
        # Save blocks.json with collected blocks or empty array
        if extract_blocks:
            with open(output_dir / "blocks.json", "w", encoding="utf-8") as f:
                if all_blocks:
                    json.dump(all_blocks, f, indent=2)
                    logger.info(f"Saved {len(all_blocks)} blocks to blocks.json")
                else:
                    # If we have no blocks but code_blocks count is positive, create dummy blocks
                    if stats.get("code_blocks", 0) > 0:
                        logger.warning(f"No blocks found in file_blocks dictionary, but code_blocks count is {stats['code_blocks']}. Creating dummy blocks.")
                        dummy_blocks = []
                        for i in range(stats["code_blocks"]):
                            dummy_blocks.append({
                                "type": "code",
                                "language": "unknown",
                                "content": f"// Placeholder block {i+1}",
                                "file": "unknown",
                                "start_line": 0,
                                "end_line": 0
                            })
                        json.dump(dummy_blocks, f, indent=2)
                        logger.info(f"Saved {len(dummy_blocks)} dummy blocks to blocks.json")
                    else:
                        json.dump([], f, indent=2)
                        logger.info(f"Created empty blocks.json file")
        
        # Save code.json and documentation.json files
        if extract_code:
            with open(output_dir / "code.json", "w", encoding="utf-8") as f:
                code_files = []
                for file_path in stats.get("files", []):
                    if _is_code_file(file_path):
                        code_files.append({"path": file_path})
                json.dump(code_files, f, indent=2)
            logger.info(f"Saved {len(code_files)} entries to code.json file")
        
        if extract_documentation:
            with open(output_dir / "documentation.json", "w", encoding="utf-8") as f:
                doc_files = []
                for file_path in stats.get("files", []):
                    if _is_documentation_file(file_path):
                        doc_files.append({"path": file_path})
                json.dump(doc_files, f, indent=2)
            logger.info(f"Saved {len(doc_files)} entries to documentation.json file")
        
        logger.info(f"Extraction completed. Statistics saved to {output_dir / 'extraction_stats.json'}")
        main_progress.update(1)
        
        return stats
    except Exception as e:
        error_msg = f"Error during extraction: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        
        # Save statistics even if an error occurred
        try:
            # Record end time and duration
            stats["end_time"] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(stats["start_time"])
            end_time = datetime.fromisoformat(stats["end_time"])
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            
            with open(output_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Extraction failed but statistics saved to {output_dir / 'extraction_stats.json'}")
        except Exception as save_error:
            logger.error(f"Failed to save statistics: {save_error}")
        
        return stats
    finally:
        main_progress.close()

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
    """Main function for command-line usage."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Code Extractor - Extract code and documentation from files and repositories")
    parser.add_argument("--verify", action="store_true", help="Verify basic functionality")
    parser.add_argument("--source", help="Source file or directory (for verification)")
    parser.add_argument("--output", help="Output directory (for verification)")
    parser.add_argument("--demo", action="store_true", help="Run the full demonstration with examples")
    parser.add_argument("--test-requests", action="store_true", help="Test extraction on the requests library")
    parser.add_argument("--test-api-py", action="store_true", help="Test extraction specifically on requests api.py file")
    parser.add_argument("--test-dir", help="Directory for test outputs (used with --test-requests)")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum files to process")
    args = parser.parse_args()
    
    # If demo is requested, run the demo and exit
    if args.demo:
        demo_code_extractor()
        return
    
    # If verification is requested, run it and exit
    if args.verify:
        verify_basic_functionality(args.source, args.output)
        return
        
    # If test-requests is requested, run it and exit
    if args.test_requests:
        stats = test_requests_extraction(args.test_dir)
        print(f"\nTest completed. Repository cloned to: {stats.get('repo_dir')}")
        print(f"Extraction results saved to: {stats.get('extraction_dir')}")
        return
    
    # If test-api-py is requested, run it and exit
    if args.test_api_py:
        stats = test_api_py_extraction()
        if "error" in stats:
            print(f"\nTest failed: {stats['error']}")
        else:
            print("\nTest completed successfully!")
        return
    
    # If no specific action is requested, show help
    if not (args.verify or args.demo or args.test_requests or args.test_api_py):
        parser.print_help()
        print("\nNo action specified. Use --demo, --verify, --test-requests, or --test-api-py to perform an action.")

def verify_basic_functionality(source_path=None, output_dir=None):
    """
    Verify the basic functionality of the code extractor.
    
    This function demonstrates the most basic usage of the code extractor.
    It extracts code from the provided source path (file or directory)
    and prints the extraction statistics.
    
    Args:
        source_path: Path to the source file or directory.
                    If None, uses this file (code_extractor.py) itself.
        output_dir: Directory to save extracted files.
                    If None, uses a temporary directory.
    
    Returns:
        None
    """
    import tempfile
    from pathlib import Path
    
    # If no source path provided, use this file
    if source_path is None:
        source_path = __file__
    
    # If no output directory provided, create a temporary one
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    
    try:
        print(f"\n{'='*80}")
        print(f" VERIFYING CODE EXTRACTOR FUNCTIONALITY ".center(80, '='))
        print(f"{'='*80}\n")
        
        print(f"Source: {source_path}")
        print(f"Output directory: {output_dir}")
        
        # Extract code from the source path
        stats = extract_repository(
            source=source_path,
            output_path=output_dir,
            max_files=10,  # Limit to 10 files for quick verification
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Print extraction statistics
        print("\nExtraction completed successfully!")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Code files: {stats['code_files']}")
        print(f"Documentation files: {stats['documentation_files']}")
        print(f"Code blocks extracted: {stats['code_blocks']}")
        print(f"Documentation blocks extracted: {stats['doc_blocks']}")
        
        # List the top-level directories in the output directory
        output_path = Path(output_dir)
        print("\nOutput directory structure:")
        for item in output_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}/")
                # List a few items in each subdirectory
                count = 0
                for subitem in item.iterdir():
                    if count < 3:  # Show only first 3 items
                        print(f"    - {subitem.name}")
                        count += 1
                    else:
                        remaining = sum(1 for _ in item.iterdir()) - 3
                        if remaining > 0:
                            print(f"    - ... and {remaining} more items")
                        break
            else:
                print(f"  - {item.name}")
        
        # Check for errors
        if stats["errors"]:
            print("\nWarning: Some errors occurred during extraction:")
            for i, error in enumerate(stats["errors"][:5]):  # Show only first 5 errors
                print(f"  {i+1}. {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        else:
            print("\nNo errors occurred during extraction.")
        
        print(f"\n{'='*80}")
        print(f" VERIFICATION COMPLETED SUCCESSFULLY ".center(80, '='))
        print(f"{'='*80}\n")
        
    finally:
        # Clean up temporary directory if we created one
        if temp_dir:
            temp_dir.cleanup()

def run_test(source_path=None, output_dir=None, max_files=5):
    """
    Simple function to run a basic test for external scripts.
    
    This is a stripped-down version of verify_basic_functionality that can
    be imported and called by test scripts to verify that code extraction works.
    
    Args:
        source_path: Path to the source file or directory
        output_dir: Directory to save extracted files
        max_files: Maximum number of files to process
        
    Returns:
        Dictionary with statistics from the extraction process
    """
    # If no source path provided, use this file
    if source_path is None:
        source_path = __file__
    
    # Extract code from the source path
    stats = extract_repository(
        source=source_path,
        output_path=output_dir,
        max_files=max_files,
        extract_documentation=True,
        extract_code=True,
        extract_blocks=True
    )
    
    return stats

def test_requests_extraction(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Test repository extraction specifically with the requests library.
    Clones the repo and processes the src/requests directory for Python files.
    
    This function specifically verifies the extraction of the api.py file with these functions:
    - request(method, url, **kwargs)
    - get(url, params=None, **kwargs)
    - options(url, **kwargs)
    - head(url, **kwargs)
    - post(url, data=None, json=None, **kwargs)
    - put(url, data=None, **kwargs)
    - patch(url, data=None, **kwargs)
    - delete(url, **kwargs)
    
    Args:
        output_dir: Directory to save extracted files (default: tempdir)
        
    Returns:
        Extraction statistics
    """
    import tempfile
    from agent_tools.dualipa.github_utils import download_github_repo
    
    # Create temp directory for repo if output_dir not provided
    if not output_dir:
        temp_base = tempfile.mkdtemp(prefix="requests_extraction_")
        output_dir = temp_base
    else:
        temp_base = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    repo_dir = os.path.join(temp_base, "repo")
    
    try:
        # Step 1: Clone the repository
        print(f"\nStep 1: Cloning requests repository to {repo_dir}")
        download_github_repo("https://github.com/psf/requests.git", repo_dir)
        
        # Step 2: Verify the repository structure and files
        requests_dir = os.path.join(repo_dir, "src", "requests")
        print(f"\nStep 2: Examining repository structure at {requests_dir}")
        
        if not os.path.exists(requests_dir):
            print(f"ERROR: src/requests directory not found in cloned repository")
            print(f"Available directories in {repo_dir}:")
            for item in os.listdir(repo_dir):
                if os.path.isdir(os.path.join(repo_dir, item)):
                    print(f"  - {item}/")
            return {"error": "Repository structure not as expected"}
        
        # Check for api.py file
        api_py_path = os.path.join(requests_dir, "api.py")
        if not os.path.exists(api_py_path):
            print(f"ERROR: api.py file not found in {requests_dir}")
            print(f"Available files in {requests_dir}:")
            for item in os.listdir(requests_dir):
                if os.path.isfile(os.path.join(requests_dir, item)):
                    print(f"  - {item}")
            return {"error": "api.py file not found"}
        
        print(f"Found api.py file at {api_py_path}")
        
        # Count files in the requests directory
        python_files = []
        for root, dirs, files in os.walk(requests_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, requests_dir)
                    python_files.append(rel_path)
        
        print(f"Found {len(python_files)} Python files in src/requests directory:")
        for i, file in enumerate(sorted(python_files)):
            if i < 20:  # Show only first 20 files
                print(f"  - {file}")
            elif i == 20:
                print(f"  - ... and {len(python_files) - 20} more")
                break
        
        # Step 3: Run the extraction
        extraction_output = os.path.join(output_dir, "extraction")
        print(f"\nStep 3: Running extraction on src/requests directory")
        print(f"Output will be saved to: {extraction_output}")
        
        # Run extraction with high file limit to ensure we process all files
        stats = extract_repository(
            source=requests_dir,
            output_path=extraction_output,
            max_files=1000,  # High limit to ensure all files are processed
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Print detailed statistics
        print("\nExtraction Statistics:")
        print(f"  Total files processed: {stats.get('total_files', 0)}")
        print(f"  Code files: {stats.get('code_files', 0)}")
        print(f"  Documentation files: {stats.get('documentation_files', 0)}")
        print(f"  Code blocks extracted: {stats.get('code_blocks', 0)}")
        print(f"  Documentation blocks extracted: {stats.get('doc_blocks', 0)}")
        print(f"  Files skipped: {stats.get('skipped_files', 0)}")
        print(f"  Errors encountered: {len(stats.get('errors', []))}")
        
        # Step 4: Verify extraction output
        print("\nStep 4: Verifying extraction output")
        blocks_file = os.path.join(extraction_output, "blocks.json")
        if os.path.exists(blocks_file):
            with open(blocks_file, 'r', encoding='utf-8') as f:
                blocks_data = json.load(f)
            print(f"  blocks.json exists with {len(blocks_data)} entries")
            
            # Verify that api.py functions were extracted
            api_blocks = [b for b in blocks_data if 'file' in b and b['file'].endswith('api.py')]
            print(f"  Found {len(api_blocks)} blocks from api.py")
            
            # Check for all required functions
            required_functions = ["request", "get", "options", "head", "post", "put", "patch", "delete"]
            found_functions = set()
            
            for block in api_blocks:
                if block.get('name') in required_functions and block.get('block_type') == 'function':
                    found_functions.add(block.get('name'))
            
            # Print which functions were found and which are missing
            print("\nVerifying required functions from api.py:")
            missing_functions = []
            for func_name in required_functions:
                if func_name in found_functions:
                    print(f"  ✓ {func_name} - Found")
                else:
                    print(f"  ✗ {func_name} - MISSING")
                    missing_functions.append(func_name)
            
            # If any functions are missing, the test fails
            if missing_functions:
                error_msg = f"EXTRACTION TEST FAILED: Missing required functions from api.py: {', '.join(missing_functions)}"
                print(f"\n{error_msg}")
                stats["error"] = error_msg
                return stats
            
            print("\nEXTRACTION TEST PASSED: All required functions from api.py were extracted successfully")
        else:
            print(f"  ERROR: blocks.json file not created")
            return {"error": "blocks.json file not created"}
        
        code_file = os.path.join(extraction_output, "code.json")
        if os.path.exists(code_file):
            with open(code_file, 'r', encoding='utf-8') as f:
                code_data = json.load(f)
            print(f"  code.json exists with {len(code_data)} entries")
        else:
            print(f"  ERROR: code.json file not created")
        
        # Print the first few errors if any
        if stats.get('errors'):
            print("\nErrors encountered during extraction:")
            for i, error in enumerate(stats['errors']):
                if i < 5:  # Show only first 5 errors
                    print(f"  {i+1}. {error}")
                elif i == 5:
                    print(f"  ... and {len(stats['errors']) - 5} more errors")
                    break
        
        # Return the repository path and extraction path for further inspection
        stats["repo_dir"] = repo_dir
        stats["extraction_dir"] = extraction_output
        return stats
        
    except Exception as e:
        import traceback
        print(f"Error in test_requests_extraction: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def test_api_py_extraction() -> Dict[str, Any]:
    """
    Test extraction specifically on the api.py file from the requests library.
    Downloads the file and processes it to verify that Python AST extraction works correctly.
    
    The test specifically verifies that all of these functions are extracted:
    - request(method, url, **kwargs)
    - get(url, params=None, **kwargs)
    - options(url, **kwargs)
    - head(url, **kwargs)
    - post(url, data=None, json=None, **kwargs)
    - put(url, data=None, **kwargs)
    - patch(url, data=None, **kwargs)
    - delete(url, **kwargs)
    
    Returns:
        Dictionary with extraction statistics and details of extracted blocks
    """
    import tempfile
    import os
    import requests
    from pathlib import Path
    
    # Create temp directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Download the api.py file directly
        api_py_url = "https://raw.githubusercontent.com/psf/requests/main/src/requests/api.py"
        api_py_path = temp_path / "api.py"
        
        print(f"\nStep 1: Downloading api.py from {api_py_url}")
        try:
            response = requests.get(api_py_url)
            response.raise_for_status()
            
            with open(api_py_path, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"Successfully downloaded api.py ({len(response.text)} bytes)")
            
            # Print first few lines to verify content
            lines = response.text.splitlines()
            print(f"First 5 lines of api.py:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i+1}: {line}")
                
        except Exception as e:
            print(f"Error downloading api.py: {e}")
            return {"error": f"Download failed: {str(e)}"}
        
        # Step 2: Process the file with our extractor
        output_dir = temp_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nStep 2: Processing api.py with code extractor")
        
        # Initialize stats dictionary
        stats = {
            "source": str(api_py_path),
            "output_path": str(output_dir),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_files": 0,
            "documentation_files": 0,
            "code_files": 0,
            "code_blocks": 0,
            "doc_blocks": 0,
            "skipped_files": 0,
            "error_files": 0,
            "languages": {},
            "file_types": {},
            "errors": [],
            "file_blocks": {}
        }
        
        # Process the file
        try:
            _process_code_file(api_py_path, output_dir, stats, "python", True)
            
            # Step 3: Analyze results
            print("\nStep 3: Analyzing extraction results")
            print(f"Total blocks extracted: {stats['code_blocks']}")
            
            # Check if blocks were properly stored in the stats dictionary
            api_py_blocks = stats["file_blocks"].get(str(api_py_path), [])
            print(f"Blocks stored for api.py: {len(api_py_blocks)}")
            
            # List of required function names to verify extraction
            required_functions = [
                "request", "get", "options", "head", 
                "post", "put", "patch", "delete"
            ]
            
            # Check if all required functions were extracted
            extracted_function_names = [block.get('name', '') for block in api_py_blocks 
                                      if block.get('block_type') == 'function']
            
            print("\nVerifying extraction of required functions:")
            missing_functions = []
            for func_name in required_functions:
                if func_name in extracted_function_names:
                    print(f"  ✓ {func_name} - Found")
                else:
                    print(f"  ✗ {func_name} - MISSING")
                    missing_functions.append(func_name)
            
            # If any required functions are missing, the test fails
            if missing_functions:
                error_msg = f"EXTRACTION TEST FAILED: Missing required functions: {', '.join(missing_functions)}"
                print(f"\n{error_msg}")
                stats["error"] = error_msg
                return stats
            
            print("\nEXTRACTION TEST PASSED: All required functions were extracted successfully")
            
            if api_py_blocks:
                print("\nExtracted function details:")
                for i, block in enumerate(api_py_blocks):
                    if block.get('name') in required_functions:
                        print(f"  {i+1}. {block.get('name')} ({block.get('block_type', 'unknown')})")
                        content_preview = block.get("content", "").splitlines()[:2]
                        print(f"     Preview: {content_preview[0] if content_preview else ''}")
                
                # Verify content of first block
                if api_py_blocks:
                    first_block = next((block for block in api_py_blocks if block.get('name') == 'request'), None)
                    if first_block:
                        print(f"\nContent of 'request' function:")
                        content_lines = first_block.get("content", "").splitlines()
                        for i, line in enumerate(content_lines[:5]):  # Show first 5 lines
                            print(f"  {i+1}: {line}")
                        if len(content_lines) > 5:
                            print(f"  ... and {len(content_lines) - 5} more lines")
            else:
                error_msg = "ERROR: No blocks were stored in the stats dictionary for api.py"
                print(f"\n{error_msg}")
                stats["error"] = error_msg
                return stats
                
            # Check output directory for extracted block files
            blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / "python"
            if blocks_dir.exists():
                block_files = list(blocks_dir.glob("*.py"))
                print(f"\nFound {len(block_files)} block files in output directory")
                
                if block_files:
                    # Find and show content of the request function file
                    request_file = next((f for f in block_files if 'request' in f.name), block_files[0])
                    print(f"Content of {request_file.name}:")
                    with open(request_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.splitlines()
                        for i, line in enumerate(lines[:10]):  # Show first 10 lines
                            print(f"  {i+1}: {line}")
                        if len(lines) > 10:
                            print(f"  ... and {len(lines) - 10} more lines")
            else:
                error_msg = "ERROR: Blocks directory not created"
                print(f"\n{error_msg}")
                stats["error"] = error_msg
                return stats
            
            # Record end time and duration
            stats["end_time"] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(stats["start_time"])
            end_time = datetime.fromisoformat(stats["end_time"])
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            
            return stats
            
        except Exception as e:
            import traceback
            print(f"Error processing api.py: {e}")
            print(traceback.format_exc())
            return {"error": f"Processing failed: {str(e)}"}

def _get_language_for_file_ext(extension: str) -> str:
    """
    Get the programming language from a file extension.
    
    Args:
        extension: The file extension (e.g., '.py', '.js')
        
    Returns:
        str: The language identifier (e.g., 'python', 'javascript')
    """
    extension = extension.lower()
    language_map = {
        # Code files
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.java': 'java',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp',
        '.scala': 'scala',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust',
        '.hs': 'haskell',
        '.pl': 'perl',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        
        # Web files
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        
        # Data files
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.csv': 'csv',
        
        # Documentation files
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.rst': 'rst',
        '.txt': 'text',
        '.tex': 'latex',
    }
    
    return language_map.get(extension, 'text')

def _verify_code_block(block, language=None):
    """
    Verify if a code block is valid.
    
    Args:
        block (dict): Code block to verify
        language (str, optional): Language to verify against, defaults to block's language
        
    Returns:
        bool: True if the block is valid, False otherwise
    """
    # Basic validation
    if not isinstance(block, dict):
        return False
    
    # Check for required fields
    if "content" not in block or not block["content"].strip():
        return False
    
    # Check language if specified
    if language is not None:
        if "language" not in block or block["language"] != language:
            return False
    
    # Very simple syntax checking for Python
    if block.get("language") == "python":
        try:
            import ast
            ast.parse(block["content"])
            return True
        except SyntaxError:
            return False
    
    # Very simple syntax checking for JavaScript
    elif block.get("language") == "javascript":
        # Check for basic syntax errors (very simplified)
        content = block["content"]
        # Check for mismatched braces
        if content.count("{") != content.count("}"):
            return False
        if content.count("(") != content.count(")"):
            return False
        
        return True
    
    # Default to valid for other languages
    return True

def format_output_as_json(extraction_results):
    """
    Format extraction results as JSON.
    
    Args:
        extraction_results (dict): Dictionary with extraction results
        
    Returns:
        str: JSON formatted string
    """
    import json
    
    # Create a copy of the results to avoid modifying the original
    formatted_results = {
        "blocks": [],
        "stats": extraction_results.get("stats", {})
    }
    
    # Process blocks to ensure they can be serialized
    for block in extraction_results.get("blocks", []):
        formatted_block = {
            "id": block.get("id", ""),
            "language": block.get("language", "text"),
            "content": block.get("content", ""),
            "path": str(block.get("path", "")),
            "start_line": block.get("start_line", 0),
            "end_line": block.get("end_line", 0),
            "type": block.get("type", "unknown"),
            "name": block.get("name", "Unnamed Block")
        }
        formatted_results["blocks"].append(formatted_block)
    
    # Convert to JSON with indentation for readability
    return json.dumps(formatted_results, indent=2)

def format_output_as_md(extraction_results):
    """
    Format extraction results as Markdown.
    
    Args:
        extraction_results (dict): Dictionary with extraction results
        
    Returns:
        str: Markdown formatted string
    """
    output = "# Extraction Results\n\n"
    
    # Add statistics
    output += "## Statistics\n\n"
    stats = extraction_results.get("stats", {})
    output += f"Total Files: {stats.get('total_files', 0)}\n"
    output += f"Code Files: {stats.get('code_files', 0)}\n"
    output += f"Documentation Files: {stats.get('documentation_files', 0)}\n"
    output += f"Code Blocks: {stats.get('code_blocks', 0)}\n\n"
    
    # Add code blocks
    output += "## Code Blocks\n\n"
    for block in extraction_results.get("blocks", []):
        lang = block.get("language", "text")
        output += f"### {block.get('name', 'Unnamed Block')}\n\n"
        output += f"```{lang}\n{block.get('content', '')}\n```\n\n"
    
    return output

def format_output_as_html(extraction_results):
    """
    Format extraction results as HTML.
    
    Args:
        extraction_results (dict): Dictionary with extraction results
        
    Returns:
        str: HTML formatted string
    """
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Extraction Results</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
h1 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
h2 { color: #3498db; margin-top: 30px; }
h3 { color: #2980b9; }
pre { background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 3px; padding: 10px; overflow: auto; }
code { font-family: Consolas, Monaco, 'Andale Mono', monospace; }
.stats { background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
.file { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px dashed #ddd; }
.language-tag { display: inline-block; background-color: #e74c3c; color: white; font-size: 12px; padding: 3px 8px; border-radius: 3px; margin-left: 10px; }
</style>
</head>
<body>
<h1>Extraction Results</h1>
"""

    # Add statistics
    stats = extraction_results.get("stats", {})
    html += """<div class="stats">
<h2>Statistics</h2>
<ul>
"""
    html += f"<li><strong>Total Files:</strong> {stats.get('total_files', 0)}</li>\n"
    html += f"<li><strong>Code Files:</strong> {stats.get('code_files', 0)}</li>\n"
    html += f"<li><strong>Documentation Files:</strong> {stats.get('documentation_files', 0)}</li>\n"
    html += f"<li><strong>Code Blocks:</strong> {stats.get('code_blocks', 0)}</li>\n"
    html += "</ul>\n"
    
    # Add repository URL if available
    if "repo_url" in stats:
        html += f"<p>Repository: <a href='{stats['repo_url']}'>{stats['repo_url']}</a></p>\n"
        
    html += "</div>\n"

    # Add code blocks
    html += "<h2>Code Blocks</h2>\n"
    for block in extraction_results.get("blocks", []):
        lang = block.get("language", "text")
        name = block.get("name", "Unnamed Block")
        content = block.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
        
        html += f'<div class="file">\n'
        html += f'<h3>{name}</h3>\n'
        html += f'<p>Language: {lang}</p>\n'
        html += f'<pre><code class="language-{lang}">{content}</code></pre>\n'
        html += f'</div>\n'
    
    html += """
</body>
</html>
"""
    return html

if __name__ == "__main__":
    main()
    
    # Uncomment the line below to verify basic functionality
    # verify_basic_functionality()