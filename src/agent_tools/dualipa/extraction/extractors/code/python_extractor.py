"""
Python code extraction for DuaLipa.

This module handles extraction of Python code using AST parsing,
focusing on functions, classes, and methods.

Key Features:
1. AST-based parsing
2. Function extraction
3. Class extraction
4. Method extraction
5. Import tracking

Dependencies:
- ast: For Python AST parsing (https://docs.python.org/3/library/ast.html)
- loguru: For logging (https://github.com/Delgan/loguru)
- textwrap: For text formatting (https://docs.python.org/3/library/textwrap.html)

Documentation Links:
- AST Module: https://docs.python.org/3/library/ast.html
- AST Walking: https://greentreesnakes.readthedocs.io/en/latest/
- Loguru: https://loguru.readthedocs.io/
- Python Text Processing: https://docs.python.org/3/library/text.html

Input/Output Specifications:

parse_python_ast(content: str) -> Optional[ast.AST]:
    Input:
        - content: Python source code
    Output:
        - AST object if successful, None if parsing fails

extract_python_blocks(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    Input:
        - file_path: Path to Python file
    Output:
        - Tuple containing:
            1. List of dictionaries:
                - uuid: str
                - type: str (function, class, method)
                - name: str
                - content: str
                - metadata: Dict[str, Any]
                    - line_start: int
                    - line_end: int
                    - imports: List[str]
                    - decorators: List[str]
                    - returns: Optional[str]
                    - class_name: str (for methods only)
            2. Statistics dictionary:
                - total_blocks: int
                - classes: int
                - functions: int
                - imports: int
                - file_blocks: Dict[str, List]

_get_node_text(node: ast.AST, source: str) -> Optional[str]:
    Input:
        - node: AST node
        - source: Original source code
    Output:
        - Extracted text if successful, None otherwise

_get_import_text(node: ast.AST, source: str) -> str:
    Input:
        - node: AST node (Import or ImportFrom)
        - source: Original source code
    Output:
        - Formatted import statement

_get_return_annotation(node: ast.FunctionDef, source: str) -> Optional[str]:
    Input:
        - node: Function definition node
        - source: Original source code
    Output:
        - Return type annotation if present, None otherwise

Related Files:
- js_ts_extractor.py: Similar extraction for JS/TS
- generic_extractor.py: Fallback extraction methods
"""

import ast
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats

def parse_python_ast(content: str) -> Optional[ast.AST]:
    """
    Parse Python content into an AST.
    
    Args:
        content: Python source code
        
    Returns:
        AST object if successful, None if parsing fails
    """
    try:
        return ast.parse(content)
    except SyntaxError as e:
        logger.error(f"Python syntax error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return None

def extract_python_blocks(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract code blocks from a Python file using AST parsing.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        Tuple of (extracted blocks, statistics)
    """
    try:
        # Initialize stats
        stats = init_stats()
        blocks = []
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse AST
        tree = parse_python_ast(content)
        if not tree:
            stats["errors"].append(f"Failed to parse {file_path}")
            return [], stats
            
        # Track imports for context
        imports = []
        current_class = None
        
        # Process nodes
        for node in ast.walk(tree):
            # Track imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(_get_import_text(node, content))
                stats["imports"] += 1
                continue
                
            # Extract classes
            if isinstance(node, ast.ClassDef):
                current_class = node
                stats["classes"] += 1
                
                # Get class content
                class_content = _get_node_text(node, content)
                if not class_content:
                    continue
                    
                # Create class block
                blocks.append({
                    "uuid": str(uuid.uuid4()),
                    "type": "class",
                    "name": node.name,
                    "content": class_content,
                    "metadata": {
                        "line_start": node.lineno,
                        "line_end": node.end_lineno,
                        "imports": imports.copy(),
                        "decorators": [_get_node_text(d, content) for d in node.decorator_list],
                        "bases": [_get_node_text(b, content) for b in node.bases]
                    }
                })
                continue
                
            # Extract functions
            if isinstance(node, ast.FunctionDef):
                stats["functions"] += 1
                
                # Get function content
                func_content = _get_node_text(node, content)
                if not func_content:
                    continue
                    
                # Determine if this is a method
                is_method = current_class is not None and node in current_class.body
                block_type = "method" if is_method else "function"
                
                # Create function/method block
                block = {
                    "uuid": str(uuid.uuid4()),
                    "type": block_type,
                    "name": node.name,
                    "content": func_content,
                    "metadata": {
                        "line_start": node.lineno,
                        "line_end": node.end_lineno,
                        "imports": imports.copy(),
                        "decorators": [_get_node_text(d, content) for d in node.decorator_list],
                        "returns": _get_return_annotation(node, content)
                    }
                }
                
                # Add class context for methods
                if is_method:
                    block["metadata"]["class_name"] = current_class.name
                    
                blocks.append(block)
                continue
                
        # Update stats
        stats.update({
            "total_blocks": len(blocks),
            "file_blocks": {file_path: blocks}
        })
        
        return blocks, stats
        
    except Exception as e:
        logger.error(f"Error extracting Python blocks from {file_path}: {e}")
        return [], stats

def _get_node_text(node: ast.AST, source: str) -> Optional[str]:
    """Extract source text for an AST node."""
    try:
        # Get node lines
        start = node.lineno - 1
        end = node.end_lineno
        lines = source.splitlines()[start:end]
        
        # Handle empty content
        if not lines:
            return None
            
        # Join lines and dedent
        return textwrap.dedent("\n".join(lines))
        
    except Exception as e:
        logger.error(f"Error getting node text: {e}")
        return None

def _get_import_text(node: ast.AST, source: str) -> str:
    """Extract import statement text."""
    try:
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
            return f"import {', '.join(names)}"
        elif isinstance(node, ast.ImportFrom):
            names = [n.name for n in node.names]
            module = node.module or ""
            return f"from {module} import {', '.join(names)}"
        return ""
    except Exception:
        return ""

def _get_return_annotation(node: ast.FunctionDef, source: str) -> Optional[str]:
    """Extract return type annotation if present."""
    try:
        if node.returns:
            return _get_node_text(node.returns, source)
        return None
    except Exception:
        return None

def usage_example() -> None:
    """Example usage of Python code extraction."""
    # Example Python file
    python_content = textwrap.dedent('''
    from typing import List, Optional
    
    class Person:
        """A simple person class."""
        
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age
            
        def greet(self) -> str:
            return f"Hello, {self.name}!"
            
    def process_people(people: List[Person]) -> None:
        """Process a list of people."""
        for person in people:
            print(person.greet())
    ''')
    
    # Save to temp file
    with open('temp.py', 'w') as f:
        f.write(python_content)
        
    # Extract blocks
    blocks, stats = extract_python_blocks('temp.py')
    
    print(f"Found {stats['total_blocks']} blocks:")
    for block in blocks:
        print(f"\nType: {block['type']}")
        print(f"Name: {block['name']}")
        if block['type'] == 'method':
            print(f"Class: {block['metadata']['class_name']}")
        print("Content:")
        print(textwrap.indent(block['content'], "    "))
        
    print("\nStatistics:")
    print(f"Classes: {stats['classes']}")
    print(f"Functions: {stats['functions']}")
    print(f"Imports: {stats['imports']}")
    
    # Cleanup
    import os
    os.remove('temp.py')

if __name__ == "__main__":
    print("Running Python extractor usage example...")
    usage_example()
    print("Done!") 