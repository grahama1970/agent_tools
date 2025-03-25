"""
Code block verification utilities for DuaLipa.

This module handles verification of code block syntax and structure,
ensuring that extracted code is valid and can be executed.

Key Features:
1. Syntax verification
2. Block structure validation
3. Language-specific checks
4. Indentation handling
5. Tree-sitter validation for JS/TS

Dependencies:
- ast: For Python syntax checking
- loguru: For logging
- tree-sitter: For JS/TS parsing
- tree_sitter_languages: For language support

Documentation Links:
- AST Module: https://docs.python.org/3/library/ast.html
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Loguru: https://loguru.readthedocs.io/

Input/Output Specifications:

verify_block(block: Dict[str, Any]) -> bool:
    Input:
        - block: Code block dictionary
    Output:
        - True if verified, False otherwise
    Example Input:
        {
            "type": "function",
            "name": "example",
            "content": "def example():\n    pass",
            "metadata": {
                "language": "python",
                "file": "example.py"
            }
        }
    Example Output:
        True

Related Files:
- language_utils.py: Used for language detection
- validation_utils.py: Used before verification
- tree_sitter_utils.py: Used for JS/TS verification
"""

import ast
import textwrap
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from .language_utils import get_language_info, normalize_language
from .tree_sitter_utils import get_parser

# tree-sitter is always available through tree-sitter-language-pack
from tree_sitter import Node

def verify_code_block(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify a code block's syntax and structure.
    
    Args:
        block: Code block dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Get language info
        language = normalize_language(block["metadata"]["language"])
        info = get_language_info(language)
        
        if not info:
            return False, f"Unsupported language: {language}"
            
        # Verify based on language
        if language == "python":
            return verify_python_block(block)
        elif language in ("javascript", "typescript"):
            return verify_js_ts_block(block)
        else:
            # For other languages, just check basic structure
            return verify_block_syntax(block)
            
    except Exception as e:
        logger.error(f"Error verifying code block: {e}")
        return False, str(e)

def verify_python_block(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify Python code block using AST parsing.
    
    Args:
        block: Code block dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Dedent content
        content = textwrap.dedent(block["content"])
        
        # Handle class methods by wrapping in class context
        if block["type"] == "method" and "class_name" in block["metadata"]:
            class_name = block["metadata"]["class_name"]
            content = f"class {class_name}:\n" + "\n".join(f"    {line}" for line in content.splitlines())
            
        # Parse with AST
        ast.parse(content)
        return True, None
        
    except SyntaxError as e:
        return False, f"Python syntax error: {str(e)}"
    except Exception as e:
        return False, f"Error verifying Python block: {str(e)}"

def verify_js_ts_block(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify JavaScript/TypeScript code block.
    
    Args:
        block: Code block dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Dedent content
        content = textwrap.dedent(block["content"])
        
        # Basic structure checks
        if block["type"] == "function":
            if not any(s in content for s in ["function", "=>"]):
                return False, "No function declaration found"
                
        elif block["type"] == "class":
            if "class" not in content:
                return False, "No class declaration found"
                
        elif block["type"] == "method":
            if not any(s in content for s in ["function", "=>", "("]):
                return False, "No method declaration found"
                
        # Check balanced braces
        if not verify_balanced_braces(content):
            return False, "Unbalanced braces"
            
        return True, None
        
    except Exception as e:
        return False, f"Error verifying JS/TS block: {str(e)}"

def verify_block_syntax(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify basic block syntax for any language.
    
    Args:
        block: Code block dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Dedent content
        content = textwrap.dedent(block["content"])
        
        # Check for empty content
        if not content.strip():
            return False, "Empty block content"
            
        # Check balanced braces and parentheses
        if not verify_balanced_braces(content):
            return False, "Unbalanced braces or parentheses"
            
        # Check indentation consistency
        if not verify_indentation(content):
            return False, "Inconsistent indentation"
            
        return True, None
        
    except Exception as e:
        return False, f"Error verifying block syntax: {str(e)}"

def verify_balanced_braces(content: str) -> bool:
    """Check for balanced braces and parentheses."""
    stack = []
    pairs = {
        "{": "}",
        "(": ")",
        "[": "]"
    }
    
    for char in content:
        if char in "{([":
            stack.append(char)
        elif char in "})]":
            if not stack:
                return False
            if char != pairs[stack.pop()]:
                return False
                
    return len(stack) == 0

def verify_indentation(content: str) -> bool:
    """Check for consistent indentation."""
    lines = content.splitlines()
    indent_size = None
    
    for line in lines:
        if not line.strip():
            continue
            
        # Get indentation level
        indent = len(line) - len(line.lstrip())
        
        # Skip first indented line
        if indent_size is None and indent > 0:
            indent_size = indent
            continue
            
        # Check if indentation is multiple of first indent
        if indent > 0 and indent_size and indent % indent_size != 0:
            return False
            
    return True

def verify_block(block: Dict[str, Any]) -> bool:
    """
    Verify a code block's content.
    
    Args:
        block: Code block dictionary
        
    Returns:
        True if verified, False otherwise
    
    Example Input:
        {
            "type": "function",
            "name": "example",
            "content": "def example():\n    pass",
            "metadata": {
                "language": "python",
                "file": "example.py"
            }
        }
    
    Example Output:
        True
    """
    try:
        # Get language info
        language = block["metadata"]["language"]
        info = get_language_info(language)
        if not info:
            return False
            
        content = block["content"]
        
        # Verify Python blocks
        if language == "python":
            try:
                ast.parse(content)
                return True
            except SyntaxError:
                return False
                
        # Verify JS/TS blocks
        elif language in ("javascript", "typescript"):
            try:
                parser = get_parser(language)
                if not parser:
                    logger.warning(f"Could not get parser for {language}, skipping verification")
                    return True  # Skip verification if parser not available
                tree = parser.parse(bytes(content, "utf8"))
                # Check for syntax errors in the tree
                if tree.root_node.has_error:
                    return False
                    
                # Additional validation for JavaScript: check for common errors
                if language == "javascript" and "return }" in content:
                    # Missing semicolon or value after return
                    return False
                    
                return True
            except Exception as e:
                logger.warning(f"Error verifying {language} code: {e}")
                return False
            
        # Generic verification
        return True
        
    except Exception as e:
        logger.error(f"Error verifying block: {e}")
        return False

def usage_example() -> None:
    """Example usage of verification utilities."""
    # Example Python block
    python_block = {
        "type": "function",
        "name": "factorial",
        "content": """
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        """,
        "metadata": {
            "language": "python",
            "source_file": "math.py",
            "line_start": 1,
            "line_end": 4
        }
    }
    
    # Example JavaScript block
    js_block = {
        "type": "function",
        "name": "factorial",
        "content": """
        function factorial(n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }
        """,
        "metadata": {
            "language": "javascript",
            "source_file": "math.js",
            "line_start": 1,
            "line_end": 6
        }
    }
    
    # Verify blocks
    print("Python block verification:")
    is_valid, error = verify_code_block(python_block)
    print(f"Valid: {is_valid}")
    if error:
        print(f"Error: {error}")
        
    print("\nJavaScript block verification:")
    is_valid, error = verify_code_block(js_block)
    print(f"Valid: {is_valid}")
    if error:
        print(f"Error: {error}")
    
    # Test using new verify_block function
    print("\nVerify block function:")
    print(f"Python valid: {verify_block(python_block)}")
    print(f"JavaScript valid: {verify_block(js_block)}")
        
    # Test syntax verification
    print("\nSyntax verification:")
    content = "function test() { if (x > 0) { return x; }"  # Missing brace
    print(f"Balanced braces: {verify_balanced_braces(content)}")
    
    content = """
    def test():
       print("a")
          print("b")
       print("c")
    """
    print(f"Consistent indentation: {verify_indentation(content)}") 