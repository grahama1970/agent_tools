"""
Tree-sitter language parsing utilities.

This module provides a simple wrapper around tree-sitter-languages
for parsing source code across multiple programming languages.
"""

from tree_sitter_languages import get_language, get_parser

def parse_code(code: str, language: str) -> dict:
    """
    Parse code using tree-sitter-languages.
    
    Args:
        code: Source code to parse
        language: Programming language of the code
        
    Returns:
        Dictionary containing the parse tree and metadata
    """
    parser = get_parser(language)
    tree = parser.parse(bytes(code, 'utf8'))
    return {
        'tree': tree,
        'root': tree.root_node,
        'language': language
    } 