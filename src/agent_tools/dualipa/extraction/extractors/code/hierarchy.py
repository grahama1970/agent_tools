"""
Code hierarchy analysis for DuaLipa.

This module handles code structure analysis and hierarchy extraction,
focusing on class relationships, function dependencies, and module organization.

Key Features:
1. Class hierarchy analysis
2. Function dependency tracking
3. Module relationship mapping
4. Import graph generation

Dependencies:
- ast: For Python parsing (https://docs.python.org/3/library/ast.html)
- tree-sitter: For JS/TS parsing (https://tree-sitter.github.io/tree-sitter/)
- loguru: For logging (https://github.com/Delgan/loguru)
- tree_sitter_languages: For language support (https://github.com/grantjenks/py-tree-sitter-languages)

Documentation Links:
- AST Module: https://docs.python.org/3/library/ast.html
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Loguru: https://loguru.readthedocs.io/
- Tree-sitter Languages: https://py-tree-sitter-languages.readthedocs.io/

Input/Output Specifications:

analyze_code_hierarchy(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Input:
        - file_path: Path to source file
    Output:
        - Tuple containing:
            1. Hierarchy dictionary:
                - file_path: str
                - language: str
                - imports: List[str]
                - classes: Dict[str, Dict]
                    - bases: List[str]
                    - methods: List[Dict]
                    - line_start: int
                    - line_end: int
                    - decorators: List[str]
                - functions: Dict[str, Dict]
                - dependencies: Dict[str, List[str]]
            2. Statistics dictionary

analyze_python_hierarchy(content: str, file_path: str, stats: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Input:
        - content: Python source code
        - file_path: Path to source file
        - stats: Statistics dictionary
    Output:
        - Same format as analyze_code_hierarchy

analyze_js_ts_hierarchy(content: str, file_path: str, language: str, stats: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Input:
        - content: JS/TS source code
        - file_path: Path to source file
        - language: 'javascript' or 'typescript'
        - stats: Statistics dictionary
    Output:
        - Same format as analyze_code_hierarchy, plus:
            - exports: List[str]
            - interfaces: Dict[str, Dict] (TypeScript only)

analyze_generic_hierarchy(content: str, file_path: str, language: str, stats: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Input:
        - content: Source code
        - file_path: Path to source file
        - language: Programming language
        - stats: Statistics dictionary
    Output:
        - Same format as analyze_code_hierarchy

build_code_hierarchy(blocks: List[Dict[str, Any]], source: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    Input:
        - blocks: List of code blocks
        - source: Optional source directory
        - output_dir: Optional output directory
    Output:
        - Statistics dictionary

Related Files:
- python_extractor.py: Uses hierarchy for Python code
- js_ts_extractor.py: Uses hierarchy for JS/TS code
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from loguru import logger
import textwrap

from tree_sitter import Parser
from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_utils import get_parser

# Re-export initialize_stats_dict for backward compatibility
__all__ = ['initialize_stats_dict']

def analyze_code_hierarchy(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze code hierarchy in a file.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Tuple of (hierarchy info, statistics)
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        # Detect language
        language = detect_language(file_path)
        if language == "unknown":
            stats["errors"].append(f"Unknown language for file: {file_path}")
            return {}, stats
            
        # Get language info
        info = get_language_info(language)
        if not info:
            stats["errors"].append(f"Unsupported language: {language}")
            return {}, stats
            
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Analyze based on language
        if language == "python":
            return analyze_python_hierarchy(content, file_path, stats)
        elif language in ("javascript", "typescript"):
            return analyze_js_ts_hierarchy(content, file_path, language, stats)
        else:
            return analyze_generic_hierarchy(content, file_path, language, stats)
            
    except Exception as e:
        logger.error(f"Error analyzing code hierarchy: {e}")
        return {}, stats

def analyze_python_hierarchy(
    content: str,
    file_path: str,
    stats: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze Python code hierarchy using AST."""
    try:
        # Parse AST
        tree = ast.parse(content)
        
        # Track relationships
        classes = {}
        functions = {}
        imports = []
        dependencies = {}
        
        # Process nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(_get_import_text(node))
                stats["imports"] = stats.get("imports", 0) + 1
                
            elif isinstance(node, ast.ClassDef):
                stats["classes"] = stats.get("classes", 0) + 1
                classes[node.name] = {
                    "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                    "methods": [],
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "decorators": [_get_decorator_name(d) for d in node.decorator_list]
                }
                
            elif isinstance(node, ast.FunctionDef):
                stats["functions"] = stats.get("functions", 0) + 1
                parent_class = _get_parent_class(node)
                func_info = {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "decorators": [_get_decorator_name(d) for d in node.decorator_list],
                    "returns": _get_return_annotation(node),
                    "args": _get_function_args(node)
                }
                
                if parent_class and parent_class in classes:
                    classes[parent_class]["methods"].append(func_info)
                else:
                    functions[node.name] = func_info
                    
        # Build hierarchy
        hierarchy = {
            "file_path": file_path,
            "language": "python",
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "dependencies": dependencies
        }
        
        return hierarchy, stats
        
    except Exception as e:
        logger.error(f"Error analyzing Python hierarchy: {e}")
        return {}, stats

def analyze_js_ts_hierarchy(
    content: str,
    file_path: str,
    language: str,
    stats: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze JavaScript/TypeScript code hierarchy using tree-sitter."""
    parser = get_parser(language)
    tree = parser.parse(bytes(content, "utf8"))
    
    # Track relationships
    classes = {}
    functions = {}
    imports = []
    exports = []
    interfaces = {}  # TypeScript only
    
    def visit_node(node: Any) -> None:
        """Process a tree-sitter node."""
        try:
            if node.type == "import_statement":
                imports.append(_get_node_text(node, content))
                stats["imports"] = stats.get("imports", 0) + 1
                
            elif node.type == "export_statement":
                exports.append(_get_node_text(node, content))
                stats["exports"] = stats.get("exports", 0) + 1
                
            elif node.type == "class_declaration":
                stats["classes"] = stats.get("classes", 0) + 1
                class_name = _get_class_name(node, content)
                if class_name:
                    classes[class_name] = {
                        "methods": [],
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "extends": _get_extends_class(node, content),
                        "implements": _get_implements_interfaces(node, content)
                    }
                    
            elif node.type == "method_definition":
                parent_class = _get_parent_class_ts(node, content)
                if parent_class and parent_class in classes:
                    method_info = {
                        "name": _get_method_name(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "is_static": _is_static_method(node),
                        "is_async": _is_async_method(node),
                        "return_type": _get_ts_return_type(node, content)
                    }
                    classes[parent_class]["methods"].append(method_info)
                    
            elif node.type == "interface_declaration" and language == "typescript":
                stats["interfaces"] = stats.get("interfaces", 0) + 1
                interface_name = _get_interface_name(node, content)
                if interface_name:
                    interfaces[interface_name] = {
                        "extends": _get_extends_interfaces(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "properties": _get_interface_properties(node, content)
                    }
                    
            # Visit children
            for child in node.children:
                visit_node(child)
                
        except Exception as e:
            logger.error(f"Error visiting node: {e}")
            
    # Process the tree
    visit_node(tree.root_node)
    
    # Build hierarchy
    hierarchy = {
        "file_path": file_path,
        "language": language,
        "imports": imports,
        "exports": exports,
        "classes": classes,
        "functions": functions
    }
    
    if language == "typescript":
        hierarchy["interfaces"] = interfaces
        
    return hierarchy, stats

def analyze_generic_hierarchy(
    content: str,
    file_path: str,
    language: str,
    stats: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze code hierarchy using pattern matching."""
    try:
        # Track relationships
        classes = {}
        functions = {}
        imports = []
        
        # Get language patterns
        patterns = {
            "class": {
                "c": r"(?:class|struct)\s+(\w+)(?:\s*:\s*\w+)?\s*{",
                "cpp": r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*{",
                "java": r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
                "go": r"type\s+(\w+)\s+struct\s*{",
                "ruby": r"class\s+(\w+)(?:\s*<\s*\w+)?\s*(?:do|\n|$|{)",
                "php": r"(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
                "rust": r"(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*{"
            },
            "function": {
                "c": r"(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{",
                "cpp": r"(?:virtual\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?{",
                "java": r"(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(?:\{|throws)",
                "go": r"func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*{",
                "ruby": r"(?:def)\s+(\w+)(?:\([^)]*\))?\s*(?:do|\n|$|{)",
                "php": r"(?:function|public function|private function|protected function)\s+(\w+)\s*\([^)]*\)\s*{",
                "rust": r"(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*{"
            }
        }
        
        def find_block_end(start_pos: int) -> int:
            """Find the end position of a code block by matching braces."""
            brace_count = 0
            pos = start_pos
            
            # Find opening brace
            while pos < len(content):
                if content[pos] == '{':
                    brace_count = 1
                    break
                pos += 1
                
            if brace_count == 0:
                return start_pos
                
            pos += 1
            
            # Match braces
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
                
            return pos
        
        # Extract classes
        if language in patterns["class"]:
            pattern = patterns["class"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                stats["classes"] = stats.get("classes", 0) + 1
                class_name = match.group(1)
                start_pos = match.start()
                end_pos = find_block_end(start_pos)
                classes[class_name] = {
                    "line_start": content.count('\n', 0, start_pos) + 1,
                    "line_end": content.count('\n', 0, end_pos) + 1,
                    "methods": []  # Methods will be added later
                }
                
        # Extract functions
        if language in patterns["function"]:
            pattern = patterns["function"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                stats["functions"] = stats.get("functions", 0) + 1
                func_name = match.group(1)
                start_pos = match.start()
                end_pos = find_block_end(start_pos)
                functions[func_name] = {
                    "line_start": content.count('\n', 0, start_pos) + 1,
                    "line_end": content.count('\n', 0, end_pos) + 1
                }
                
        # Build hierarchy
        hierarchy = {
            "file_path": file_path,
            "language": language,
            "imports": imports,
            "classes": classes,
            "functions": functions
        }
        
        return hierarchy, stats
        
    except Exception as e:
        logger.error(f"Error analyzing generic hierarchy: {e}")
        return {}, stats

# Helper functions
def _get_import_text(node: ast.AST) -> str:
    """Get import statement text."""
    if isinstance(node, ast.Import):
        return f"import {', '.join(n.name for n in node.names)}"
    elif isinstance(node, ast.ImportFrom):
        names = ', '.join(n.name for n in node.names)
        return f"from {node.module or ''} import {names}"
    return ""

def _get_decorator_name(node: ast.AST) -> str:
    """Get decorator name."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
    return ""

def _get_parent_class(node: ast.AST) -> Optional[str]:
    """Get parent class name for a method."""
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.ClassDef):
        return parent.name
    return None

def _get_return_annotation(node: ast.FunctionDef) -> Optional[str]:
    """Get return type annotation."""
    if node.returns:
        return ast.unparse(node.returns)
    return None

def _get_function_args(node: ast.FunctionDef) -> List[Dict[str, str]]:
    """Get function arguments with type annotations."""
    args = []
    for arg in node.args.args:
        arg_info = {"name": arg.arg}
        if arg.annotation:
            arg_info["type"] = ast.unparse(arg.annotation)
        args.append(arg_info)
    return args

def _get_node_text(node: Any, source: str) -> str:
    """Get text from a tree-sitter node."""
    try:
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source[start_byte:end_byte]
    except Exception:
        return ""

def _get_class_name(node: Any, source: str) -> Optional[str]:
    """Get class name from a tree-sitter node."""
    try:
        for child in node.children:
            if child.type == "identifier":
                return _get_node_text(child, source)
        return None
    except Exception:
        return None

def _get_extends_class(node: Any, source: str) -> Optional[str]:
    """Get extended class name."""
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        return _get_node_text(extends_child, source)
        return None
    except Exception:
        return None

def _get_implements_interfaces(node: Any, source: str) -> List[str]:
    """Get implemented interface names."""
    interfaces = []
    try:
        for child in node.children:
            if child.type == "implements_clause":
                for impl_child in child.children:
                    if impl_child.type == "identifier":
                        interfaces.append(_get_node_text(impl_child, source))
        return interfaces
    except Exception:
        return []

def _get_method_name(node: Any, source: str) -> Optional[str]:
    """Get method name from a tree-sitter node."""
    try:
        for child in node.children:
            if child.type == "property_identifier":
                return _get_node_text(child, source)
        return None
    except Exception:
        return None

def _is_static_method(node: Any) -> bool:
    """Check if method is static."""
    try:
        for child in node.children:
            if child.type == "static" or child.type == "decorator" and "static" in _get_node_text(child, ""):
                return True
        return False
    except Exception:
        return False

def _is_async_method(node: Any) -> bool:
    """Check if method is async."""
    try:
        for child in node.children:
            if child.type == "async":
                return True
        return False
    except Exception:
        return False

def _get_ts_return_type(node: Any, source: str) -> Optional[str]:
    """Get TypeScript return type."""
    try:
        for child in node.children:
            if child.type == "type_annotation":
                return _get_node_text(child, source).strip(": ")
        return None
    except Exception:
        return None

def _get_interface_name(node: Any, source: str) -> Optional[str]:
    """Get interface name from a tree-sitter node."""
    try:
        for child in node.children:
            if child.type == "identifier":
                return _get_node_text(child, source)
        return None
    except Exception:
        return None

def _get_extends_interfaces(node: Any, source: str) -> List[str]:
    """Get extended interface names."""
    interfaces = []
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        interfaces.append(_get_node_text(extends_child, source))
        return interfaces
    except Exception:
        return []

def _get_interface_properties(node: Any, source: str) -> List[Dict[str, str]]:
    """Get interface property definitions."""
    properties = []
    try:
        for child in node.children:
            if child.type == "interface_body":
                for prop in child.children:
                    if prop.type == "property_signature":
                        prop_info = {
                            "name": _get_node_text(prop.child_by_field_name("name"), source),
                            "type": _get_node_text(prop.child_by_field_name("type"), source).strip(": ")
                        }
                        properties.append(prop_info)
        return properties
    except Exception:
        return []

def usage_example() -> None:
    """Example usage of code hierarchy analysis."""
    # Example Python file with inheritance
    python_content = textwrap.dedent('''
        from abc import ABC, abstractmethod
        
        class Animal(ABC):
            def __init__(self, name: str):
                self.name = name
                
            @abstractmethod
            def make_sound(self) -> str:
                pass
                
        class Dog(Animal):
            def make_sound(self) -> str:
                return "Woof!"
                
        class Cat(Animal):
            def make_sound(self) -> str:
                return "Meow!"
    ''').strip()
    
    # Save to temp file
    with open('temp.py', 'w') as f:
        f.write(python_content)
        
    # Analyze hierarchy
    hierarchy, stats = analyze_code_hierarchy('temp.py')
    
    print("Code Hierarchy:")
    print("\nClasses:")
    for class_name, info in hierarchy.get("classes", {}).items():
        print(f"\n{class_name}:")
        if info.get("parent"):
            print(f"  Inherits from: {info['parent']}")
        if info.get("methods"):
            print("  Methods:")
            for method in info["methods"]:
                print(f"    - {method}")
                
    print("\nStatistics:")
    print(f"Classes: {stats.get('classes', 0)}")
    print(f"Methods: {stats.get('methods', 0)}")
    print(f"Abstract Methods: {stats.get('abstract_methods', 0)}")
    
    # Cleanup
    import os
    os.remove('temp.py')

if __name__ == "__main__":
    print("Running code hierarchy analysis example...")
    usage_example()
    print("Done!")

def build_code_hierarchy(blocks: List[Dict[str, Any]], source: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build a hierarchy of code blocks."""
    stats = initialize_stats_dict(source=source, output_dir=output_dir)
    # TODO: Implement code hierarchy building
    return stats 