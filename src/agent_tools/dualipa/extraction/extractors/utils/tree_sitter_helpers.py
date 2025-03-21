"""
Tree-sitter helper functions for DuaLipa.

This module provides helper functions for working with tree-sitter nodes
and extracting information from the syntax tree.

Key Features:
1. Node text extraction
2. Type information extraction
3. Class relationship extraction
4. Interface property extraction
5. Node property checking

Dependencies:
- tree-sitter: For AST parsing
- loguru: For logging

Documentation Links:
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Tree-sitter Python: https://tree-sitter.github.io/py-tree-sitter/

Related Files:
- tree_sitter_utils.py: Main tree-sitter initialization
- js_ts_extractor.py: JavaScript/TypeScript extraction
"""

from typing import Any, Dict, List, Optional
from loguru import logger
import re

def get_node_text(node: Any, source: str) -> str:
    """
    Get text from a tree-sitter node.
    
    Args:
        node: Tree-sitter node
        source: Source code string
        
    Returns:
        Text contained in the node
        
    Example:
        ```python
        text = get_node_text(function_node, source_code)
        # Returns "function add(a, b) { return a + b; }"
        ```
    """
    try:
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source[start_byte:end_byte]
    except Exception as e:
        logger.error(f"Error getting node text: {e}")
        return ""

def get_block_content(content: str, match: re.Match) -> str:
    """
    Get block content from regex match, handling nested braces.
    
    Args:
        content: Source code string
        match: Regex match object
        
    Returns:
        The full block content including balanced braces
        
    Example:
        ```python
        match = re.search(r'function\\s+(\\w+)', code)
        block = get_block_content(code, match)
        # Returns "function add(a, b) { return a + b; }"
        ```
    """
    try:
        # Find block end
        start = match.end()
        brace_count = 1
        end = start
        
        while end < len(content) and brace_count > 0:
            if content[end] == '{':
                brace_count += 1
            elif content[end] == '}':
                brace_count -= 1
            end += 1
            
        return content[match.start():end]
        
    except Exception as e:
        logger.error(f"Error extracting block content: {e}")
        return match.group(0)

def is_async_node(node: Any) -> bool:
    """
    Check if node is an async function/method.
    
    Args:
        node: Tree-sitter node
        
    Returns:
        True if node has 'async' modifier
        
    Example:
        ```python
        if is_async_node(node):
            # Handle async function
        ```
    """
    try:
        for child in node.children:
            if child.type == "async":
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking async status: {e}")
        return False

def is_static_node(node: Any) -> bool:
    """
    Check if node is static (for class methods).
    
    Args:
        node: Tree-sitter node
        
    Returns:
        True if node has 'static' modifier
        
    Example:
        ```python
        if is_static_node(node):
            # Handle static method
        ```
    """
    try:
        for child in node.children:
            if child.type == "static":
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking static status: {e}")
        return False

def get_ts_return_type(node: Any, source: str) -> Optional[str]:
    """
    Get TypeScript return type annotation.
    
    Args:
        node: Tree-sitter node
        source: Source code string
        
    Returns:
        Return type string if present, None otherwise
        
    Example:
        ```python
        return_type = get_ts_return_type(node, source)
        # Returns "number" for "function add(a: number, b: number): number"
        ```
    """
    try:
        for child in node.children:
            if child.type == "type_annotation":
                return get_node_text(child, source).strip(": ")
        return None
    except Exception as e:
        logger.error(f"Error getting return type: {e}")
        return None

def get_extends_class(node: Any, source: str) -> Optional[str]:
    """
    Get extended class name for class declaration.
    
    Args:
        node: Tree-sitter node for class declaration
        source: Source code string
        
    Returns:
        Name of extended class if present, None otherwise
        
    Example:
        ```python
        parent_class = get_extends_class(class_node, source)
        # Returns "BaseClass" for "class Child extends BaseClass"
        ```
    """
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        return get_node_text(extends_child, source)
        return None
    except Exception as e:
        logger.error(f"Error getting extended class: {e}")
        return None

def get_implements_interfaces(node: Any, source: str) -> List[str]:
    """
    Get implemented interface names for class declaration.
    
    Args:
        node: Tree-sitter node for class declaration
        source: Source code string
        
    Returns:
        List of implemented interface names
        
    Example:
        ```python
        interfaces = get_implements_interfaces(class_node, source)
        # Returns ["Serializable", "Cloneable"] for "class Example implements Serializable, Cloneable"
        ```
    """
    interfaces = []
    try:
        for child in node.children:
            if child.type == "implements_clause":
                for impl_child in child.children:
                    if impl_child.type == "identifier":
                        interfaces.append(get_node_text(impl_child, source))
        return interfaces
    except Exception as e:
        logger.error(f"Error getting implemented interfaces: {e}")
        return []

def get_extends_interfaces(node: Any, source: str) -> List[str]:
    """
    Get extended interface names for interface declaration.
    
    Args:
        node: Tree-sitter node for interface declaration
        source: Source code string
        
    Returns:
        List of extended interface names
        
    Example:
        ```python
        interfaces = get_extends_interfaces(interface_node, source)
        # Returns ["BaseInterface"] for "interface Example extends BaseInterface"
        ```
    """
    interfaces = []
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        interfaces.append(get_node_text(extends_child, source))
        return interfaces
    except Exception as e:
        logger.error(f"Error getting extended interfaces: {e}")
        return []

def get_interface_properties(node: Any, source: str) -> List[Dict[str, str]]:
    """
    Get interface property definitions.
    
    Args:
        node: Tree-sitter node for interface declaration
        source: Source code string
        
    Returns:
        List of property dictionaries with name and type
        
    Example:
        ```python
        properties = get_interface_properties(interface_node, source)
        # Returns [{"name": "id", "type": "number"}, {"name": "name", "type": "string"}]
        ```
    """
    properties = []
    try:
        for child in node.children:
            if child.type == "interface_body":
                for prop in child.children:
                    if prop.type == "property_signature":
                        name_node = prop.child_by_field_name("name")
                        type_node = prop.child_by_field_name("type")
                        if name_node and type_node:
                            prop_info = {
                                "name": get_node_text(name_node, source),
                                "type": get_node_text(type_node, source).strip(": ")
                            }
                            properties.append(prop_info)
        return properties
    except Exception as e:
        logger.error(f"Error getting interface properties: {e}")
        return []

def usage_example() -> None:
    """Example usage of tree-sitter helper functions."""
    from tree_sitter import Parser
    from tree_sitter_language_pack import get_parser
    
    # Sample TypeScript code
    ts_code = """
    interface Person {
        id: number;
        name: string;
    }
    
    class Employee extends BaseEmployee implements Person {
        static count: number = 0;
        id: number;
        name: string;
        
        constructor(id: number, name: string) {
            super();
            this.id = id;
            this.name = name;
            Employee.count++;
        }
        
        async fetchDetails(): Promise<any> {
            return { id: this.id, name: this.name };
        }
    }
    """
    
    # Parse the code
    parser = get_parser("typescript")
    if not parser:
        print("Failed to get TypeScript parser")
        return
        
    tree = parser.parse(bytes(ts_code, "utf8"))
    if not tree:
        print("Failed to parse TypeScript code")
        return
        
    # Find interface and class declarations
    for node in tree.root_node.children:
        if node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = get_node_text(name_node, ts_code)
                print(f"Interface: {name}")
                
                # Get extended interfaces
                extends = get_extends_interfaces(node, ts_code)
                if extends:
                    print(f"  Extends: {', '.join(extends)}")
                    
                # Get properties
                properties = get_interface_properties(node, ts_code)
                print(f"  Properties:")
                for prop in properties:
                    print(f"    {prop['name']}: {prop['type']}")
                    
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = get_node_text(name_node, ts_code)
                print(f"Class: {name}")
                
                # Get extended class
                extends = get_extends_class(node, ts_code)
                if extends:
                    print(f"  Extends: {extends}")
                    
                # Get implemented interfaces
                implements = get_implements_interfaces(node, ts_code)
                if implements:
                    print(f"  Implements: {', '.join(implements)}")
                    
                # Find methods
                for child in node.children:
                    if child.type == "class_body":
                        for method in child.children:
                            if method.type == "method_definition":
                                method_name = get_node_text(method.child_by_field_name("name"), ts_code)
                                print(f"  Method: {method_name}")
                                print(f"    Static: {is_static_node(method)}")
                                print(f"    Async: {is_async_node(method)}")
                                
                                # Get return type
                                return_type = get_ts_return_type(method, ts_code)
                                if return_type:
                                    print(f"    Return type: {return_type}")

if __name__ == "__main__":
    usage_example()