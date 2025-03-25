"""
React component extraction utilities for DuaLipa.

This module provides utilities for identifying and extracting React components
from JavaScript and TypeScript files.

Key Features:
1. React component detection
2. JSX/TSX parsing
3. Component property extraction
4. Component structure analysis

Dependencies:
- tree-sitter: For AST parsing
- loguru: For logging
- re: For pattern matching
- uuid: For unique identification

Documentation Links:
- React Documentation: https://reactjs.org/docs/components-and-props.html
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/

Related Files:
- tree_sitter_utils.py: Main tree-sitter initialization
- tree_sitter_helpers.py: Helper functions for tree-sitter
- js_ts_extractor.py: JavaScript/TypeScript extraction
"""

import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_utils import get_parser, extract_js_ts_imports_exports

def is_react_component(content: str) -> bool:
    """
    Check if content contains a React component.
    
    Args:
        content: Source code
        
    Returns:
        True if content contains a React component, False otherwise
        
    Example:
        ```python
        if is_react_component(code):
            # Handle React component
        ```
    """
    patterns = [
        r'import\s+.*?React.*?\s+from\s+[\'"]react[\'"]',
        r'class\s+\w+\s+extends\s+(?:React\.)?Component',
        r'function\s+[A-Z]\w*\s*\([^)]*\)\s*{',
        r'const\s+[A-Z]\w*\s*=\s*(?:React\.)?memo\(',
        r'export\s+default\s+[A-Z]\w+'
    ]
    return any(re.search(pattern, content) for pattern in patterns)

def extract_react_component(file_path: str, content: str, language: str) -> Optional[Dict[str, Any]]:
    """
    Extract a React component as a single block.
    
    Args:
        file_path: Path to source file
        content: Source code
        language: 'javascript' or 'typescript'
        
    Returns:
        Component block dictionary if successful, None otherwise
        
    Example:
        ```python
        component = extract_react_component("Button.jsx", content, "javascript")
        if component:
            # Process React component
        ```
    """
    try:
        # Try to find component name
        name_match = re.search(r'(?:class|function|const)\s+([A-Z]\w*)', content)
        if not name_match:
            return None
            
        component_name = name_match.group(1)
        
        # Try tree-sitter parsing for imports/exports
        tree = None
        imports = []
        exports = []
        
        try:
            parser = get_parser(language)
            if parser:
                tree = parser.parse(bytes(content, "utf8"))
                imports, exports = extract_js_ts_imports_exports(content, tree)
        except Exception as parse_e:
            logger.warning(f"Error parsing React component with tree-sitter: {parse_e}")
            # Fallback to regex for imports/exports
            imports, exports = extract_js_ts_imports_exports(content, None)
        
        # Check if component is exported
        is_exported = False
        export_match = re.search(r'export\s+(?:default\s+)?(?:class|function|const)\s+' + component_name, content)
        if export_match or any(component_name in exp for exp in exports):
            is_exported = True
            
        # Add component-specific export if needed
        component_exports = exports.copy()
        if is_exported and not any(component_name in exp for exp in exports):
            component_exports.append(f"export default {component_name}")
            
        return {
            "uuid": str(uuid.uuid4()),
            "type": "react_component",
            "name": component_name,
            "content": content,  # Keep entire file content for React components
            "line_start": 1,
            "line_end": len(content.splitlines()),
            "metadata": {
                "language": language,
                "framework": "react",
                "extraction_method": "react_detection",
                "extraction_quality": "high",
                "file": file_path,
                "imports": imports,
                "exports": component_exports
            }
        }
    except Exception as e:
        logger.error(f"Error extracting React component: {e}")
        return None

def extract_react_props(content: str, component_name: str) -> List[Dict[str, str]]:
    """
    Extract React component props definitions.
    
    Args:
        content: Source code content
        component_name: Name of the React component
        
    Returns:
        List of component props with their types if available
        
    Example:
        ```python
        props = extract_react_props(content, "Button")
        # Returns [{"name": "onClick", "type": "() => void"}, {"name": "label", "type": "string"}]
        ```
    """
    props = []
    
    # Look for TypeScript Props interface
    interface_pattern = re.compile(r'interface\s+([A-Z]\w*Props)\s*{([^}]*)}', re.DOTALL)
    interface_match = interface_pattern.search(content)
    
    if interface_match:
        interface_name = interface_match.group(1)
        interface_body = interface_match.group(2)
        
        # Extract props from interface
        prop_pattern = re.compile(r'(\w+)(?:\?)?:\s*([^;]+);')
        for prop_match in prop_pattern.finditer(interface_body):
            prop_name = prop_match.group(1)
            prop_type = prop_match.group(2).strip()
            props.append({
                "name": prop_name,
                "type": prop_type
            })
    
    # Look for props destructuring in function components
    if "function" in content or "const" in content:
        # Match function parameter destructuring
        destruc_pattern = re.compile(r'(?:function|const)\s+' + component_name + r'\s*=?\s*\(\s*{([^}]*)}')
        destruc_match = destruc_pattern.search(content)
        
        if destruc_match:
            destruc_body = destruc_match.group(1)
            # Extract prop names
            for prop in re.findall(r'(\w+)(?::\s*([^,]+))?', destruc_body):
                prop_name = prop[0]
                prop_type = prop[1].strip() if len(prop) > 1 and prop[1] else "any"
                
                # Check if this prop is already added from interface
                if not any(p["name"] == prop_name for p in props):
                    props.append({
                        "name": prop_name,
                        "type": prop_type
                    })
    
    return props

def is_jsx_tsx_file(file_path: str) -> bool:
    """
    Check if file is a JSX or TSX file based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file has .jsx or .tsx extension
        
    Example:
        ```python
        if is_jsx_tsx_file("Button.jsx"):
            # Handle JSX file
        ```
    """
    return file_path.lower().endswith(('.jsx', '.tsx'))

def usage_example():
    """Example usage of React component extraction."""
    # Sample React component in TSX
    tsx_content = """
    import React, { useState } from 'react';
    
    interface ButtonProps {
        label: string;
        onClick?: () => void;
        disabled?: boolean;
    }
    
    export function Button({ label, onClick, disabled = false }: ButtonProps) {
        const [isHovered, setIsHovered] = useState(false);
        
        const handleClick = () => {
            if (!disabled && onClick) {
                onClick();
            }
        };
        
        return (
            <button
                onClick={handleClick}
                disabled={disabled}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                style={{ opacity: isHovered ? 0.8 : 1 }}
            >
                {label}
            </button>
        );
    }
    
    export default Button;
    """
    
    # Check if content contains React component
    is_react = is_react_component(tsx_content)
    print(f"Is React component: {is_react}")
    
    if is_react:
        # Extract component
        component = extract_react_component("Button.tsx", tsx_content, "typescript")
        
        if component:
            print(f"Component name: {component['name']}")
            print(f"Component type: {component['type']}")
            print(f"Metadata:")
            for key, value in component['metadata'].items():
                if key not in ("imports", "exports"):
                    print(f"  {key}: {value}")
                    
            # Print imports and exports more concisely
            print(f"  imports: {len(component['metadata']['imports'])} items")
            print(f"  exports: {len(component['metadata']['exports'])} items")
            
            # Extract props
            props = extract_react_props(tsx_content, component['name'])
            print("\nComponent props:")
            for prop in props:
                print(f"  {prop['name']}: {prop['type']}")

if __name__ == "__main__":
    usage_example()