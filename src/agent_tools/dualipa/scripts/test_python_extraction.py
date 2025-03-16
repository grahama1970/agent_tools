#!/usr/bin/env python
"""
Standalone test for Python code extraction.

This script directly tests the Python code block extraction without loading any 
unnecessary modules. It shows detailed output for human verification of AST extracted data.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
import ast
import re

# Add parent directory to path so we can import the extraction function directly
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

# Function to get full AST details for debugging
def ast_to_dict(node):
    """Convert an AST node to a dict for display"""
    if isinstance(node, ast.AST):
        # Get all fields
        fields = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                # Convert lists of nodes
                fields[field] = [ast_to_dict(item) for item in value]
            else:
                # Convert single nodes
                fields[field] = ast_to_dict(value)
        return {node.__class__.__name__: fields}
    elif isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    else:
        # Return primitive values as is
        return node

def extract_python_blocks(file_path, content, output_directory, debug=False):
    """
    Extract Python code blocks using AST.
    
    Args:
        file_path: Path to the source file (can be real or notional)
        content: Python code content as string
        output_directory: Directory where extracted blocks will be saved
        debug: Whether to save debug information
    
    Returns:
        List of dictionaries with block information and AST data
    """
    try:
        # Parse the Python code
        tree = ast.parse(content)
        
        # Create output directory for Python blocks
        blocks_dir = Path(output_directory) / "python_blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        blocks = []
        lines = content.split('\n')
        module_name = Path(file_path).stem
        
        # Create a relative path from the file_path for module context
        module_path = Path(file_path)
        relative_path = str(module_path).replace('\\', '/')  # Normalize for all platforms
        
        # Extract module-level imports
        module_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_imports.append({
                        'type': 'import',
                        'name': name.name,
                        'alias': name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    module_imports.append({
                        'type': 'from_import',
                        'module': node.module,
                        'name': name.name,
                        'alias': name.asname
                    })
        
        # Extract function and class definitions
        for i, node in enumerate(ast.iter_child_nodes(tree)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block_type = 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'
                
                # Get function/class name and docstring
                block_name = node.name
                docstring = ast.get_docstring(node)
                
                # Get start and end line
                start_line = node.lineno - 1  # ast is 1-indexed, we want 0-indexed
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else None
                
                # If end_line is not available directly from AST (older Python versions)
                if end_line is None:
                    # Find the end by looking for the indentation level
                    indent_level = None
                    for j, line in enumerate(lines[start_line:], start_line):
                        if j == start_line:
                            # First line, get indentation level
                            indent_match = re.match(r'^(\s*)', line)
                            indent_level = len(indent_match.group(1)) if indent_match else 0
                        elif line.strip() and not line.strip().startswith('#'):
                            # If this line has less indentation than the function/class definition
                            # and it's not a comment, it's outside our block
                            current_indent = len(re.match(r'^(\s*)', line).group(1)) if re.match(r'^(\s*)', line) else 0
                            if current_indent <= indent_level:
                                end_line = j - 1
                                break
                    
                    # If we couldn't find the end, use the last line
                    if end_line is None:
                        end_line = len(lines) - 1
                
                # Extract the code for this block
                block_code = '\n'.join(lines[start_line:end_line+1])
                
                # Process decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        # Handle decorator with arguments: @decorator(arg)
                        decorators.append(decorator.func.id)
                    # Note: We're not handling more complex decorators here
                
                # Extract class inheritance if it's a class
                bases = []
                if block_type == 'class':
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                
                block_info = {
                    'type': block_type,
                    'name': block_name,
                    'file_path': relative_path,
                    'docstring': docstring or "",
                    'start_line': start_line,
                    'end_line': end_line,
                    'code': block_code,
                    'imports': module_imports,
                    'decorators': decorators,
                    'bases': bases
                }
                
                # Extract function-specific information
                if block_type == 'function':
                    parameters = []
                    parameter_details = []
                    
                    args = node.args
                    
                    # Process positional-only arguments (Python 3.8+)
                    if hasattr(args, 'posonlyargs'):
                        for i, arg in enumerate(args.posonlyargs):
                            param_name = arg.arg
                            parameters.append(param_name)
                            
                            # Get type annotation if available
                            annotation = None
                            if arg.annotation:
                                annotation = _get_annotation_str(arg.annotation)
                            
                            # Check if this parameter has a default value
                            has_default = i >= len(args.posonlyargs) - len(args.defaults)
                            default_value = None
                            if has_default:
                                default_idx = i - (len(args.posonlyargs) - len(args.defaults))
                                default_node = args.defaults[default_idx]
                                default_value = _get_default_value(default_node)
                            
                            parameter_details.append({
                                'name': param_name,
                                'has_default': has_default,
                                'default_value': default_value,
                                'annotation': annotation
                            })
                    
                    # Process positional arguments
                    for i, arg in enumerate(args.args):
                        param_name = arg.arg
                        parameters.append(param_name)
                        
                        # Get type annotation if available
                        annotation = None
                        if arg.annotation:
                            annotation = _get_annotation_str(arg.annotation)
                        
                        # Check if this parameter has a default value
                        has_default = i >= len(args.args) - len(args.defaults)
                        default_value = None
                        if has_default:
                            default_idx = i - (len(args.args) - len(args.defaults))
                            default_node = args.defaults[default_idx]
                            default_value = _get_default_value(default_node)
                        
                        parameter_details.append({
                            'name': param_name,
                            'has_default': has_default,
                            'default_value': default_value,
                            'annotation': annotation
                        })
                    
                    # Process keyword-only arguments
                    for i, arg in enumerate(args.kwonlyargs):
                        param_name = arg.arg
                        parameters.append(param_name)
                        
                        # Get type annotation if available
                        annotation = None
                        if arg.annotation:
                            annotation = _get_annotation_str(arg.annotation)
                        
                        # Check if this parameter has a default value
                        has_default = i < len(args.kw_defaults) and args.kw_defaults[i] is not None
                        default_value = None
                        if has_default:
                            default_node = args.kw_defaults[i]
                            default_value = _get_default_value(default_node)
                        
                        parameter_details.append({
                            'name': param_name,
                            'has_default': has_default,
                            'default_value': default_value,
                            'annotation': annotation
                        })
                    
                    # Handle *args
                    if args.vararg:
                        param_name = f"*{args.vararg.arg}"
                        parameters.append(param_name)
                        
                        # Get type annotation if available
                        annotation = None
                        if args.vararg.annotation:
                            annotation = _get_annotation_str(args.vararg.annotation)
                        
                        parameter_details.append({
                            'name': param_name,
                            'has_default': False,
                            'default_value': None,
                            'annotation': annotation
                        })
                    
                    # Handle **kwargs
                    if args.kwarg:
                        param_name = f"**{args.kwarg.arg}"
                        parameters.append(param_name)
                        
                        # Get type annotation if available
                        annotation = None
                        if args.kwarg.annotation:
                            annotation = _get_annotation_str(args.kwarg.annotation)
                        
                        parameter_details.append({
                            'name': param_name,
                            'has_default': False,
                            'default_value': None,
                            'annotation': annotation
                        })
                    
                    # Get return type annotation
                    returns = None
                    if node.returns:
                        returns = _get_annotation_str(node.returns)
                    
                    block_info['parameters'] = parameters
                    block_info['parameter_details'] = parameter_details
                    block_info['returns'] = returns
                    
                    # Extract local variables defined in the function
                    local_variables = []
                    
                    def find_variables(node):
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, ast.Assign):
                                for target in child.targets:
                                    if isinstance(target, ast.Name):
                                        local_variables.append(target.id)
                            # Recursively search for variables in child nodes
                            # Skip nested function/class definitions
                            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                find_variables(child)
                    
                    find_variables(node)
                    block_info['local_variables'] = local_variables
                
                # Create file name with index to avoid duplicates
                base_name = f"{module_name}_{block_name}_{i+1}.py"
                block_file = blocks_dir / base_name
                
                # Create a useful comment header with block information
                header = [
                    f"# Original file: {relative_path}",
                    f"# Block type: {block_type}",
                    f"# Block name: {block_name}",
                ]
                
                # Add docstring as comment if available
                if docstring:
                    header.append(f"# Docstring: {docstring}")
                
                # Add decorators if available
                if decorators:
                    header.append(f"# Decorators: {', '.join(decorators)}")
                
                # Add inheritance information if it's a class
                if block_type == 'class' and bases:
                    header.append(f"# Inherits from: {', '.join(bases)}")
                
                # Add parameter information if it's a function
                if block_type == 'function':
                    if parameters:
                        params_formatted = []
                        for param in parameter_details:
                            param_str = param['name']
                            if param.get('annotation'):
                                param_str += f": {param['annotation']}"
                            if param['has_default']:
                                param_str += f" = {param['default_value']}"
                            params_formatted.append(param_str)
                        
                        header.append(f"# Parameters: {', '.join(params_formatted)}")
                    
                    if returns:
                        header.append(f"# Returns: {returns}")
                
                # Add import information
                if module_imports:
                    header.append("\n# Required imports:")
                    for imp in module_imports:
                        if imp['type'] == 'import':
                            imp_str = f"# import {imp['name']}"
                            if imp['alias']:
                                imp_str += f" as {imp['alias']}"
                            header.append(imp_str)
                        else:  # from_import
                            imp_str = f"# from {imp['module']} import {imp['name']}"
                            if imp['alias']:
                                imp_str += f" as {imp['alias']}"
                            header.append(imp_str)
                
                header.append("\n")
                
                # Write the block to a file with header
                with open(block_file, 'w') as f:
                    f.write('\n'.join(header) + '\n')
                    f.write(block_code)
                
                block_info['file'] = str(block_file)
                blocks.append(block_info)
        
        # Save AST data for debugging if requested
        ast_data = None
        if debug:
            ast_data = _ast_to_dict(tree)
            debug_file = Path(output_directory) / "ast_debug.json"
            with open(debug_file, 'w') as f:
                json.dump(ast_data, f, indent=2)
        
        return blocks, ast_data
    
    except Exception as e:
        print(f"Error extracting Python blocks: {str(e)}")
        traceback.print_exc()
        return [], None

def _get_annotation_str(annotation_node):
    """Extract the string representation of a type annotation."""
    if isinstance(annotation_node, ast.Name):
        return annotation_node.id
    elif isinstance(annotation_node, ast.Attribute):
        return _get_attribute_str(annotation_node)
    elif isinstance(annotation_node, ast.Subscript):
        if isinstance(annotation_node.value, ast.Name):
            base = annotation_node.value.id
            slice_str = _get_slice_str(annotation_node.slice)
            return f"{base}[{slice_str}]"
        elif isinstance(annotation_node.value, ast.Attribute):
            base = _get_attribute_str(annotation_node.value)
            slice_str = _get_slice_str(annotation_node.slice)
            return f"{base}[{slice_str}]"
    return "Any"  # Default fallback

def _get_attribute_str(attr_node):
    """Extract string representation of an attribute."""
    if isinstance(attr_node.value, ast.Name):
        return f"{attr_node.value.id}.{attr_node.attr}"
    elif isinstance(attr_node.value, ast.Attribute):
        return f"{_get_attribute_str(attr_node.value)}.{attr_node.attr}"
    return f"?.{attr_node.attr}"  # Fallback

def _get_slice_str(slice_node):
    """Extract the string representation of a slice."""
    # Handle different Python versions' AST structures
    if isinstance(slice_node, ast.Index):  # Python < 3.9
        return _get_value_str(slice_node.value)
    elif isinstance(slice_node, ast.Tuple):  # For complex slices like List[int, str]
        elts = []
        for elt in slice_node.elts:
            elts.append(_get_value_str(elt))
        return ", ".join(elts)
    else:  # Python >= 3.9, direct value
        return _get_value_str(slice_node)

def _get_value_str(value_node):
    """Convert a simple AST value node to its string representation."""
    if isinstance(value_node, ast.Name):
        return value_node.id
    elif isinstance(value_node, ast.Constant):
        if isinstance(value_node.value, str):
            return f'"{value_node.value}"'
        return str(value_node.value)
    elif isinstance(value_node, (ast.Num, ast.Str)):  # For older Python versions
        if isinstance(value_node, ast.Str):
            return f'"{value_node.s}"'
        return str(value_node.n)
    elif isinstance(value_node, ast.Tuple):
        elts = []
        for elt in value_node.elts:
            elts.append(_get_value_str(elt))
        return ", ".join(elts)
    return "Any"  # Default fallback

def _get_default_value(default_node):
    """Extract default value as a string representation."""
    if isinstance(default_node, ast.Constant):
        if default_node.value is None:
            return "None"
        elif isinstance(default_node.value, str):
            return f'"{default_node.value}"'
        return str(default_node.value)
    elif isinstance(default_node, ast.Name):
        return default_node.id
    elif isinstance(default_node, ast.List):
        return "[]"  # Simple representation for lists
    elif isinstance(default_node, ast.Dict):
        return "{}"  # Simple representation for dicts
    elif isinstance(default_node, ast.Call):
        if isinstance(default_node.func, ast.Name):
            return f"{default_node.func.id}(...)"
    # For older Python versions
    elif hasattr(ast, 'NameConstant') and isinstance(default_node, ast.NameConstant):
        if default_node.value is None:
            return "None"
        return str(default_node.value)
    elif hasattr(ast, 'Num') and isinstance(default_node, ast.Num):
        return str(default_node.n)
    elif hasattr(ast, 'Str') and isinstance(default_node, ast.Str):
        return f'"{default_node.s}"'
    return "..."  # Generic placeholder for complex defaults

def _ast_to_dict(node):
    """Convert an AST node to a dictionary representation for debugging."""
    if isinstance(node, ast.AST):
        fields = {}
        for name, value in ast.iter_fields(node):
            # Skip lineno and col_offset to keep output cleaner
            if name not in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'ctx'):
                fields[name] = _ast_to_dict(value)
        
        # Add useful metadata for certain node types
        if isinstance(node, ast.FunctionDef):
            # Add parameter information
            fields['parameters'] = [arg.arg for arg in node.args.args]
            
            # Add returns information
            if node.returns:
                fields['returns'] = _get_annotation_str(node.returns)
                
            # Add decorator information
            fields['decorators'] = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    fields['decorators'].append(decorator.id)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                    fields['decorators'].append(decorator.func.id)
                    
        elif isinstance(node, ast.ClassDef):
            # Add inheritance information
            fields['bases'] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    fields['bases'].append(base.id)
                    
        # Add node name for reference
        node_name = node.__class__.__name__
        return {node_name: fields}
    
    elif isinstance(node, list):
        return [_ast_to_dict(i) for i in node]
    
    elif isinstance(node, str) or isinstance(node, int) or node is None:
        return node
    
    else:
        return str(node)

def main():
    """Run the test with sample Python code."""
    # Sample Python code from a multi-function file - now with imports, types, and decorators
    SAMPLE_CODE = """
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

def decorator_function(func):
    \"\"\"A simple decorator.\"\"\"
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@decorator_function
def greet(name: str) -> str:
    \"\"\"Greet a person by name.\"\"\"
    return f"Hello, {name}!"

class BaseClass:
    \"\"\"A base class.\"\"\"
    def __init__(self, name: str):
        self.name = name
        
    def identify(self) -> str:
        return f"I am {self.name}"

class Calculator(BaseClass):
    \"\"\"Simple calculator class.\"\"\"
    def __init__(self, name: str = "Calculator"):
        super().__init__(name)
        
    def add(self, a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        return a + b
        
    def subtract(self, a: float, b: float) -> float:
        \"\"\"Subtract b from a.\"\"\"
        return a - b

@decorator_function
def another_function(x: int, y: int, z: Optional[int] = None) -> int:
    \"\"\"A function with multiple parameters.\"\"\"
    if z is None:
        z = x + y
    result = x * y * z
    return result

def process_items(items: List[str]) -> Dict[str, Any]:
    \"\"\"Process a list of items.\"\"\"
    results = {}
    for i, item in enumerate(items):
        results[item] = i
    return results

if __name__ == "__main__":
    print(greet("World"))
    calc = Calculator()
    print(calc.add(5, 3))
"""

    # Parse arguments
    debug_mode = "--debug" in sys.argv
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i+1 < len(sys.argv):
            output_path = sys.argv[i+1]

    # Create output directory
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        # Use a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
    
    try:
        print("="*80)
        print("TESTING ENHANCED PYTHON CODE EXTRACTION".center(80))
        print("="*80)
        print("\nSample Python Code:")
        print("-"*40)
        print(SAMPLE_CODE)
        print("-"*40)
        
        # Extract blocks
        print("\nExtracting blocks using AST...")
        file_path = Path("src/module/sample.py")  # Use a fake path with directories to show module path
        blocks, ast_data = extract_python_blocks(file_path, SAMPLE_CODE, output_dir, debug=debug_mode)
        
        # Print results
        print(f"\nExtracted {len(blocks)} blocks:")
        for i, block in enumerate(blocks, 1):
            print(f"\n{'='*20} BLOCK {i}: {block['type']} '{block['name']}' {'='*20}")
            print(f"  File: {block['file_path']}")
            print(f"  Lines: {block['start_line']+1}-{block['end_line']+1}")
            print(f"  Docstring: {block['docstring']}")
            
            # Show imports if we're at the first block
            if i == 1:
                if block['imports']:
                    print("\n  Module Imports:")
                    for imp in block['imports']:
                        if imp['type'] == 'import':
                            imp_str = f"    import {imp['name']}"
                            if imp['alias']:
                                imp_str += f" as {imp['alias']}"
                            print(imp_str)
                        else:  # from_import
                            imp_str = f"    from {imp['module']} import {imp['name']}"
                            if imp['alias']:
                                imp_str += f" as {imp['alias']}"
                            print(imp_str)
            
            # Show decorators
            if block['decorators']:
                print("\n  Decorators:")
                for decorator in block['decorators']:
                    print(f"    @{decorator}")
            
            # Show inheritance for classes
            if block['type'] == 'class' and block['bases']:
                print(f"\n  Inherits from: {', '.join(block['bases'])}")
            
            # Show parameter info for functions
            if block['type'] == 'function':
                print("\n  Parameter Information:")
                for param in block['parameter_details']:
                    param_str = f"    - {param['name']}"
                    if param.get('annotation'):
                        param_str += f": {param['annotation']}"
                    
                    if param['has_default']:
                        param_str += f" = {param['default_value']}"
                    else:
                        param_str += " (required)"
                    print(param_str)
                
                # Show return type
                if block.get('returns'):
                    print(f"\n  Returns: {block['returns']}")
                    
                # Show local variables
                if block['local_variables']:
                    print("\n  Local Variables:")
                    for var in block['local_variables']:
                        print(f"    - {var}")
            
            print(f"\n  Code:\n{'-'*40}")
            print(f"{block['code']}")
            print(f"{'-'*40}")
            print(f"\n  Saved to: {block['file']}")
            
            # Print LLM value with enhanced details
            print(f"\n  LLM Value: This block can be used by LLMs to understand:")
            print(f"    - The purpose via docstring: {block['docstring']}")
            if block['type'] == 'function':
                print(f"    - The interface via parameters: {', '.join(block['parameters'])}")
                if block.get('returns'):
                    print(f"    - The return type: {block['returns']}")
            if block['type'] == 'class' and block['bases']:
                print(f"    - The inheritance hierarchy: {', '.join(block['bases'])}")
            if block['decorators']:
                print(f"    - Applied decorators: {', '.join(block['decorators'])}")
            print(f"    - The implementation details via code")
            print(f"    - Module context via imports")
        
        # Print expected vs actual results
        print("\nVERIFICATION RESULTS:")
        print("="*40)
        
        # Expected results - updated with more details
        expected_blocks = [
            {"name": "decorator_function", "type": "function"},
            {"name": "greet", "type": "function", "decorators": ["decorator_function"], "returns": "str"},
            {"name": "BaseClass", "type": "class"},
            {"name": "Calculator", "type": "class", "bases": ["BaseClass"]},
            {"name": "another_function", "type": "function", "decorators": ["decorator_function"]},
            {"name": "process_items", "type": "function", "returns": "Dict[str, Any]"}
        ]
        
        # Check if all expected blocks were found
        all_found = True
        for expected in expected_blocks:
            # Find matching block
            matching_blocks = [b for b in blocks if b["name"] == expected["name"] and b["type"] == expected["type"]]
            
            if matching_blocks:
                block = matching_blocks[0]
                status = "✅ FOUND"
                
                # Verify decorators if specified
                if "decorators" in expected and set(expected["decorators"]) != set(block["decorators"]):
                    status = "⚠️ DECORATORS MISMATCH"
                    all_found = False
                
                # Verify inheritance if it's a class
                if expected["type"] == "class" and "bases" in expected:
                    if set(expected["bases"]) != set(block["bases"]):
                        status = "⚠️ INHERITANCE MISMATCH"
                        all_found = False
                
                # Verify return type if specified
                if "returns" in expected and expected["returns"] != block.get("returns"):
                    status = "⚠️ RETURN TYPE MISMATCH"
                    all_found = False
            else:
                status = "❌ MISSING"
                all_found = False
                
            print(f"{status} {expected['type']} '{expected['name']}'")
                
        # Check if there were any unexpected blocks
        for block in blocks:
            expected = any(b["name"] == block["name"] and b["type"] == block["type"] for b in expected_blocks)
            if not expected:
                print(f"⚠️ UNEXPECTED {block['type']} '{block['name']}' found")
                all_found = False
        
        if all_found:
            print("\n✅ TEST PASSED: All expected blocks extracted correctly!")
        else:
            print("\n❌ TEST FAILED: Some blocks were missing or unexpected blocks were found")
        
        print("\nLLM USE CASES FOR ENHANCED AST DATA:")
        print("="*40)
        print("The enhanced AST information provides the following additional data for LLMs:")
        print("1. Module imports for understanding dependencies")
        print("2. File path and module hierarchy for understanding project structure")
        print("3. Decorators for understanding function behavior modification")
        print("4. Class inheritance for understanding object relationships")
        print("5. Type annotations for understanding data flow and interfaces")
        print("6. Return types for understanding function outputs")
        print("\nThis additional data enables LLMs to:")
        print("- Generate more accurate and context-aware imports")
        print("- Understand and follow project-specific design patterns")
        print("- Apply correct decorators consistent with the codebase")
        print("- Maintain proper type annotations in generated code")
        print("- Respect inheritance hierarchies when extending classes")
        print("- Better understand the overall project structure")
        
        print("="*80)
        
        # If in debug mode, print AST summary
        if debug_mode and ast_data:
            print("\nAST DEBUG INFORMATION SUMMARY:")
            print(f"Full AST data saved to: {output_dir / 'ast_debug.json'}")
            
        return 0 if all_found else 1
        
    finally:
        # Clean up temporary directory if we created one
        if temp_dir:
            temp_dir.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 