#!/usr/bin/env python3
"""
example_extract.py

A simplified example showing how to extract and structure blocks properly
for frictionless validation.
"""

import json
import uuid

def extract_python_function(code_snippet):
    """
    Extract a Python function and return a properly structured block.
    
    Args:
        code_snippet: Python code containing a function
        
    Returns:
        A JSON-formatted block with all required fields
    """
    # Create a basic block with all required fields
    block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "example_function",  # In real implementation, parse the name
        "content": code_snippet,
        "language": "python",
        "file_path": "snippet.py",
        "start_line": 1,
        "end_line": code_snippet.count('\n') + 1,
        "parent_uuid": None,
        "child_uuids": [],
        "metadata": {
            "language": "python",
            "has_docstring": '"""' in code_snippet,
            "doc_string": extract_docstring(code_snippet) if '"""' in code_snippet else None,
            "arguments": ["arg1", "arg2"],  # In real implementation, parse the arguments
            "returns": ["result"]  # In real implementation, parse the returns
        }
    }
    
    return block


def extract_docstring(code):
    """
    Extract a docstring from Python code.
    
    Args:
        code: Python code containing a docstring
        
    Returns:
        The extracted docstring or None
    """
    if '"""' not in code:
        return None
    
    # Find the start of the docstring (after the first """)
    start = code.find('"""') + 3
    
    # Find the end of the docstring (the next """)
    end = code.find('"""', start)
    if end == -1:
        return None
    
    # Extract the docstring
    docstring = code[start:end].strip()
    return docstring


def extract_markdown_section(markdown):
    """
    Extract a markdown section and return a properly structured block.
    
    Args:
        markdown: Markdown content
        
    Returns:
        A JSON-formatted block with all required fields
    """
    # Create a basic block with all required fields
    block = {
        "uuid": str(uuid.uuid4()),
        "type": "section",
        "name": "Documentation Section",  # In real implementation, parse the title
        "content": markdown,
        "language": "markdown",
        "file_path": "snippet.md",
        "start_line": 1,
        "end_line": markdown.count('\n') + 1,
        "parent_uuid": None,
        "child_uuids": [],
        "metadata": {
            "doc_type": "markdown",
            "section_hierarchy": ["Documentation", "Section"],
            "header_level": 1  # In real implementation, detect header level
        }
    }
    
    return block


if __name__ == "__main__":
    # Example Python function
    python_code = '''def calculate_area(length, width):
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
        
    Returns:
        The area as length * width
    """
    return length * width'''
    
    # Extract and print the block
    python_block = extract_python_function(python_code)
    print("\nPython Function Block:")
    print(json.dumps(python_block, indent=2))
    
    # Example Markdown section
    markdown = '''# API Reference

## Installation

Install using pip:

```bash
pip install dualipa
```'''
    
    # Extract and print the block
    markdown_block = extract_markdown_section(markdown)
    print("\nMarkdown Section Block:")
    print(json.dumps(markdown_block, indent=2))