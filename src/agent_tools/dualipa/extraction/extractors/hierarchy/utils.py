"""
Utility functions for code hierarchy analysis.

This module provides common utility functions used across the hierarchy analysis modules.

Key Features:
1. Usage examples and demonstrations
2. Testing utilities
3. Common helper functions

Dependencies:
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- typing: For type hints (https://docs.python.org/3/library/typing.html)
- textwrap: For text formatting (https://docs.python.org/3/library/textwrap.html)
"""

import os
import textwrap
from pathlib import Path
from typing import Dict, Any


def usage_example() -> None:
    """Example usage of code hierarchy analysis."""
    # Import at function scope to avoid circular imports
    from .core import analyze_code_hierarchy
    
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
    temp_file = 'temp_hierarchy_example.py'
    with open(temp_file, 'w') as f:
        f.write(python_content)
        
    # Analyze hierarchy
    hierarchy, stats = analyze_code_hierarchy(temp_file)
    
    print("Code Hierarchy:")
    print("\nClasses:")
    for class_name, info in hierarchy.get("classes", {}).items():
        print(f"\n{class_name}:")
        if info.get("bases"):
            print(f"  Inherits from: {info['bases']}")
        if info.get("methods"):
            print("  Methods:")
            for method in info["methods"]:
                print(f"    - {method['name']}")
                
    print("\nStatistics:")
    print(f"Classes: {stats.get('classes', 0)}")
    print(f"Functions: {stats.get('functions', 0)}")
    
    # Cleanup
    os.remove(temp_file)


def format_hierarchy_summary(hierarchy: Dict[str, Any]) -> str:
    """
    Format a summary of the hierarchy information.
    
    Args:
        hierarchy: Hierarchy dictionary
        
    Returns:
        Formatted summary string
    """
    file_path = hierarchy.get("file_path", "Unknown")
    language = hierarchy.get("language", "Unknown")
    
    classes = hierarchy.get("classes", {})
    functions = hierarchy.get("functions", {})
    imports = hierarchy.get("imports", [])
    
    summary = [
        f"File: {Path(file_path).name}",
        f"Language: {language}",
        f"Classes: {len(classes)}",
        f"Functions: {len(functions)}",
        f"Imports: {len(imports)}"
    ]
    
    # Add class details
    if classes:
        summary.append("\nClasses:")
        for class_name, info in classes.items():
            methods = len(info.get("methods", []))
            summary.append(f"  - {class_name} ({methods} methods)")
    
    return "\n".join(summary)


if __name__ == "__main__":
    print("Running code hierarchy analysis example...")
    usage_example()
    print("Done!")