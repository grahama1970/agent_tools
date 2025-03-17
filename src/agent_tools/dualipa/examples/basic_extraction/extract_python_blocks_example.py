#!/usr/bin/env python3
"""Example demonstrating correct usage of _extract_python_blocks function."""

import tempfile
from pathlib import Path

# Import the function we want to demonstrate
from agent_tools.dualipa.code_extractor import _extract_python_blocks

def demonstrate_extract_python_blocks():
    """Show how to correctly use _extract_python_blocks with proper parameters."""
    
    # Sample Python code to extract blocks from
    sample_code = """
def hello_world():
    \"\"\"A simple greeting function.\"\"\"
    return "Hello, World!"

class ExampleClass:
    \"\"\"An example class with methods.\"\"\"
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        return f"Hello, {self.name}!"
"""
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert to Path object - THIS IS CRITICAL
        output_dir = Path(temp_dir)
        
        # Create a file path - MUST BE PATH OBJECT NOT STRING
        file_path = Path(temp_dir) / "sample_python.py"
        
        # Initialize stats dictionary with REQUIRED keys
        stats = {
            "code_blocks": 0,  # Counter for blocks
            "errors": [],      # List to store any errors
            "file_blocks": {}  # Dictionary to track blocks by file
        }
        
        # Write sample code to file
        with open(file_path, 'w') as f:
            f.write(sample_code)
        
        # Call the function with proper parameters
        try:
            num_blocks = _extract_python_blocks(
                file_path=file_path,  # Path object, not string
                content=sample_code,  
                output_dir=output_dir,
                stats=stats
            )
            
            print(f"Successfully extracted {num_blocks} Python blocks")
            print(f"Stats: {stats}")
            
            # Verify extracted blocks
            blocks_dir = output_dir / "blocks" / "code" / "python"
            if blocks_dir.exists():
                print("\nExtracted blocks:")
                for block_file in blocks_dir.glob("*.py"):
                    with open(block_file, 'r') as f:
                        content = f.read()
                    print(f"\n--- {block_file.name} ---")
                    print(content[:200] + "..." if len(content) > 200 else content)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    demonstrate_extract_python_blocks() 