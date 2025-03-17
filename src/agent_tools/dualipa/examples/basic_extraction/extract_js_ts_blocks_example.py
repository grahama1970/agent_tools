#!/usr/bin/env python3
"""Example demonstrating correct usage of _extract_js_ts_blocks function."""

import tempfile
from pathlib import Path

# Import the function we want to demonstrate
from agent_tools.dualipa.code_extractor import _extract_js_ts_blocks

def demonstrate_extract_js_ts_blocks():
    """Show how to correctly use _extract_js_ts_blocks with proper parameters."""
    
    # Sample JavaScript code with functions and classes
    sample_js = """
// A simple JavaScript function
function calculateSum(a, b) {
  /**
   * Calculates the sum of two numbers
   */
  return a + b;
}

// ES6 class example
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  
  greet() {
    return `Hello, my name is ${this.name} and I am ${this.age} years old.`;
  }
}

// React component example
const Button = ({ onClick, children }) => {
  return (
    <button onClick={onClick} className="button">
      {children}
    </button>
  );
};
"""
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert to Path object - THIS IS CRITICAL
        output_dir = Path(temp_dir)
        
        # Create a file path - MUST BE PATH OBJECT NOT STRING
        file_path = Path(temp_dir) / "sample_javascript.js"
        
        # Initialize stats dictionary with REQUIRED keys
        stats = {
            "code_blocks": 0,  # Counter for blocks
            "errors": [],      # List to store any errors
            "file_blocks": {}  # Dictionary to track blocks by file
        }
        
        # Write sample JavaScript to file
        with open(file_path, 'w') as f:
            f.write(sample_js)
        
        # Call the function with proper parameters
        try:
            num_blocks = _extract_js_ts_blocks(
                file_path=file_path,  # Path object, not string
                content=sample_js,
                output_dir=output_dir,
                stats=stats
            )
            
            print(f"Successfully extracted {num_blocks} JavaScript blocks")
            print(f"Stats: {stats}")
            
            # Verify extracted blocks
            blocks_dir = output_dir / "blocks" / "code" / "javascript"
            if blocks_dir.exists():
                print("\nExtracted blocks:")
                for block_file in blocks_dir.glob("*.js"):
                    with open(block_file, 'r') as f:
                        content = f.read()
                    print(f"\n--- {block_file.name} ---")
                    print(content[:200] + "..." if len(content) > 200 else content)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    demonstrate_extract_js_ts_blocks() 