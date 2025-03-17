#!/usr/bin/env python3
"""Example demonstrating correct usage of _extract_markdown_blocks function."""

import tempfile
from pathlib import Path

# Import the function we want to demonstrate
from agent_tools.dualipa.code_extractor import _extract_markdown_blocks

def demonstrate_extract_markdown_blocks():
    """Show how to correctly use _extract_markdown_blocks with proper parameters."""
    
    # Sample Markdown content with headings and code blocks
    sample_markdown = """
# Main Heading

This is an introduction paragraph.

## First Section

This is content under the first section.

```python
def example_function():
    return "This is a code block"
```

## Second Section

This is content under the second section.

### Subsection

This is content in a subsection.
"""
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert to Path object - THIS IS CRITICAL
        output_dir = Path(temp_dir)
        
        # Create a file path - MUST BE PATH OBJECT NOT STRING
        file_path = Path(temp_dir) / "sample_markdown.md"
        
        # Initialize stats dictionary with REQUIRED keys
        stats = {
            "code_blocks": 0,  # Counter for blocks
            "errors": [],      # List to store any errors
            "file_blocks": {}  # Dictionary to track blocks by file
        }
        
        # Write sample markdown to file
        with open(file_path, 'w') as f:
            f.write(sample_markdown)
        
        # Call the function with proper parameters
        try:
            num_blocks = _extract_markdown_blocks(
                file_path=file_path,  # Path object, not string
                content=sample_markdown,  
                output_dir=output_dir,
                stats=stats
            )
            
            print(f"Successfully extracted {num_blocks} Markdown blocks")
            print(f"Stats: {stats}")
            
            # Verify extracted blocks
            blocks_dir = output_dir / "blocks" / "docs" / "markdown"
            if blocks_dir.exists():
                print("\nExtracted markdown blocks:")
                for block_file in blocks_dir.glob("*.md"):
                    with open(block_file, 'r') as f:
                        content = f.read()
                    print(f"\n--- {block_file.name} ---")
                    print(content[:200] + "..." if len(content) > 200 else content)
            
            # Also check for code blocks extracted from markdown
            code_blocks_dir = output_dir / "blocks" / "code" / "python"
            if code_blocks_dir.exists():
                print("\nExtracted code blocks from markdown:")
                for block_file in code_blocks_dir.glob("*.py"):
                    with open(block_file, 'r') as f:
                        content = f.read()
                    print(f"\n--- {block_file.name} ---")
                    print(content[:200] + "..." if len(content) > 200 else content)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    demonstrate_extract_markdown_blocks() 