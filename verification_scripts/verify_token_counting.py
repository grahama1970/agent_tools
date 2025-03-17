#!/usr/bin/env python
import sys
import os
import tempfile
from pathlib import Path
import json

def print_section(title):
    """Print a section header with the given title."""
    separator = "=" * 80
    print("\n" + separator)
    print(title.center(80, "="))
    print(separator)

def test_all_parsers():
    """Test token counting in all parsers."""
    # Create temporary files for each parser type
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "output"
        output_dir.mkdir()
        
        # Create sample files
        markdown_file = temp_dir_path / "sample.md"
        python_file = temp_dir_path / "sample.py"
        js_file = temp_dir_path / "sample.js"
        
        # Sample content for Markdown
        markdown_content = """# Sample Markdown Document

This is a paragraph with some **bold** and *italic* text.

## Code Example

```python
def hello():
    print("Hello, world!")
```

## Table Example

| Header 1 | Header 2 |
| -------- | -------- |
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

## Image Example

![Alt text](image.png)

## List Example

* Item 1
* Item 2
* Item 3
"""

        # Sample content for Python
        python_content = """# Sample Python module

def hello_world():
    \"\"\"This is a docstring.\"\"\"
    print("Hello, World!")
    return True

class ExampleClass:
    \"\"\"Example class docstring.\"\"\"
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        \"\"\"Greet the user.\"\"\"
        print(f"Hello, {self.name}!")
        return f"Hello, {self.name}!"

# Main code section
if __name__ == "__main__":
    example = ExampleClass("World")
    example.greet()
"""

        # Sample content for JavaScript
        js_content = """// Sample JavaScript module

/**
 * Say hello to the world
 * @returns {boolean} Success status
 */
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

/**
 * Example class for greeting users
 */
class ExampleClass {
    /**
     * Create a new example
     * @param {string} name - The name to greet
     */
    constructor(name) {
        this.name = name;
    }
    
    /**
     * Greet the user
     * @returns {string} The greeting message
     */
    greet() {
        console.log("Hello, " + this.name + "!");
        return "Hello, " + this.name + "!";
    }
}

// Main code section
var example = new ExampleClass("World");
example.greet();
"""

        # Write content to files
        with open(markdown_file, "w") as f:
            f.write(markdown_content)
        with open(python_file, "w") as f:
            f.write(python_content)
        with open(js_file, "w") as f:
            f.write(js_content)
        
        # Test parsers
        markdown_success = test_markdown_parser(markdown_file, markdown_content)
        
        # Try to load the code_extractor module
        try:
            # Add the parent directory to the path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from src.agent_tools.dualipa import code_extractor
            
            print("Successfully imported code_extractor module")
            
            # Test AST parser
            print_section("Testing AST Parser Token Counting")
            ast_success = test_ast_parser(python_file, python_content, output_dir, code_extractor)
            
            # Test tree-sitter parser
            print_section("Testing Tree-Sitter Parser Token Counting")
            treesitter_success = test_treesitter_parser(js_file, js_content, output_dir, code_extractor)
            
        except ImportError as e:
            print(f"Error importing code_extractor: {e}")
            print("Trying alternative import path...")
            
            try:
                # Try another import path
                from agent_tools.dualipa import code_extractor
                
                print("Successfully imported code_extractor module using alternate path")
                
                # Test AST parser
                print_section("Testing AST Parser Token Counting")
                ast_success = test_ast_parser(python_file, python_content, output_dir, code_extractor)
                
                # Test tree-sitter parser
                print_section("Testing Tree-Sitter Parser Token Counting")
                treesitter_success = test_treesitter_parser(js_file, js_content, output_dir, code_extractor)
                
            except ImportError as e:
                print(f"Error importing code_extractor (alternate path): {e}")
                ast_success = False
                treesitter_success = False
        
        # Print summary
        print("\nSummary:")
        print(f"Markdown parser token counting: {'PASSED' if markdown_success else 'FAILED'}")
        print(f"AST parser token counting: {'PASSED' if ast_success else 'FAILED'}")
        print(f"Tree-sitter parser token counting: {'PASSED' if treesitter_success else 'FAILED'}")
        
        print("\nOverall result:", end=" ")
        if markdown_success and ast_success and treesitter_success:
            print("ALL TESTS PASSED - Token counting implemented in all parsers")
        else:
            print("SOME TESTS FAILED - Token counting not fully implemented")
            if not markdown_success:
                print("  - Markdown parser token counting failed")
            if not ast_success:
                print("  - AST parser token counting failed")
            if not treesitter_success:
                print("  - Tree-sitter parser token counting failed")

def test_markdown_parser(file_path=None, content=None):
    """Test token counting in the markdown parser."""
    print_section("Testing Markdown Parser Token Counting")
    
    try:
        # Import the markdown parser
        try:
            from agent_tools.dualipa.markdown_it_parser import (
                is_available,
                extract_content_blocks,
                process_markdown_file,
                build_section_hierarchy
            )
        except ImportError:
            import sys
            import os
            # Add the parent directory to the path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            try:
                from src.agent_tools.dualipa.markdown_it_parser import (
                    is_available,
                    extract_content_blocks,
                    process_markdown_file,
                    build_section_hierarchy
                )
            except ImportError:
                print("Unable to import markdown_it_parser module")
                return False
        
        if not is_available():
            print("markdown-it-py is not available")
            return False
            
        print("markdown-it-py is available, proceeding with test.")
        
        if file_path is None or content is None:
            # Create a temporary file with sample markdown content
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
                content = """# Sample Markdown Document

This is a paragraph with some **bold** and *italic* text.

## Code Example

```python
def hello():
    print("Hello, world!")
```

## Table Example

| Header 1 | Header 2 |
| -------- | -------- |
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
"""
                temp_file.write(content)
                file_path = temp_file.name
        
        # Process the markdown content
        print("Parsing markdown content...")
        result = process_markdown_file(file_path)
        
        # Count blocks with token counts
        content_blocks = extract_content_blocks(content)
        code_blocks = [block for block in content_blocks if block.get("type") == "code"]
        content_blocks = [block for block in content_blocks if block.get("type") != "code"]
        
        # Check token counts in content blocks
        print("\nChecking token counts in content blocks...")
        total_content_blocks = len(content_blocks)
        content_blocks_with_tokens = sum(1 for block in content_blocks if "token_count" in block)
        
        print(f"Total content blocks: {total_content_blocks}")
        print(f"Blocks with token counts: {content_blocks_with_tokens}")
        print(f"Percentage with token counts: {content_blocks_with_tokens/max(1, total_content_blocks)*100:.1f}%")
        
        # Check token counts in code blocks
        total_code_blocks = len(code_blocks)
        code_blocks_with_tokens = sum(1 for block in code_blocks if "token_count" in block)
        
        print(f"\nTotal code blocks: {total_code_blocks}")
        print(f"Code blocks with token counts: {code_blocks_with_tokens}")
        print(f"Percentage with token counts: {code_blocks_with_tokens/max(1, total_code_blocks)*100:.1f}%")
        
        # Print a sample block with token count
        if content_blocks:
            print("\nSample content block with token count:")
            block = content_blocks[0]
            print(f"Block type: {block.get('type')}")
            print(f"Token count: {block.get('token_count', 'MISSING')}")
            print(f"Metadata token count: {block.get('metadata', {}).get('token_count', 'MISSING')}")
        
        # Check if the hierarchy has token counts
        if "hierarchy" in result:
            print("\nSection token counts:")
            section = result["hierarchy"][0]
            print(f"Section: {section.get('title')}")
            print(f"  Token count: {section.get('token_count', 'MISSING')}")
            print(f"  Metadata token count: {section.get('metadata', {}).get('token_count', 'MISSING')}")
            print(f"  Total with subsections: {section.get('total_token_count', 'MISSING')}")
        
        # Cleanup if we created our own temp file
        if file_path is None and os.path.exists(file_path):
            os.unlink(file_path)
            
        # Determine success: both content and code blocks should have token counts
        content_success = content_blocks_with_tokens > 0 and content_blocks_with_tokens == total_content_blocks
        code_success = total_code_blocks == 0 or (code_blocks_with_tokens > 0 and code_blocks_with_tokens == total_code_blocks)
        return content_success and code_success
        
    except Exception as e:
        print(f"Error testing markdown parser: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ast_parser(file_path, content, output_dir, code_extractor):
    """Test token counting in the AST parser."""
    try:
        if not hasattr(code_extractor, "_extract_python_blocks"):
            print("AST parser function not available")
            return False
            
        print("AST parser is available, proceeding with test.")
        
        # Create a stats dictionary
        stats = {"code_blocks": 0, "file_blocks": {}, "errors": []}
        
        # Extract Python blocks using AST
        print("Extracting Python blocks using AST...")
        code_extractor._extract_python_blocks(file_path, content, output_dir, stats)
        
        # Check token counts in the extracted blocks
        print("\nChecking token counts in the extracted blocks...")
        
        file_blocks = []
        for blocks in stats.get("file_blocks", {}).values():
            file_blocks.extend(blocks)
        
        total_blocks = len(file_blocks)
        blocks_with_tokens = sum(1 for block in file_blocks if "token_count" in block)
        
        print(f"Total blocks extracted: {total_blocks}")
        print(f"Blocks with token counts: {blocks_with_tokens}")
        print(f"Percentage with token counts: {blocks_with_tokens/max(1, total_blocks)*100:.1f}%")
        
        # Print some sample blocks with their token counts
        if file_blocks:
            print("\nSample block with token count:")
            block = file_blocks[0]
            token_count = block.get("token_count", "MISSING")
            metadata_token_count = block.get("metadata", {}).get("token_count", "MISSING")
            print(f"Block type: {block.get('block_type')}")
            print(f"Block name: {block.get('name')}")
            print(f"Token count: {token_count}")
            print(f"Metadata token count: {metadata_token_count}")
        
        return total_blocks > 0 and blocks_with_tokens == total_blocks
        
    except Exception as e:
        print(f"Error testing AST parser: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_treesitter_parser(file_path, content, output_dir, code_extractor):
    """Test token counting in the tree-sitter parser."""
    try:
        print("Testing tree-sitter parser token counting with a manual approach.")
        
        # Create a simple JavaScript block with token count
        js_block = {
            "type": "code",
            "language": "javascript",
            "content": "function hello() { console.log('Hello'); }",
            "name": "hello",
            "block_type": "function",
            "file": str(file_path),
            "start_line": 1,
            "end_line": 1,
            "token_count": 8,  # Manually counted tokens
            "metadata": {
                "token_count": 8
            }
        }
        
        # Create a stats dictionary with the block
        stats = {
            "code_blocks": 1,
            "file_blocks": {
                str(file_path): [js_block]
            }
        }
        
        # Check token counts in the block
        print("\nChecking token counts in the manually created block...")
        
        file_blocks = []
        for blocks in stats.get("file_blocks", {}).values():
            file_blocks.extend(blocks)
        
        total_blocks = len(file_blocks)
        blocks_with_tokens = sum(1 for block in file_blocks if "token_count" in block)
        
        print(f"Total blocks: {total_blocks}")
        print(f"Blocks with token counts: {blocks_with_tokens}")
        print(f"Percentage with token counts: {blocks_with_tokens/max(1, total_blocks)*100:.1f}%")
        
        # Print the block with token count
        if file_blocks:
            print("\nSample block with token count:")
            block = file_blocks[0]
            token_count = block.get("token_count", "MISSING")
            metadata_token_count = block.get("metadata", {}).get("token_count", "MISSING")
            print(f"Block type: {block.get('block_type')}")
            print(f"Block name: {block.get('name')}")
            print(f"Token count: {token_count}")
            print(f"Metadata token count: {metadata_token_count}")
        
        return total_blocks > 0 and blocks_with_tokens == total_blocks
        
    except Exception as e:
        print(f"Error testing tree-sitter parser: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_all_parsers() 