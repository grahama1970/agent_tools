#!/usr/bin/env python3
"""
Test JavaScript and TypeScript code extraction functionality.

Official Documentation:
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
- tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
"""

import os
from pathlib import Path
import tempfile
import textwrap
import pytest

from agent_tools.dualipa.code_extractor import (
    _extract_js_ts_blocks,
    initialize_stats_dict
)

def test_js_function_extraction():
    """Test extraction of JavaScript functions from code."""
    # Create a temporary file with JavaScript content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        js_file = temp_dir_path / "example.js"
        
        # Example JavaScript code with functions
        js_content = textwrap.dedent("""
        /**
         * Calculates the sum of two numbers.
         * @param {number} a - First number
         * @param {number} b - Second number
         * @return {number} The sum of a and b
         */
        function add(a, b) {
            return a + b;
        }

        // Arrow function example
        const multiply = (a, b) => {
            return a * b;
        };

        // Anonymous function assignment
        const divide = function(a, b) {
            if (b === 0) {
                throw new Error("Division by zero");
            }
            return a / b;
        };
        """)
        
        with open(js_file, "w") as f:
            f.write(js_content)
        
        # Create output directory
        output_dir = temp_dir_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize stats
        stats = initialize_stats_dict(source=js_file, output_dir=output_dir)
        
        # Extract blocks
        block_count = _extract_js_ts_blocks(js_file, js_content, output_dir, stats)
        
        # Verify extraction
        assert block_count > 0, "Should extract at least one block"
        assert stats["code_blocks"] > 0, "Should increment code_blocks in stats"
        assert "javascript" in stats["languages"], "Should track JavaScript in languages"
        assert ".js" in stats["file_types"], "Should track .js extension"
        
        # Verify extracted blocks
        blocks_dir = output_dir / "blocks" / "code" / "javascript"
        assert blocks_dir.exists(), "Should create blocks directory"
        
        block_files = list(blocks_dir.glob("*.js"))
        assert len(block_files) > 0, "Should create block files"
        
        # Verify block contents
        found_functions = set()
        for block_file in block_files:
            with open(block_file) as f:
                content = f.read()
                if "function add" in content:
                    found_functions.add("add")
                elif "const multiply" in content:
                    found_functions.add("multiply")
                elif "const divide" in content:
                    found_functions.add("divide")
        
        assert found_functions, "Should find at least one function"

def test_ts_class_extraction():
    """Test extraction of TypeScript classes from code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        ts_file = temp_dir_path / "example.ts"
        
        # Example TypeScript code with a class
        ts_content = textwrap.dedent("""
        /**
         * Represents a person with a name and age.
         */
        class Person {
            private name: string;
            private age: number;
            
            constructor(name: string, age: number) {
                this.name = name;
                this.age = age;
            }
            
            getName(): string {
                return this.name;
            }
            
            getAge(): number {
                return this.age;
            }
            
            isAdult(): boolean {
                return this.age >= 18;
            }
        }
        """)
        
        with open(ts_file, "w") as f:
            f.write(ts_content)
        
        # Create output directory
        output_dir = temp_dir_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize stats
        stats = initialize_stats_dict(source=ts_file, output_dir=output_dir)
        
        # Extract blocks
        block_count = _extract_js_ts_blocks(ts_file, ts_content, output_dir, stats)
        
        # Verify extraction
        assert block_count > 0, "Should extract at least one block"
        assert stats["code_blocks"] > 0, "Should increment code_blocks in stats"
        assert "typescript" in stats["languages"], "Should track TypeScript in languages"
        assert ".ts" in stats["file_types"], "Should track .ts extension"
        
        # Verify extracted blocks
        blocks_dir = output_dir / "blocks" / "code" / "typescript"
        assert blocks_dir.exists(), "Should create blocks directory"
        
        block_files = list(blocks_dir.glob("*.ts"))
        assert len(block_files) > 0, "Should create block files"
        
        # Verify block contents
        found_items = set()
        for block_file in block_files:
            with open(block_file) as f:
                content = f.read()
                if "class Person" in content:
                    found_items.add("Person")
                elif "getName()" in content:
                    found_items.add("getName")
                elif "getAge()" in content:
                    found_items.add("getAge")
                elif "isAdult()" in content:
                    found_items.add("isAdult")
        
        assert "Person" in found_items, "Should extract Person class"
        assert len(found_items) > 1, "Should extract class methods"

def test_tsx_component_extraction():
    """
    Test extraction of React components from TSX files.
    
    This test verifies that:
    1. Tree-sitter correctly extracts React components (uppercase names)
    2. The component's content is preserved completely
    3. We don't try to extract methods separately (handled by generic parser)
    
    Official Documentation:
    - tree-sitter: https://tree-sitter.github.io/tree-sitter/
    - tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
    - tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
    """
    # Use a real React component from the test repo
    test_file = Path("test_repos/react/packages/react-devtools-shell/src/app/ToDoList/ListItem.js")
    
    with open(test_file) as f:
        component_content = f.read()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        tsx_file = temp_dir_path / "ListItem.tsx"
        
        # Write the component to a temporary file
        with open(tsx_file, "w") as f:
            f.write(component_content)
        
        # Create output directory
        output_dir = temp_dir_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize stats
        stats = initialize_stats_dict(source=tsx_file, output_dir=output_dir)
        
        # Extract blocks
        block_count = _extract_js_ts_blocks(tsx_file, component_content, output_dir, stats)
        
        # Verify extraction
        assert block_count > 0, "Should extract at least one block"
        assert stats["code_blocks"] > 0, "Should increment code_blocks in stats"
        assert "typescript" in stats["languages"], "Should track TypeScript in languages"
        assert ".tsx" in stats["file_types"], "Should track .tsx extension"
        
        # Verify extracted blocks
        blocks_dir = output_dir / "blocks" / "code" / "typescript"
        assert blocks_dir.exists(), "Should create blocks directory"
        
        # Look for both .tsx and .ts files since this is a React component
        block_files = list(blocks_dir.glob("*.tsx"))
        assert len(block_files) > 0, "Should create block files"
        
        # Verify block contents - only check for the component itself
        found_items = set()
        for block_file in block_files:
            with open(block_file) as f:
                content = f.read()
                if "ListItem" in content:
                    found_items.add("ListItem")
        
        # Only verify the component is extracted - methods are handled by generic parsing
        assert "ListItem" in found_items, "Should extract ListItem component"
        assert len(found_items) == 1, "Should only extract the component itself"
        
        # Verify the component content is complete
        component_file = next(f for f in block_files if "ListItem" in f.name)
        with open(component_file) as f:
            content = f.read()
            assert "handleDelete" in content, "Component should contain handleDelete"
            assert "handleToggle" in content, "Component should contain handleToggle"
            assert "useCallback" in content, "Component should contain useCallback"
