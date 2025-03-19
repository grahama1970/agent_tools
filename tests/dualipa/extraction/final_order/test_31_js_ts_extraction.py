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
        
        # Verify block format matches specification
        blocks = stats["file_blocks"][str(ts_file)]
        for block in blocks:
            # Verify required fields from extraction_format.md
            assert "uuid" in block, "Block should have UUID"
            assert "id" in block, "Block should have human-readable ID"
            assert block["type"] == "code", "Block type should be 'code'"
            assert block["language"] == "typescript", "Block language should be 'typescript'"
            assert "title" in block, "Block should have title"
            assert "content" in block, "Block should have content"
            assert "file_path" in block, "Block should have file path"
            assert "breadcrumb" in block, "Block should have breadcrumb"
            assert isinstance(block["breadcrumb"], list), "Breadcrumb should be a list"
            assert "parent_uuid" in block, "Block should have parent_uuid"
            assert "child_uuids" in block, "Block should have child_uuids"
            assert isinstance(block["child_uuids"], list), "child_uuids should be a list"
            
            # Verify code-specific fields
            assert "dependencies" in block, "Block should have dependencies"
            assert "imports" in block["dependencies"], "Dependencies should track imports"
            assert "referenced_types" in block["dependencies"], "Dependencies should track type references"
            
            assert "test_coverage" in block, "Block should have test_coverage"
            assert "test_file" in block["test_coverage"], "test_coverage should have test_file"
            assert "coverage_percentage" in block["test_coverage"], "test_coverage should have coverage_percentage"
            
            assert "version_history" in block, "Block should have version_history"
            assert "last_modified" in block["version_history"], "version_history should have last_modified"
            
            assert "qa_generation" in block, "Block should have qa_generation"
            assert "difficulty_levels" in block["qa_generation"], "qa_generation should have difficulty_levels"
            assert "knowledge_prerequisites" in block["qa_generation"], "qa_generation should have knowledge_prerequisites"
            assert "focus_areas" in block["qa_generation"], "qa_generation should have focus_areas"
            assert "qa_examples" in block["qa_generation"], "qa_generation should have qa_examples"
            
            # Verify content analysis
            if "class Person" in block["content"]:
                assert "Type hints" in block["qa_generation"]["knowledge_prerequisites"], "Should detect type usage"
                assert "Class design" in block["qa_generation"]["focus_areas"], "Should focus on class design"
                assert "Type system" in block["qa_generation"]["focus_areas"], "Should focus on type system"
                assert "intermediate" in block["qa_generation"]["difficulty_levels"], "Types should increase difficulty"
                
                # Verify type references are captured
                type_refs = block["dependencies"]["referenced_types"]
                assert "string" in type_refs or "String" in type_refs, "Should capture string type"
                assert "number" in type_refs or "Number" in type_refs, "Should capture number type"
                assert "boolean" in type_refs or "Boolean" in type_refs, "Should capture boolean type"
                
                # Verify OOP concepts are detected
                assert "OOP" in block["qa_generation"]["knowledge_prerequisites"], "Should detect OOP usage"
                assert "private" in block["content"], "Should detect private members"
                assert "constructor" in block["content"], "Should detect constructor"

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
