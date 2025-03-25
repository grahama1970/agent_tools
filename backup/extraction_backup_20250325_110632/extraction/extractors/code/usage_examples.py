"""
Usage examples for DuaLipa code extraction.

This module contains example usages of the code extraction modules,
to demonstrate their functionality and provide testing examples.

Key Features:
1. Demonstrates code extraction for different languages
2. Shows extraction output formats
3. Provides example inputs and outputs for testing

Dependencies:
- agent_tools.dualipa.extraction.extractors.code.code_extractor
- loguru: For logging
- pathlib: For file path handling

Documentation Links:
- Code Extraction: ../docs/extraction.md
- Block Format: ../docs/block_format.md
"""

import textwrap
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from agent_tools.dualipa.extraction.extractors.code.code_extractor import extract_code_blocks
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
from agent_tools.dualipa.extraction.extractors.utils.react_extractor import is_react_component, extract_react_component, extract_react_props

def code_extractor_example() -> None:
    """Example usage of code extraction."""
    # Example Python file
    python_content = textwrap.dedent('''
        class ExampleClass:
            """Example class docstring."""
            
            def __init__(self, name: str):
                self.name = name
                
            def greet(self) -> str:
                return f"Hello, {self.name}!"
                
        def example_function(x: int, y: int) -> int:
            """Example function docstring."""
            return x + y
    ''').strip()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(python_content)
        temp_path = f.name
        
    try:
        # Create output directory
        output_dir = Path(tempfile.mkdtemp())
        
        # Extract blocks
        blocks = extract_code_blocks(temp_path, output_dir)
        
        print("Extracted Blocks:")
        for block in blocks:
            print(f"\nType: {block['type']}")
            print(f"Name: {block['name']}")
            print(f"Lines: {block['line_start']}-{block['line_end']}")
            print("Metadata:", block['metadata'])
            print("Content:")
            print(block['content'])
            
        # Show output directory contents
        print("\nOutput Directory Contents:")
        for file_path in output_dir.glob("**/*"):
            if file_path.is_file():
                print(f"- {file_path.relative_to(output_dir)}")
                
    finally:
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
def js_ts_extractor_example() -> None:
    """Example usage of JavaScript/TypeScript extraction."""
    # Example TypeScript file
    ts_content = textwrap.dedent('''
        import React from 'react';
        
        interface Props {
            name: string;
        }
        
        export class Greeter extends React.Component<Props> {
            render() {
                return <div>Hello {this.props.name}!</div>;
            }
        }
        
        function add(a: number, b: number): number {
            return a + b;
        }
    ''').strip()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.tsx', mode='w', delete=False) as f:
        f.write(ts_content)
        temp_path = f.name
        
    try:
        # Create output directory
        output_dir = Path(tempfile.mkdtemp())
        
        # Extract blocks using js_ts_extractor directly
        blocks, stats = extract_js_ts_blocks(temp_path, output_dir)
        
        print("Extracted TypeScript Blocks:")
        for block in blocks:
            print(f"\nType: {block['type']}")
            print(f"Name: {block['name']}")
            print(f"Lines: {block['line_start']}-{block['line_end']}")
            print("Metadata:", {k: v for k, v in block['metadata'].items() if k not in ['imports', 'exports']})
            print("Content:")
            print(block['content'])
            
        # Show statistics
        print("\nExtraction Statistics:")
        for key, value in stats.items():
            if key != 'file_blocks':
                print(f"- {key}: {value}")
            
    finally:
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)

def react_component_example() -> None:
    """Example usage of React component extraction."""
    # Example React component
    tsx_content = textwrap.dedent('''
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
    ''').strip()
    
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
            
            # Extract props
            props = extract_react_props(tsx_content, component['name'])
            print("\nComponent props:")
            for prop in props:
                print(f"  {prop['name']}: {prop['type']}")

if __name__ == "__main__":
    print("=== Code Extractor Example ===")
    code_extractor_example()
    
    print("\n=== JS/TS Extractor Example ===")
    js_ts_extractor_example()
    
    print("\n=== React Component Example ===")
    react_component_example()