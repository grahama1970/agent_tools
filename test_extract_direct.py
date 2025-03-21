#!/usr/bin/env python3

import os
import sys
import tempfile
from pathlib import Path

from agent_tools.dualipa.extraction.extractors.code import extract_python_blocks
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict

# Configure paths properly
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"Python path: {sys.path}")

try:
    from agent_tools.dualipa.code_extractor import _extract_python_blocks
    print("Successfully imported _extract_python_blocks")
    
    # Create a simple test file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write('def hello_world():\n    return "Hello, World!"')
        f.flush()
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Initialize stats dictionary
            stats = {
                "code_blocks": 0, 
                "errors": [],
                "file_blocks": {}
            }
            
            # Extract code blocks
            file_path = Path(f.name)
            content = 'def hello_world():\n    return "Hello, World!"'
            
            print(f"File path: {file_path}")
            print(f"File exists: {file_path.exists()}")
            
            try:
                result = _extract_python_blocks(file_path, content, output_dir, stats)
                print(f"Result: {result}")
                print(f"Result type: {type(result)}")
                print(f"Stats: {stats}")
                
                # Examine output directory
                print("\nOutput directory structure:")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"  - {file_path}")
                        
                        # Read and print file content
                        with open(file_path, 'r') as f:
                            print(f"    Content: {f.read()[:100]}...")
            except Exception as e:
                print(f"Error during extraction: {e}")
                
except ImportError as e:
    print(f"Import error: {e}") 