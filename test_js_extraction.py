#!/usr/bin/env python3

from agent_tools.dualipa.code_extractor import _extract_js_ts_blocks
import tempfile
from pathlib import Path

def test_js_extraction():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test JavaScript content
        content = """
        function test() { 
            console.log("Hello"); 
            return true;
        }
        
        class Example {
            constructor(name) {
                this.name = name;
            }
            
            greet() {
                return `Hello, ${this.name}!`;
            }
        }
        """
        
        # Create stats dictionary
        stats = {'code_blocks': 0, 'file_blocks': {}, 'errors': []}
        
        # Set up file path
        file_path = Path('test.js')
        
        # Run extraction
        _extract_js_ts_blocks(file_path, content, Path(temp_dir), stats, 'javascript')
        
        # Print results
        print('Stats:', stats)
        
        # Print blocks with token counts
        print("\nExtracted blocks with token counts:")
        for file_path, blocks in stats.get('file_blocks', {}).items():
            for block in blocks:
                print(f"Block: {block.get('name')}")
                print(f"  Type: {block.get('block_type')}")
                print(f"  Token count: {block.get('token_count', 'MISSING')}")
                print(f"  Metadata token count: {block.get('metadata', {}).get('token_count', 'MISSING')}")
                print(f"  Content: {block.get('content')[:50]}...")
                print()

if __name__ == "__main__":
    test_js_extraction() 