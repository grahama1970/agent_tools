#!/usr/bin/env python3
import json
import sys
from pprint import pprint

def analyze_deepseek_blocks():
    """Analyze the deepseek.md blocks from the extraction output."""
    with open('/tmp/tmpbrehuk8e.json', 'r') as f:
        data = json.load(f)
    
    # Find all blocks related to deepseek.md
    deepseek_blocks = [block for block in data if 'path' in block and 'deepseek.md' in block.get('path', '')]
    print(f'Number of deepseek.md blocks: {len(deepseek_blocks)}')
    
    # Count blocks by type
    block_types = {}
    for block in deepseek_blocks:
        block_type = block.get('type', 'unknown')
        block_types[block_type] = block_types.get(block_type, 0) + 1
    
    print("\nBlock types:")
    for block_type, count in block_types.items():
        print(f"  {block_type}: {count}")
    
    # Find the main file block
    file_blocks = [block for block in deepseek_blocks if block.get('type') == 'file']
    if file_blocks:
        print("\nFile block:")
        file_block = file_blocks[0]
        print(f"  UUID: {file_block.get('uuid')}")
        print(f"  Path: {file_block.get('path')}")
        print(f"  Child UUIDs: {len(file_block.get('children', []))}")
        
        # Get sections
        section_blocks = [block for block in deepseek_blocks if block.get('type') == 'section']
        print(f"\nSection blocks: {len(section_blocks)}")
        for i, section in enumerate(section_blocks[:5], 1):  # Print first 5 sections
            print(f"  Section {i}:")
            print(f"    Title: {section.get('name', 'untitled')}")
            print(f"    Parent: {section.get('parent')}")
            print(f"    Children: {len(section.get('children', []))}")
            print(f"    Section Hierarchy: {section.get('section_hierarchy', '')}")
            
        # Get tables
        table_blocks = [block for block in deepseek_blocks if block.get('type') == 'table']
        print(f"\nTable blocks: {len(table_blocks)}")
        for i, table in enumerate(table_blocks[:3], 1):  # Print first 3 tables
            print(f"  Table {i}:")
            print(f"    Parent: {table.get('parent')}")
            print(f"    Position: {table.get('position')}")
            print(f"    Content preview: {table.get('content', '')[:50]}...")
            
        # Get code blocks
        code_blocks = [block for block in deepseek_blocks if block.get('type') == 'code_block']
        print(f"\nCode blocks: {len(code_blocks)}")
        for i, code in enumerate(code_blocks[:3], 1):  # Print first 3 code blocks
            print(f"  Code block {i}:")
            print(f"    Parent: {code.get('parent')}")
            print(f"    Position: {code.get('position')}")
            print(f"    Language: {code.get('language')}")
            print(f"    Content preview: {code.get('content', '')[:50]}...")
        
        # Get text blocks
        text_blocks = [block for block in deepseek_blocks if block.get('type') == 'text']
        print(f"\nText blocks: {len(text_blocks)}")
        for i, text in enumerate(text_blocks[:3], 1):  # Print first 3 text blocks
            print(f"  Text block {i}:")
            print(f"    Parent: {text.get('parent')}")
            print(f"    Position: {text.get('position')}")
            print(f"    Content preview: {text.get('content', '')[:50]}...")
            
    else:
        print("No file block found for deepseek.md")

if __name__ == "__main__":
    analyze_deepseek_blocks()