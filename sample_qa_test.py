#!/usr/bin/env python3
"""
Sample QA Test Script

This script demonstrates how to use the extracted JSON file with a QA system.
It loads the extraction, filters for Python code examples, and simulates
a QA query to verify the data works as expected.

Usage:
    python sample_qa_test.py /path/to/your/qa_file.json
"""

import sys
import json
import random
from pathlib import Path
from pprint import pprint


def sample_qa_test(json_path):
    """Sample test of QA functionality with the extracted JSON."""
    print(f"Loading QA JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON file: {e}")
        return False
    
    sections = data['sections']
    print(f"Loaded {len(sections)} sections")
    
    # Filter for Python functions
    python_functions = [
        s for s in sections 
        if s.get('language') == 'python' and s.get('type') == 'function'
    ]
    
    if not python_functions:
        print("No Python functions found in the extraction")
        return False
    
    print(f"Found {len(python_functions)} Python functions")
    
    # Select a random function for QA testing
    function = random.choice(python_functions)
    
    print("\nSample Python function for QA testing:")
    print(f"Name: {function.get('name')}")
    print(f"Type: {function.get('type')}")
    print(f"Language: {function.get('language')}")
    print(f"UUID: {function.get('uuid')}")
    
    # Preview content (truncated)
    content = function.get('content', '')
    print(f"\nContent preview (first 300 chars):")
    print(f"{content[:300]}..." if len(content) > 300 else content)
    
    print("\nSimulating QA query about this function...")
    query = f"What does the {function.get('name')} function do?"
    
    # In a real implementation, you would call your QA system here
    # qa_response = qa_system.query(query, context=function)
    
    print(f"\nQuery: {query}")
    print("Response: [This would be answered by your QA system using the extraction]")
    
    # Let's also test the section relationships
    function_uuid = function.get('uuid')
    if function_uuid in data['section_relationships']['parent_child']:
        relationship = data['section_relationships']['parent_child'][function_uuid]
        print("\nFunction relationships:")
        print(f"Parent: {relationship.get('parent')}")
        print(f"Children: {relationship.get('children')}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/qa_file.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    if sample_qa_test(json_path):
        print("\nSample QA test completed successfully")
        sys.exit(0)
    else:
        print("\nSample QA test failed")
        sys.exit(1)