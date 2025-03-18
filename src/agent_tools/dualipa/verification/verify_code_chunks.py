#!/usr/bin/env python3
"""
Verify code chunking functionality.

This script tests the code chunking functionality in the DuaLipa library,
used for breaking down code blocks into manageable chunks for processing.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Import the required modules
from agent_tools.dualipa.chunking import (
    chunk_code,
    chunk_text,
    chunk_by_tokens,
    merge_chunks
)

def print_header(text, underline='='):
    """Print a header with underline."""
    print(f"\n{text}")
    print(underline * len(text))

def get_test_code():
    """Return a large piece of test code for chunking."""
    code = """
import os
import sys
from typing import List, Dict, Any, Optional

def calculate_factorial(n: int) -> int:
    \"\"\"
    Calculate the factorial of a given number.
    
    Args:
        n: The number to calculate factorial for
        
    Returns:
        The factorial value
    \"\"\"
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def is_prime(n: int) -> bool:
    \"\"\"
    Check if a number is prime.
    
    Args:
        n: The number to check
        
    Returns:
        True if the number is prime, False otherwise
    \"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MathOperations:
    \"\"\"
    A class that provides various mathematical operations.
    \"\"\"
    
    def __init__(self, initial_value: float = 0):
        self.value = initial_value
        self.operations_performed = 0
    
    def add(self, x: float) -> float:
        \"\"\"Add a value to the current value.\"\"\"
        self.value += x
        self.operations_performed += 1
        return self.value
    
    def subtract(self, x: float) -> float:
        \"\"\"Subtract a value from the current value.\"\"\"
        self.value -= x
        self.operations_performed += 1
        return self.value
    
    def multiply(self, x: float) -> float:
        \"\"\"Multiply the current value by x.\"\"\"
        self.value *= x
        self.operations_performed += 1
        return self.value
    
    def divide(self, x: float) -> float:
        \"\"\"Divide the current value by x.\"\"\"
        if x == 0:
            raise ValueError("Cannot divide by zero")
        self.value /= x
        self.operations_performed += 1
        return self.value
    
    def get_stats(self) -> Dict[str, Any]:
        \"\"\"Get statistics about the operations.\"\"\"
        return {
            "current_value": self.value,
            "operations_performed": self.operations_performed
        }

def process_list(items: List[Any]) -> Dict[str, Any]:
    \"\"\"
    Process a list of items and return statistics.
    
    Args:
        items: List of items to process
        
    Returns:
        Dictionary with statistics about the list
    \"\"\"
    if not items:
        return {"count": 0, "has_items": False}
    
    result = {
        "count": len(items),
        "has_items": True,
        "types": {},
        "numeric_stats": None
    }
    
    # Collect type information
    for item in items:
        item_type = type(item).__name__
        if item_type not in result["types"]:
            result["types"][item_type] = 0
        result["types"][item_type] += 1
    
    # Calculate statistics for numeric values
    numeric_items = [item for item in items if isinstance(item, (int, float))]
    if numeric_items:
        result["numeric_stats"] = {
            "count": len(numeric_items),
            "min": min(numeric_items),
            "max": max(numeric_items),
            "sum": sum(numeric_items),
            "average": sum(numeric_items) / len(numeric_items)
        }
    
    return result
"""
    return code * 2  # Duplicate to make it longer

def verify_chunk_text():
    """Verify chunking text into smaller pieces."""
    print_header("Testing text chunking", "-")
    
    test_text = get_test_code()
    max_chunk_size = 500
    
    try:
        print(f"Chunking text of {len(test_text)} characters with max chunk size {max_chunk_size}...")
        chunks = chunk_text(test_text, max_chunk_size)
        
        # Print chunk information
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show only first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk)}")
            print(f"  First line: {chunk.splitlines()[0] if chunk.splitlines() else ''}")
            print(f"  Last line: {chunk.splitlines()[-1] if chunk.splitlines() else ''}")
        
        # Verify that each chunk is smaller than the max size
        all_valid = all(len(chunk) <= max_chunk_size for chunk in chunks)
        print(f"\nAll chunks are within size limit: {'✅' if all_valid else '❌'}")
        
        # Verify that the combined text is the same as the original
        combined = "".join(chunks)
        is_identical = combined == test_text
        print(f"Combined chunks match original text: {'✅' if is_identical else '❌'}")
        
        return all_valid and is_identical
    except Exception as e:
        print(f"❌ Error during text chunking: {str(e)}")
        return False

def verify_chunk_code():
    """Verify chunking code into smaller pieces while respecting structure."""
    print_header("Testing code chunking", "-")
    
    test_code = get_test_code()
    max_chunk_size = 500
    
    try:
        print(f"Chunking code of {len(test_code)} characters with max chunk size {max_chunk_size}...")
        chunks = chunk_code(test_code, "python", max_chunk_size)
        
        # Print chunk information
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show only first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk)}")
            print(f"  First line: {chunk.splitlines()[0] if chunk.splitlines() else ''}")
            print(f"  Last line: {chunk.splitlines()[-1] if chunk.splitlines() else ''}")
        
        # Verify that the combined code is the same as the original (ignoring whitespace)
        combined = "".join(chunks)
        # Remove all whitespace for comparison to handle indentation differences
        original_no_ws = ''.join(test_code.split())
        combined_no_ws = ''.join(combined.split())
        is_identical = combined_no_ws == original_no_ws
        print(f"Combined chunks preserve code content: {'✅' if is_identical else '❌'}")
        
        return is_identical
    except Exception as e:
        print(f"❌ Error during code chunking: {str(e)}")
        return False

def verify_chunk_by_tokens():
    """Verify chunking by token count."""
    print_header("Testing chunking by tokens", "-")
    
    test_text = get_test_code()
    max_tokens = 200
    
    try:
        print(f"Chunking text with max token count {max_tokens}...")
        chunks = chunk_by_tokens(test_text, max_tokens)
        
        # Print chunk information
        print(f"Created {len(chunks)} chunks:")
        total_tokens = 0
        for i, chunk in enumerate(chunks[:3]):  # Show only first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk['text'])}")
            print(f"  Tokens: {chunk['token_count']}")
            print(f"  First line: {chunk['text'].splitlines()[0] if chunk['text'].splitlines() else ''}")
            total_tokens += chunk['token_count']
        
        print(f"\nTotal tokens in first 3 chunks: {total_tokens}")
        
        # Verify that each chunk is smaller than the max tokens
        all_valid = all(chunk['token_count'] <= max_tokens for chunk in chunks)
        print(f"All chunks are within token limit: {'✅' if all_valid else '❌'}")
        
        # Verify that the combined text is the same as the original
        combined = "".join(chunk['text'] for chunk in chunks)
        # Remove all whitespace for comparison
        original_no_ws = ''.join(test_text.split())
        combined_no_ws = ''.join(combined.split())
        is_identical = combined_no_ws == original_no_ws
        print(f"Combined chunks preserve content: {'✅' if is_identical else '❌'}")
        
        return all_valid and is_identical
    except Exception as e:
        print(f"❌ Error during token chunking: {str(e)}")
        return False

def verify_merge_chunks():
    """Verify merging chunks together."""
    print_header("Testing chunk merging", "-")
    
    # Create some test chunks
    chunks = [
        "This is chunk 1.\nIt has multiple lines.",
        "This is chunk 2.\nAnother multi-line chunk.",
        "This is chunk 3.\nYet another chunk.",
        "This is chunk 4.\nThe final chunk."
    ]
    
    try:
        print("Merging chunks with max size 50...")
        merged = merge_chunks(chunks, max_chunk_size=50)
        
        # Print merged chunk information
        print(f"Created {len(merged)} merged chunks:")
        for i, chunk in enumerate(merged):
            print(f"\nMerged Chunk {i+1}:")
            print(f"  Length: {len(chunk)}")
            print(f"  Content: {chunk[:50]}...")
        
        # Verify that each merged chunk is smaller than the max size
        all_valid = all(len(chunk) <= 50 for chunk in merged)
        print(f"All merged chunks are within size limit: {'✅' if all_valid else '❌'}")
        
        # Verify that the combined text contains all original chunks
        combined = "".join(merged)
        original = "".join(chunks)
        is_complete = all(chunk in combined for chunk in chunks)
        print(f"Merged chunks contain all original content: {'✅' if is_complete else '❌'}")
        
        return all_valid and is_complete
    except Exception as e:
        print(f"❌ Error during chunk merging: {str(e)}")
        return False

def main():
    """Run all verification tests."""
    print_header("Code Chunking Verification")
    
    # Run all verification tests
    text_chunk_success = verify_chunk_text()
    code_chunk_success = verify_chunk_code()
    token_chunk_success = verify_chunk_by_tokens()
    merge_chunk_success = verify_merge_chunks()
    
    # Calculate overall success
    all_success = (
        text_chunk_success and
        code_chunk_success and
        token_chunk_success and
        merge_chunk_success
    )
    
    # Print summary
    print_header("Verification Summary")
    print(f"Text Chunking: {'✅' if text_chunk_success else '❌'}")
    print(f"Code Chunking: {'✅' if code_chunk_success else '❌'}")
    print(f"Token Chunking: {'✅' if token_chunk_success else '❌'}")
    print(f"Chunk Merging: {'✅' if merge_chunk_success else '❌'}")
    print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 