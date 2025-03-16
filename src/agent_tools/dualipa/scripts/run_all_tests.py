#!/usr/bin/env python
"""
Script to run all the basic extraction tests in sequence.
This is useful for verifying that all components are working correctly.

Example usage:
    python run_all_tests.py ./test_output
"""

import os
import sys
import shutil
import time
from pathlib import Path
import subprocess

def run_command(command, title=None):
    """Run a command and print its output."""
    if title:
        print("\n" + "="*80)
        print(f" {title} ".center(80, "="))
        print("="*80 + "\n")
    
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    # Get the output directory from command line args
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./test_output_{timestamp}"
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")
    
    # Get the absolute path to the scripts directory
    scripts_dir = Path(__file__).resolve().parent
    
    # Define test cases with subdirectories
    test_cases = [
        {
            "title": "TEST 1: Extract Code From Self",
            "script": "extract_single_file.py",
            "args": [str(scripts_dir / "extract_single_file.py"), f"{output_dir}/test1"]
        },
        {
            "title": "TEST 2: Extract Code From Directory",
            "script": "extract_local_dir.py",
            "args": [str(scripts_dir), f"{output_dir}/test2"]
        },
        {
            "title": "TEST 3: Extract From Markdown",
            "script": "extract_markdown_blocks.py",
            "args": [str(scripts_dir.parent / "task.md"), f"{output_dir}/test3"]
        }
    ]
    
    # Run each test case
    results = []
    for i, test in enumerate(test_cases):
        # Create the output subdirectory
        os.makedirs(test["args"][1], exist_ok=True)
        
        # Run the test
        cmd = [sys.executable, str(scripts_dir / test["script"])] + test["args"]
        start_time = time.time()
        exit_code = run_command(cmd, test["title"])
        elapsed = time.time() - start_time
        
        # Record the result
        status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
        results.append({
            "test": test["title"],
            "status": status,
            "time": f"{elapsed:.2f}s",
            "output_dir": test["args"][1]
        })
    
    # Print summary
    print("\n" + "="*80)
    print(" TEST SUMMARY ".center(80, "="))
    print("="*80)
    
    for result in results:
        print(f"{result['status']} - {result['test']} ({result['time']})")
        print(f"        Output: {result['output_dir']}")
    
    # Check if all tests passed
    all_passed = all(r["status"] == "✅ PASSED" for r in results)
    print("\n" + "="*80)
    if all_passed:
        print(" ALL TESTS PASSED ".center(80, "="))
    else:
        print(" SOME TESTS FAILED ".center(80, "="))
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 