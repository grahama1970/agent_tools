#!/usr/bin/env python3
"""
Run all verification scripts in sequence.

This script runs all verification scripts in the current directory
to verify that the DuaLipa library works correctly.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Define the list of verification scripts to run (in order)
VERIFICATION_SCRIPTS = [
    "verify_repo_operations.py",
    "verify_code_extraction.py",
    "verify_code_chunks.py",
    "verify_markdown_parser.py",
    "verify_multilang_extraction.py",
    "verify_repository_extraction.py"
]

def run_verification_script(script_path):
    """Run a verification script and return its success status."""
    print(f"\n{'=' * 80}")
    print(f"Running {script_path}")
    print(f"{'=' * 80}\n")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            check=False,
            capture_output=False
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Verification {script_path} passed in {duration:.2f} seconds!")
            return True
        else:
            print(f"\n❌ Verification {script_path} failed in {duration:.2f} seconds with code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running {script_path}: {str(e)}")
        return False

def main():
    """Run all verification scripts."""
    print("\n" + "=" * 80)
    print("Running All Verification Scripts")
    print("=" * 80 + "\n")
    
    # Get the directory of this script
    current_dir = Path(__file__).parent
    
    # Track success and failure counts
    success_count = 0
    failure_count = 0
    
    # Get all verification scripts if none are specified
    scripts_to_run = VERIFICATION_SCRIPTS
    
    # Run all verification scripts
    for script_name in scripts_to_run:
        script_path = current_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Verification script {script_path} not found")
            failure_count += 1
            continue
        
        # Run the verification script
        success = run_verification_script(script_path)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Verification Summary: {success_count} passed, {failure_count} failed")
    print("=" * 80 + "\n")
    
    # Return non-zero exit code if any verification failed
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 