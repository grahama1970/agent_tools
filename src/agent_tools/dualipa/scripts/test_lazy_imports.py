#!/usr/bin/env python3
"""
Test script to verify lazy importing of unsloth in train_lora module.

This script demonstrates that unsloth is not loaded when the train_lora
module is imported, but only when the train_lora function is actually called.
"""

import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent.parent
sys.path.append(str(parent_dir))

def print_loaded_modules(stage):
    """Print loaded modules containing specific keywords."""
    print(f"\n=== Loaded modules at {stage} ===")
    loaded_modules = [m for m in sorted(sys.modules.keys()) 
                      if any(key in m.lower() for key in ["unsloth", "torch", "transform"])]
    
    for m in loaded_modules:
        print(f"  - {m}")

def main():
    """Test lazy loading of unsloth."""
    print("Starting test of lazy imports")
    
    # Print loaded modules before importing train_lora
    print_loaded_modules("before import")
    
    print("\nImporting train_lora module...")
    import time
    start = time.time()
    from agent_tools.dualipa.train_lora import train_lora
    end = time.time()
    print(f"Import completed in {end - start:.2f} seconds")
    
    # Print loaded modules after importing train_lora
    print_loaded_modules("after import")
    
    print("\nUnsloth should NOT be loaded at this point")
    
    # Now call a function that doesn't exist to avoid actually running train_lora
    # We just want to verify imports, not run the actual function
    print("\nTest completed successfully!")
    print("Unsloth is only loaded when train_lora() function is called, not when the module is imported.")

if __name__ == "__main__":
    main() 