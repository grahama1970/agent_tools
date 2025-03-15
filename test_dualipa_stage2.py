"""
Test Script for DuaLipa Pipeline Stage 2: QA Pair Generation and Dataset Formatting

This script tests the second stage of the DuaLipa pipeline:
1. Taking the extracted repository content
2. Generating question-answer pairs
3. Formatting them into a dataset suitable for training

Usage:
    python test_dualipa_stage2.py [EXTRACTED_DATA_PATH]
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_stage2(extracted_data_path=None):
    """Test Stage 2 of the DuaLipa pipeline: QA Generation and Formatting."""
    print("Testing DuaLipa Pipeline Stage 2: QA Pair Generation and Dataset Formatting")
    print("=" * 80)
    
    # If no extracted data path is provided, run stage 1 first to get data
    if not extracted_data_path:
        print("No extracted data path provided, running Stage 1 first...")
        temp_dir = tempfile.mkdtemp(prefix="dualipa_temp_")
        
        try:
            # Import Stage 1 functions
            from src.agent_tools.dualipa.github_utils import download_github_repo
            from src.agent_tools.dualipa.code_extractor import extract_repository
            
            # Run stage 1 with a small test repository
            repo_url = "https://github.com/yamadashy/repomix"
            download_path = download_github_repo(repo_url, None)
            
            # Extract repository content
            extracted_data_path = os.path.join(temp_dir, "extracted_data.json")
            extract_repository(
                source=download_path,
                output_path=temp_dir,
                include_extensions=[".py", ".md", ".txt", ".js", ".html", ".css"],
                max_file_size_kb=500,
                ignore_patterns=["node_modules", "__pycache__", ".git"]
            )
        except Exception as e:
            print(f"❌ Failed to prepare test data: {e}")
            return False
    
    # Verify extracted data exists
    if not os.path.exists(extracted_data_path):
        print(f"❌ Extracted data not found at: {extracted_data_path}")
        return False
    
    # Create temporary output directory for formatted dataset
    formatted_output_dir = tempfile.mkdtemp(prefix="dualipa_format_")
    print(f"Output directory for formatted dataset: {formatted_output_dir}")
    
    try:
        # Step 1: Import required modules
        try:
            from src.agent_tools.dualipa.format_dataset import format_for_lora
            print("✅ Successfully imported required modules")
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
            return False
        
        # Step 2: Read the extracted data
        print("\n📂 Step 2: Reading extracted repository data...")
        try:
            with open(extracted_data_path, 'r') as f:
                extracted_data = json.load(f)
                
            file_count = len(extracted_data.get('files', []))
            print(f"✅ Found {file_count} files in extracted data")
            
            if file_count == 0:
                print("⚠️ Warning: No files found in extracted data")
                
        except Exception as e:
            print(f"❌ Failed to read extracted data: {e}")
            return False
        
        # Step 3: Format dataset for LoRA training
        print("\n🧩 Step 3: Generating QA pairs and formatting dataset...")
        try:
            # Define formatting parameters
            format_kwargs = {
                "qa_format": "instruct",  # Use instruction format
                "max_qa_pairs": 10,  # Limit for testing
                "min_code_lines": 3,
                "temperature": 0.7,
                "validator_config": {
                    "run_validation": True,
                    "fix_invalid": True
                }
            }
            
            # Generate QA pairs and format dataset
            format_result = format_for_lora(
                extracted_data_path,
                formatted_output_dir,
                **format_kwargs
            )
            
            # Check if formatting was successful
            if not format_result:
                print("❌ Formatting returned no result")
                return False
                
            # Check if output files were created
            train_file = os.path.join(formatted_output_dir, "train.jsonl")
            eval_file = os.path.join(formatted_output_dir, "eval.jsonl")
            
            if not os.path.exists(train_file):
                print(f"❌ Training file not created: {train_file}")
                return False
                
            if not os.path.exists(eval_file):
                print(f"⚠️ Evaluation file not created: {eval_file}")
            
            # Count QA pairs
            train_qa_count = sum(1 for _ in open(train_file, 'r'))
            print(f"✅ Generated {train_qa_count} training QA pairs")
            
            if os.path.exists(eval_file):
                eval_qa_count = sum(1 for _ in open(eval_file, 'r'))
                print(f"✅ Generated {eval_qa_count} evaluation QA pairs")
            
            # Check QA pair quality
            print("\n📊 Analyzing QA pair quality...")
            with open(train_file, 'r') as f:
                sample_qa = json.loads(f.readline())
                
                # Print a sample QA pair
                print("\nSample QA Pair:")
                print("-" * 40)
                print(f"Question: {sample_qa.get('question', 'N/A')}")
                print("-" * 40)
                print(f"Answer: {sample_qa.get('answer', 'N/A')[:200]}...")
                print("-" * 40)
            
            print("\n🎉 Stage 2 (QA Pair Generation and Formatting) completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to format dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Clean up
        print("\n🧹 Cleaning up temporary directories...")
        try:
            import shutil
            if 'download_path' in locals():
                shutil.rmtree(download_path, ignore_errors=True)
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            # Don't remove formatted output as user might want to inspect it
            print(f"✅ Cleanup completed (keeping formatted output at {formatted_output_dir} for inspection)")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    # Get extracted data path from command line if provided
    extracted_data_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_stage2(extracted_data_path)
    sys.exit(0 if success else 1) 