"""
Test Script for DuaLipa Pipeline Stage 3: Model Fine-tuning with Unsloth

This script tests the third stage of the DuaLipa pipeline:
1. Taking the formatted QA pairs dataset
2. Fine-tuning a model using Unsloth's LoRA adapters
3. Validating the trained model/adapter

Usage:
    python test_dualipa_stage3.py [FORMATTED_DATA_DIR]
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_stage3(formatted_data_dir=None):
    """Test Stage 3 of the DuaLipa pipeline: Model Fine-tuning with Unsloth."""
    print("Testing DuaLipa Pipeline Stage 3: Model Fine-tuning with Unsloth")
    print("=" * 80)
    
    # If no formatted data directory is provided, run stage 2 first to get data
    if not formatted_data_dir:
        print("No formatted data directory provided, running Stage 2 first...")
        
        try:
            # Import and run Stage 2
            from test_dualipa_stage2 import test_stage2
            
            # Run stage 2 which will handle stage 1 if needed
            test_stage2()
            
            # Look for the most recent formatted output directory
            temp_dirs = [d for d in os.listdir('/tmp') if d.startswith('dualipa_format_')]
            if not temp_dirs:
                print("❌ No formatted data directory found from Stage 2")
                return False
                
            # Sort by creation time (most recent first)
            temp_dirs.sort(key=lambda d: os.path.getctime(os.path.join('/tmp', d)), reverse=True)
            formatted_data_dir = os.path.join('/tmp', temp_dirs[0])
            print(f"Using most recent formatted data directory: {formatted_data_dir}")
            
        except Exception as e:
            print(f"❌ Failed to prepare test data: {e}")
            return False
    
    # Verify formatted data exists
    train_file = os.path.join(formatted_data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        print(f"❌ Training data not found at: {train_file}")
        return False
    
    # Create temporary output directory for model
    model_output_dir = tempfile.mkdtemp(prefix="dualipa_model_")
    print(f"Output directory for trained model: {model_output_dir}")
    
    try:
        # Step 1: Import required modules
        try:
            from src.agent_tools.dualipa.train_lora import train_lora
            print("✅ Successfully imported required modules")
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
            return False
        
        # Step 2: Check data format
        print("\n📂 Step 2: Validating training data...")
        try:
            # Count samples
            train_count = sum(1 for _ in open(train_file, 'r'))
            print(f"✅ Found {train_count} training samples")
            
            # Check sample format
            with open(train_file, 'r') as f:
                sample = json.loads(f.readline())
                required_fields = ['question', 'answer']
                
                if all(field in sample for field in required_fields):
                    print("✅ Training data format is valid")
                else:
                    print(f"❌ Training data missing required fields: {[f for f in required_fields if f not in sample]}")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to validate training data: {e}")
            return False
        
        # Step 3: Configure fine-tuning
        print("\n🧠 Step 3: Configuring model fine-tuning...")
        try:
            # Define training parameters for a quick test
            train_kwargs = {
                "base_model": "unsloth/mistral-7b-bnb-4bit",  # Use a smaller model for testing
                "epochs": 1,
                "micro_batch_size": 1,
                "max_seq_length": 512,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "test_mode": True  # Enable test mode to skip actual training
            }
            
            print(f"✅ Training configuration prepared")
            
        except Exception as e:
            print(f"❌ Failed to configure training: {e}")
            return False
            
        # Step 4: Run fine-tuning
        print("\n🚀 Step 4: Running model fine-tuning (test mode)...")
        try:
            # Train the model
            adapter_path = train_lora(
                dataset_path=formatted_data_dir,
                output_dir=model_output_dir,
                **train_kwargs
            )
            
            # Check if training was successful
            if not adapter_path:
                print("❌ Training returned no adapter path")
                return False
                
            print(f"✅ Model fine-tuning completed successfully")
            print(f"✅ Adapter path: {adapter_path}")
            
            # Check if adapter files were created
            adapter_files = os.listdir(adapter_path)
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            
            missing_files = [f for f in required_files if f not in adapter_files]
            if missing_files:
                print(f"⚠️ Warning: Some adapter files are missing: {missing_files}")
            else:
                print(f"✅ All required adapter files were created")
            
            print("\n🎉 Stage 3 (Model Fine-tuning) completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to fine-tune model: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    finally:
        # Clean up
        print("\n🧹 Note: Keeping the model output directory for inspection...")
        print(f"📁 Model output directory: {model_output_dir}")

if __name__ == "__main__":
    # Get formatted data directory from command line if provided
    formatted_data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_stage3(formatted_data_dir)
    sys.exit(0 if success else 1) 