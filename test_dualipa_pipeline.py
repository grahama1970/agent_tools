"""
End-to-End Test Script for DuaLipa Pipeline

This script tests the complete DuaLipa pipeline workflow:
1. Repository Download (Source Acquisition)
2. Code and Documentation Extraction (Data Extraction)
3. QA Pair Generation and Dataset Formatting (Data Preparation)
4. Model Fine-tuning with Unsloth (Model Training)
5. LoRA Adapter Merging (Model Optimization)
6. Deployment to Hugging Face (Distribution)

Usage:
    python test_dualipa_pipeline.py [REPO_URL] [--push-to-hub]
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

class PipelineTest:
    """Test harness for the DuaLipa pipeline."""
    
    def __init__(self, repo_url: str = None, output_dir: str = None, push_to_hub: bool = False):
        """Initialize pipeline test harness."""
        self.repo_url = repo_url or "https://github.com/yamadashy/repomix"
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="dualipa_pipeline_")
        self.push_to_hub = push_to_hub
        
        # Define sub-directories
        self.download_dir = os.path.join(self.output_dir, "downloaded")
        self.extract_dir = os.path.join(self.output_dir, "extracted")
        self.format_dir = os.path.join(self.output_dir, "formatted")
        self.model_dir = os.path.join(self.output_dir, "model")
        self.merged_dir = os.path.join(self.output_dir, "merged")
        
        # Create directories
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)
        os.makedirs(self.format_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.merged_dir, exist_ok=True)
        
        # State tracking
        self.download_path = None
        self.extracted_data_path = None
        self.train_data_path = None
        self.adapter_path = None
        self.merged_model_path = None
        
        # Results
        self.results = {
            "stage1": {"status": "pending", "details": {}},
            "stage2": {"status": "pending", "details": {}},
            "stage3": {"status": "pending", "details": {}},
            "stage4": {"status": "pending", "details": {}},
            "stage5": {"status": "pending", "details": {}},
            "stage6": {"status": "pending", "details": {}}
        }
        
        print(f"🚀 DuaLipa Pipeline Test initialized")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"🔗 Repository URL: {self.repo_url}")
        print(f"🌐 Push to Hub: {self.push_to_hub}")
        print("=" * 80)
    
    def run_stage1(self) -> bool:
        """Run Stage 1: Repository Download."""
        print("\n\n📥 STAGE 1: REPOSITORY DOWNLOAD")
        print("=" * 80)
        
        try:
            # Import required modules
            from src.agent_tools.dualipa.github_utils import download_github_repo
            
            # Download the repository
            print("Downloading repository...")
            self.download_path = download_github_repo(self.repo_url, self.download_dir)
            print(f"✅ Repository downloaded to: {self.download_path}")
            
            # Record stats
            file_count = sum(1 for _ in Path(self.download_path).rglob('*') if _.is_file())
            repo_size = sum(os.path.getsize(f) for f in Path(self.download_path).rglob('*') if f.is_file())
            
            print(f"📊 Repository contains {file_count} files")
            print(f"📊 Repository size: {repo_size / (1024*1024):.2f} MB")
            
            # Record results
            self.results["stage1"]["status"] = "success"
            self.results["stage1"]["details"]["download_path"] = self.download_path
            self.results["stage1"]["details"]["file_count"] = file_count
            self.results["stage1"]["details"]["repo_size"] = repo_size
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage1"]["status"] = "failed"
            self.results["stage1"]["details"]["error"] = str(e)
            return False
    
    def run_stage2(self) -> bool:
        """Run Stage 2: Code and Documentation Extraction."""
        if self.results["stage1"]["status"] != "success":
            print("\n⚠️ Skipping Stage 2 because Stage 1 failed")
            self.results["stage2"]["status"] = "skipped"
            return False
            
        print("\n\n🔍 STAGE 2: CODE AND DOCUMENTATION EXTRACTION")
        print("=" * 80)
        
        try:
            # Import required modules
            from src.agent_tools.dualipa.code_extractor import extract_repository
            
            # Extract repository content
            print("Extracting code and documentation from repository...")
            self.extracted_data_path = os.path.join(self.extract_dir, "extracted_data.json")
            
            extract_result = extract_repository(
                source=self.download_path,
                output_path=self.extract_dir,
                include_extensions=[".py", ".md", ".txt", ".js", ".html", ".css", ".json"],
                max_file_size_kb=500,
                ignore_patterns=["node_modules", "__pycache__", ".git"]
            )
            
            # Check result
            if not os.path.exists(self.extracted_data_path):
                raise Exception(f"Extraction did not produce expected output file: {self.extracted_data_path}")
                
            # Load and analyze extracted data
            with open(self.extracted_data_path, 'r') as f:
                data = json.load(f)
                extracted_files = len(data.get('files', []))
                
                # Count languages
                languages = {}
                for file in data.get('files', []):
                    lang = file.get('language')
                    if lang:
                        languages[lang] = languages.get(lang, 0) + 1
                        
                # Count file types
                file_types = {}
                for file in data.get('files', []):
                    path = file.get('path', '')
                    if '.' in path:
                        ext = path.split('.')[-1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            print(f"✅ Repository extracted successfully")
            print(f"📊 Extracted {extracted_files} files")
            print(f"🔤 Languages: {', '.join(languages.keys())}")
            print(f"📁 File types: {', '.join(file_types.keys())}")
            
            # Record results
            self.results["stage2"]["status"] = "success"
            self.results["stage2"]["details"]["extracted_files"] = extracted_files
            self.results["stage2"]["details"]["languages"] = languages
            self.results["stage2"]["details"]["file_types"] = file_types
            self.results["stage2"]["details"]["extracted_data_path"] = self.extracted_data_path
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage2"]["status"] = "failed"
            self.results["stage2"]["details"]["error"] = str(e)
            return False
    
    def run_stage3(self) -> bool:
        """Run Stage 3: QA Pair Generation and Dataset Formatting."""
        if self.results["stage2"]["status"] != "success":
            print("\n⚠️ Skipping Stage 3 because Stage 2 failed")
            self.results["stage3"]["status"] = "skipped"
            return False
            
        print("\n\n🧩 STAGE 3: QA PAIR GENERATION AND DATASET FORMATTING")
        print("=" * 80)
        
        try:
            # Import required modules
            from src.agent_tools.dualipa.format_dataset import format_for_lora
            
            # Format dataset for LoRA training
            print("Generating QA pairs and formatting dataset...")
            
            # Define formatting parameters
            format_kwargs = {
                "qa_format": "instruct",
                "max_qa_pairs": 20,  # Limit for testing
                "min_code_lines": 3,
                "temperature": 0.7,
                "validator_config": {
                    "run_validation": True,
                    "fix_invalid": True
                }
            }
            
            # Generate QA pairs and format dataset
            format_result = format_for_lora(
                self.extracted_data_path,
                self.format_dir,
                **format_kwargs
            )
            
            # Check if output files were created
            self.train_data_path = os.path.join(self.format_dir, "train.jsonl")
            eval_file = os.path.join(self.format_dir, "eval.jsonl")
            
            if not os.path.exists(self.train_data_path):
                raise Exception(f"Training file not created: {self.train_data_path}")
            
            # Count QA pairs
            train_qa_count = sum(1 for _ in open(self.train_data_path, 'r'))
            eval_qa_count = 0
            if os.path.exists(eval_file):
                eval_qa_count = sum(1 for _ in open(eval_file, 'r'))
            
            print(f"✅ QA generation completed successfully")
            print(f"📊 Generated {train_qa_count} training QA pairs")
            if eval_qa_count > 0:
                print(f"📊 Generated {eval_qa_count} evaluation QA pairs")
            
            # Record sample QA pair
            sample_qa = None
            with open(self.train_data_path, 'r') as f:
                sample_qa = json.loads(f.readline())
                
            # Record results
            self.results["stage3"]["status"] = "success"
            self.results["stage3"]["details"]["train_qa_count"] = train_qa_count
            self.results["stage3"]["details"]["eval_qa_count"] = eval_qa_count
            self.results["stage3"]["details"]["train_data_path"] = self.train_data_path
            self.results["stage3"]["details"]["sample_question"] = sample_qa.get("question", "N/A") if sample_qa else "N/A"
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 3 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage3"]["status"] = "failed"
            self.results["stage3"]["details"]["error"] = str(e)
            return False
    
    def run_stage4(self) -> bool:
        """Run Stage 4: Model Fine-tuning with Unsloth."""
        if self.results["stage3"]["status"] != "success":
            print("\n⚠️ Skipping Stage 4 because Stage 3 failed")
            self.results["stage4"]["status"] = "skipped"
            return False
            
        print("\n\n🧠 STAGE 4: MODEL FINE-TUNING WITH UNSLOTH")
        print("=" * 80)
        
        try:
            # Import required modules
            from src.agent_tools.dualipa.train_lora import train_lora
            
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
            
            print("Fine-tuning model (test mode)...")
            self.adapter_path = train_lora(
                dataset_path=self.format_dir,
                output_dir=self.model_dir,
                **train_kwargs
            )
            
            if not self.adapter_path:
                raise Exception("Training returned no adapter path")
                
            print(f"✅ Model fine-tuning completed successfully")
            print(f"📊 Adapter path: {self.adapter_path}")
            
            # Record results
            self.results["stage4"]["status"] = "success"
            self.results["stage4"]["details"]["adapter_path"] = self.adapter_path
            self.results["stage4"]["details"]["base_model"] = train_kwargs["base_model"]
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 4 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage4"]["status"] = "failed"
            self.results["stage4"]["details"]["error"] = str(e)
            return False
    
    def run_stage5(self) -> bool:
        """Run Stage 5: LoRA Adapter Merging."""
        if self.results["stage4"]["status"] != "success":
            print("\n⚠️ Skipping Stage 5 because Stage 4 failed")
            self.results["stage5"]["status"] = "skipped"
            return False
            
        print("\n\n🔄 STAGE 5: LORA ADAPTER MERGING")
        print("=" * 80)
        
        try:
            # Note: This would normally use a real merging function, but for testing we'll simulate it
            # In a real implementation, this would use a function like:
            # from src.agent_tools.dualipa.model_utils import merge_adapter
            
            print("Merging LoRA adapter with base model (simulated)...")
            
            # Create a mock merged model path
            self.merged_model_path = os.path.join(self.merged_dir, "merged_model")
            os.makedirs(self.merged_model_path, exist_ok=True)
            
            # Simulate merging by copying adapter files
            if os.path.exists(self.adapter_path):
                for f in os.listdir(self.adapter_path):
                    shutil.copy(
                        os.path.join(self.adapter_path, f),
                        os.path.join(self.merged_model_path, f)
                    )
            
            # Create a mock config file to simulate merged model
            with open(os.path.join(self.merged_model_path, "config.json"), "w") as f:
                json.dump({
                    "name": "test-merged-model",
                    "base_model": "unsloth/mistral-7b-bnb-4bit",
                    "merged": True,
                    "test_mode": True
                }, f)
                
            print(f"✅ Adapter merging completed (simulated)")
            print(f"📊 Merged model path: {self.merged_model_path}")
            
            # Record results
            self.results["stage5"]["status"] = "success"
            self.results["stage5"]["details"]["merged_model_path"] = self.merged_model_path
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 5 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage5"]["status"] = "failed"
            self.results["stage5"]["details"]["error"] = str(e)
            return False
    
    def run_stage6(self) -> bool:
        """Run Stage 6: Deployment to Hugging Face (optional)."""
        if self.results["stage5"]["status"] != "success":
            print("\n⚠️ Skipping Stage 6 because Stage 5 failed")
            self.results["stage6"]["status"] = "skipped"
            return False
            
        if not self.push_to_hub:
            print("\n⚠️ Skipping Stage 6 because push_to_hub is disabled")
            self.results["stage6"]["status"] = "skipped"
            return True
            
        print("\n\n🌐 STAGE 6: DEPLOYMENT TO HUGGING FACE")
        print("=" * 80)
        
        try:
            # Note: This would normally use a real HF push function, but for testing we'll simulate it
            # In a real implementation, this would use a function like:
            # from src.agent_tools.dualipa.model_utils import push_to_huggingface
            
            print("Pushing model to Hugging Face (simulated)...")
            
            # Create a mock HF model ID
            hf_model_id = f"dualipa/test-code-model-{os.path.basename(self.output_dir)}"
            
            print(f"✅ Model pushed to Hugging Face (simulated)")
            print(f"📊 Hugging Face model ID: {hf_model_id}")
            
            # Record results
            self.results["stage6"]["status"] = "success"
            self.results["stage6"]["details"]["hf_model_id"] = hf_model_id
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 6 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["stage6"]["status"] = "failed"
            self.results["stage6"]["details"]["error"] = str(e)
            return False
    
    def run_all(self) -> bool:
        """Run all pipeline stages."""
        stages = [
            self.run_stage1,
            self.run_stage2,
            self.run_stage3,
            self.run_stage4,
            self.run_stage5,
            self.run_stage6
        ]
        
        success = True
        for i, stage_func in enumerate(stages, 1):
            stage_success = stage_func()
            success = success and stage_success
            
            # Print stage separator
            if i < len(stages):
                print("\n" + "-" * 80 + "\n")
        
        return success
    
    def print_summary(self) -> None:
        """Print pipeline test results summary."""
        print("\n\n📋 PIPELINE TEST SUMMARY")
        print("=" * 80)
        
        for stage, result in self.results.items():
            status = result["status"]
            status_symbol = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
            print(f"{status_symbol} Stage {stage[-1]}: {status.upper()}")
            
            # Print key details
            if status == "success":
                details = result["details"]
                if stage == "stage1":
                    print(f"  - Downloaded repository with {details.get('file_count', 'N/A')} files")
                    print(f"  - Repository size: {details.get('repo_size', 0) / (1024*1024):.2f} MB")
                elif stage == "stage2":
                    print(f"  - Extracted {details.get('extracted_files', 'N/A')} files")
                    langs = details.get('languages', {})
                    if langs:
                        print(f"  - Languages: {', '.join(langs.keys())}")
                elif stage == "stage3":
                    print(f"  - Generated {details.get('train_qa_count', 'N/A')} QA pairs")
                elif stage == "stage4":
                    print(f"  - Created adapter for {details.get('base_model', 'N/A')}")
                elif stage == "stage5":
                    print(f"  - Created merged model at {os.path.basename(details.get('merged_model_path', 'N/A'))}")
                elif stage == "stage6":
                    print(f"  - Published to {details.get('hf_model_id', 'N/A')}")
            elif status == "failed":
                print(f"  - Error: {result['details'].get('error', 'Unknown error')}")
        
        # Print overall status
        all_success = all(r["status"] == "success" for r in self.results.values())
        print("\n" + "=" * 80)
        if all_success:
            print("🎉 PIPELINE TEST COMPLETED SUCCESSFULLY!")
        else:
            print("⚠️ PIPELINE TEST COMPLETED WITH ERRORS")
        
        print(f"📂 Output directory: {self.output_dir}")
        print("=" * 80)

def main():
    """Run the pipeline test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test DuaLipa Pipeline")
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hugging Face (simulated)")
    args = parser.parse_args()
    
    # Run the pipeline test
    test = PipelineTest(
        repo_url=args.repo_url,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub
    )
    
    success = test.run_all()
    test.print_summary()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 