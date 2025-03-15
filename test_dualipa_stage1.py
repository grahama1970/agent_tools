"""
Test Script for DuaLipa Pipeline Stage 1: Project Setup and Repository Download

This script tests the first stage of the DuaLipa pipeline:
1. Verify project structure and dependencies
2. Run smoke tests for core functionality
3. Test GitHub repository download

Usage:
    python test_dualipa_stage1.py [REPO_URL]
"""

import os
import sys
import tempfile
import importlib
import subprocess
from pathlib import Path
import collections

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_stage1(repo_url=None):
    """Test Stage 1 of the DuaLipa pipeline: Project Setup and Repository Download."""
    print("Testing DuaLipa Pipeline Stage 1: Project Setup and Repository Download")
    print("=" * 80)
    
    results = {
        "project_setup": {"status": "pending", "details": {}},
        "smoke_tests": {"status": "pending", "details": {}},
        "repo_download": {"status": "pending", "details": {}}
    }
    
    # 1. Project Setup Verification
    print("\n📦 1. Project Setup Verification")
    print("-" * 50)
    
    try:
        # Check for essential project directories
        essential_dirs = [
            "src/agent_tools/dualipa",
            "src/agent_tools/dualipa/tests"
        ]
        
        missing_dirs = []
        for directory in essential_dirs:
            if not os.path.isdir(directory):
                missing_dirs.append(directory)
        
        if missing_dirs:
            print(f"❌ Missing essential directories: {', '.join(missing_dirs)}")
            results["project_setup"]["status"] = "failed"
            results["project_setup"]["details"]["missing_dirs"] = missing_dirs
        else:
            print("✅ Essential project directories found")
        
        # Check for essential Python modules
        essential_modules = [
            "github_utils", 
            "code_extractor",
            "format_dataset"
        ]
        
        missing_modules = []
        for module in essential_modules:
            try:
                module_path = f"src.agent_tools.dualipa.{module}"
                importlib.import_module(module_path)
                print(f"✅ Module '{module}' imported successfully")
            except ImportError as e:
                missing_modules.append(module)
                print(f"❌ Failed to import module '{module}': {e}")
        
        if missing_modules:
            results["project_setup"]["status"] = "failed"
            results["project_setup"]["details"]["missing_modules"] = missing_modules
        else:
            results["project_setup"]["status"] = "success"
        
        # Check for essential files
        essential_files = [
            "src/agent_tools/dualipa/github_utils.py",
            "src/agent_tools/dualipa/code_extractor.py",
            "src/agent_tools/dualipa/format_dataset.py",
            "src/agent_tools/dualipa/pipeline.py",
            "src/agent_tools/dualipa/task.md"
        ]
        
        missing_files = []
        for file_path in essential_files:
            if not os.path.isfile(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing essential files: {', '.join(missing_files)}")
            results["project_setup"]["status"] = "failed"
            results["project_setup"]["details"]["missing_files"] = missing_files
        else:
            print("✅ Essential project files found")
            
    except Exception as e:
        print(f"❌ Project setup verification failed: {e}")
        results["project_setup"]["status"] = "failed"
        results["project_setup"]["details"]["error"] = str(e)
    
    # 2. Run Smoke Tests
    print("\n🔍 2. Running Smoke Tests")
    print("-" * 50)
    
    try:
        # Verify GitPython availability
        try:
            import git
            print("✅ GitPython is available")
            results["smoke_tests"]["details"]["git_available"] = True
        except ImportError:
            print("❌ GitPython is not available")
            results["smoke_tests"]["details"]["git_available"] = False
        
        # Check if essential functions exist and are callable
        from src.agent_tools.dualipa.github_utils import download_github_repo, parse_github_url
        
        smoke_test_functions = [
            (download_github_repo, "download_github_repo"),
            (parse_github_url, "parse_github_url")
        ]
        
        missing_functions = []
        for func, name in smoke_test_functions:
            if not callable(func):
                missing_functions.append(name)
                print(f"❌ Function '{name}' is not callable")
            else:
                print(f"✅ Function '{name}' is available and callable")
        
        if missing_functions:
            results["smoke_tests"]["status"] = "failed"
            results["smoke_tests"]["details"]["missing_functions"] = missing_functions
        else:
            # Perform a basic smoke test of parse_github_url
            test_url = "https://github.com/yamadashy/repomix"
            try:
                owner, repo, path, branch = parse_github_url(test_url)
                print(f"✅ parse_github_url returned: owner='{owner}', repo='{repo}', path='{path}', branch='{branch}'")
                # Just validate that we get some reasonable output, not specific values
                # as the implementation might change
                if owner and repo:
                    print(f"✅ parse_github_url works correctly")
                    results["smoke_tests"]["status"] = "success"
                else:
                    print(f"❌ parse_github_url returned unexpected empty values")
                    results["smoke_tests"]["status"] = "failed"
                    results["smoke_tests"]["details"]["parse_url_error"] = f"Got empty values for owner/repo"
            except Exception as e:
                print(f"❌ parse_github_url failed: {e}")
                results["smoke_tests"]["status"] = "failed"
                results["smoke_tests"]["details"]["parse_url_error"] = str(e)
        
    except Exception as e:
        print(f"❌ Smoke tests failed: {e}")
        results["smoke_tests"]["status"] = "failed"
        results["smoke_tests"]["details"]["error"] = str(e)
    
    # 3. Test GitHub Repository Download
    print("\n📥 3. Testing GitHub Repository Download")
    print("-" * 50)
    
    if not repo_url:
        # Default repo URL if none provided
        repo_url = "https://github.com/yamadashy/repomix"
        print(f"No repository URL provided, using default: {repo_url}")
    else:
        print(f"Testing with repository: {repo_url}")
    
    # Create temporary output directory
    temp_output_dir = tempfile.mkdtemp(prefix="dualipa_download_")
    print(f"Output directory: {temp_output_dir}")
    
    try:
        # Import the required module
        from src.agent_tools.dualipa.github_utils import download_github_repo
        
        # Download the repository
        print("Downloading repository...")
        start_time = __import__('time').time()
        download_path = download_github_repo(repo_url, temp_output_dir)
        end_time = __import__('time').time()
        
        if not download_path or not os.path.exists(download_path):
            print("❌ Repository download failed - no valid path returned")
            results["repo_download"]["status"] = "failed"
            results["repo_download"]["details"]["error"] = "No valid download path returned"
            return False
            
        print(f"✅ Repository downloaded to: {download_path}")
        print(f"⏱️ Download completed in {end_time - start_time:.2f} seconds")
        
        # Check if download was successful by examining the directory
        if not os.path.isdir(download_path):
            print("❌ Download path is not a directory")
            results["repo_download"]["status"] = "failed"
            results["repo_download"]["details"]["error"] = "Download path is not a directory"
            return False
            
        # Check if .git directory exists (indicates successful clone)
        if not os.path.exists(os.path.join(download_path, ".git")):
            print("⚠️ No .git directory found, may not be a complete repository")
        
        # Count files and analyze repository
        file_count = sum(1 for _ in Path(download_path).rglob('*') if _.is_file())
        repo_size = sum(os.path.getsize(f) for f in Path(download_path).rglob('*') if f.is_file())
        
        # Count files by extension
        file_extensions = collections.Counter()
        documentation_files = 0
        code_files = 0
        
        # Define file type groups
        doc_extensions = {'.md', '.rst', '.txt', '.docx', '.pdf'}
        code_extensions = {
            'python': {'.py', '.pyx', '.pyw', '.ipynb'},
            'javascript': {'.js', '.jsx', '.mjs'},
            'typescript': {'.ts', '.tsx'},
            'web': {'.html', '.css', '.scss', '.sass'},
            'c_cpp': {'.c', '.cpp', '.cc', '.h', '.hpp'},
            'java': {'.java'},
            'other_code': {'.go', '.rb', '.php', '.swift', '.rs', '.sh', '.bat', '.ps1'}
        }
        
        # Track code files by language
        code_by_language = {lang: 0 for lang in code_extensions}
        
        for file_path in Path(download_path).rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_extensions[ext] += 1
                
                # Count documentation files
                if ext in doc_extensions:
                    documentation_files += 1
                
                # Count code files by language
                for lang, extensions in code_extensions.items():
                    if ext in extensions:
                        code_files += 1
                        code_by_language[lang] += 1
                        break
        
        print(f"📊 Repository contains {file_count} files ({repo_size / (1024*1024):.2f} MB)")
        print(f"📄 Documentation files: {documentation_files}")
        print(f"💻 Code files: {code_files}")
        
        # Print code files by language
        for lang, count in code_by_language.items():
            if count > 0:
                print(f"  - {lang.replace('_', ' ').title()}: {count}")
        
        # Top file extensions
        top_extensions = file_extensions.most_common(10)
        print(f"📊 Top file extensions:")
        for ext, count in top_extensions:
            if ext and count > 0:  # Skip empty extensions
                print(f"  - {ext}: {count}")
        
        readme_path = os.path.join(download_path, "README.md")
        if os.path.exists(readme_path):
            print(f"📄 README.md found ({os.path.getsize(readme_path)} bytes)")
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read(300)  # Read first 300 chars
                print(f"📖 README preview: {readme_content.strip()[:200]}...")
            except Exception as e:
                print(f"⚠️ Could not read README: {e}")
        
        results["repo_download"]["status"] = "success"
        results["repo_download"]["details"]["file_count"] = file_count
        results["repo_download"]["details"]["repo_size"] = repo_size
        results["repo_download"]["details"]["download_path"] = download_path
        results["repo_download"]["details"]["file_extensions"] = dict(file_extensions.most_common(20))
        results["repo_download"]["details"]["documentation_files"] = documentation_files
        results["repo_download"]["details"]["code_files"] = code_files
        results["repo_download"]["details"]["code_by_language"] = code_by_language
        
    except Exception as e:
        print(f"❌ Repository download failed: {e}")
        import traceback
        traceback.print_exc()
        results["repo_download"]["status"] = "failed"
        results["repo_download"]["details"]["error"] = str(e)
        return False
    finally:
        # Don't clean up - keep the downloaded repository for potential use in stage 2
        print(f"📁 Downloaded repository available at: {temp_output_dir}")
    
    # Overall Results Summary
    print("\n📋 Stage 1 Results Summary")
    print("=" * 80)
    
    all_success = True
    for name, result in results.items():
        status = result["status"]
        status_symbol = "✅" if status == "success" else "❌"
        print(f"{status_symbol} {name.replace('_', ' ').title()}: {status.upper()}")
        all_success = all_success and (status == "success")
    
    if all_success:
        print("\n🎉 Stage 1 completed successfully!")
        return True
    else:
        print("\n⚠️ Stage 1 completed with issues")
        return False

if __name__ == "__main__":
    # Get repository URL from command line if provided
    repo_url = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_stage1(repo_url)
    sys.exit(0 if success else 1) 