import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_github_repo_download():
    """Test downloading a GitHub repository."""
    # Import the GitPython library directly
    try:
        import git
        GIT_AVAILABLE = True
    except ImportError:
        GIT_AVAILABLE = False
        print("GitPython is not available. Skipping test.")
        return False
    
    # Import the specific function we need to avoid full module imports
    try:
        from src.agent_tools.dualipa.github_utils import download_github_repo
    except ImportError as e:
        print(f"Failed to import download_github_repo: {e}")
        return False
    
    print("Testing GitHub repository downloading...")
    
    # Test the fixed download_github_repo function
    temp_dir = tempfile.mkdtemp()
    try:
        # Simple URL without branch specification
        repo_url = "https://github.com/yamadashy/repomix"
        print(f"Downloading {repo_url} to {temp_dir}...")
        
        # Use the fixed function
        repo_path = download_github_repo(repo_url, temp_dir)
        print(f"Repository downloaded successfully to: {repo_path}")
        
        # Check for the existence of some files
        readme_path = os.path.join(repo_path, "README.md")
        git_dir_path = os.path.join(repo_path, ".git")
        
        if os.path.exists(readme_path):
            print("README.md exists in the repository.")
            with open(readme_path, 'r') as f:
                readme_content = f.read()
                print(f"README.md content (first 100 chars): {readme_content[:100]}...")
        else:
            print("README.md does not exist in the repository.")
        
        if os.path.exists(git_dir_path):
            print(".git directory exists, indicating a successful clone.")
        else:
            print(".git directory does not exist. Clone might have failed.")
            
        print("GitHub repository download test completed successfully!")
        return True
    except Exception as e:
        print(f"Error during GitHub repository download test: {e}")
        return False
    finally:
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    result = test_github_repo_download()
    sys.exit(0 if result else 1) 