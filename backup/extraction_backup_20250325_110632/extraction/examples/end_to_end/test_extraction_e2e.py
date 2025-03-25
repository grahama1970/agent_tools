#!/usr/bin/env python3
"""
End-to-End Test for Markdown Extraction.

This script performs a complete end-to-end test of the markdown extraction pipeline:
1. Clones a repository with markdown files
2. Extracts content from the repository
3. Validates the extraction output against expected format
4. Reports results and statistics

This can be used as a blind test to ensure the extraction module works correctly
with repositories it hasn't seen before.
"""

import os
import sys
import json
import shutil
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_extraction_e2e")


def clone_repository(repo_url: str, target_dir: Path) -> bool:
    """
    Clone a Git repository.
    
    Args:
        repo_url: URL of the repository to clone
        target_dir: Directory to clone into
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Cloning repository: {repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        logger.error(f"Stderr: {e.stderr.decode()}")
        return False


def run_extraction(repo_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Run the extraction pipeline on a repository.
    
    Args:
        repo_dir: Path to the repository
        
    Returns:
        Extraction output or None if failed
    """
    try:
        # Add the parent directory to the path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import extraction functions
        from extraction_blocks import extract_all_blocks
        from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
        
        logger.info(f"Extracting from repository: {repo_dir}")
        
        # Extract all blocks
        blocks = extract_all_blocks(repo_dir)
        logger.info(f"Extracted {len(blocks)} blocks")
        
        # Convert to QA-compatible format
        qa_blocks = create_qa_compatible_blocks(blocks)
        
        # Create output
        output = create_qa_compatible_output(qa_blocks)
        
        return output
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return None


def validate_extraction(output: List[Dict[str, Any]], expected_format_path: Path) -> Dict[str, Any]:
    """
    Validate extraction output against expected format.
    
    Args:
        output: Extraction output
        expected_format_path: Path to expected format template
        
    Returns:
        Validation results
    """
    try:
        # Add the parent directory to the path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import validation function
        from validate_extraction_format import validate_extraction_output
        
        # Load expected format
        with open(expected_format_path, 'r', encoding='utf-8') as f:
            expected_format = json.load(f)
        
        # Validate output
        logger.info("Validating extraction output")
        results = validate_extraction_output(output, expected_format)
        
        return results
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return {"valid": False, "errors": [str(e)]}


def find_markdown_files(repo_dir: Path) -> List[Path]:
    """
    Find all markdown files in a repository.
    
    Args:
        repo_dir: Path to the repository
        
    Returns:
        List of paths to markdown files
    """
    return list(repo_dir.glob("**/*.md"))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="End-to-end test for markdown extraction")
    parser.add_argument("--repo", type=str, default="https://github.com/sgl-project/sglang.git",
                        help="URL of the repository to test")
    parser.add_argument("--expected", type=str, 
                        default="/home/grahama/workspace/experiments/agent_tools/test_repos/samples/deepseek_markdown_extraction_example.json",
                        help="Path to expected format JSON template file")
    parser.add_argument("--output", type=str, help="Path to save extraction output")
    parser.add_argument("--keep-repo", action="store_true", help="Keep the cloned repository after testing")
    args = parser.parse_args()
    
    # Create temporary directory for the repository
    temp_dir = Path(tempfile.mkdtemp(prefix="extraction_e2e_test_"))
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Clone repository
        if not clone_repository(args.repo, temp_dir):
            logger.error("Failed to clone repository")
            sys.exit(1)
        
        # List markdown files
        markdown_files = find_markdown_files(temp_dir)
        logger.info(f"Found {len(markdown_files)} markdown files")
        
        # Check for deepseek.md specifically
        deepseek_files = [f for f in markdown_files if "deepseek.md" in str(f)]
        if deepseek_files:
            logger.info(f"Found deepseek.md at: {deepseek_files[0]}")
        else:
            logger.warning("No deepseek.md file found in repository")
        
        # Run extraction
        output = run_extraction(temp_dir)
        if output is None:
            logger.error("Extraction failed")
            sys.exit(1)
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved extraction output to {output_path}")
        
        # Validate output
        results = validate_extraction(output, Path(args.expected))
        
        # Print results
        if results["valid"]:
            logger.info("✅ End-to-end test successful")
            logger.info(f"Statistics: {results['stats']}")
        else:
            logger.error("❌ End-to-end test failed")
            for error in results["errors"]:
                logger.error(f"  - {error}")
            logger.info(f"Statistics: {results.get('stats', {})}")
            sys.exit(1)
        
    finally:
        # Clean up
        if args.keep_repo:
            logger.info(f"Repository kept at: {temp_dir}")
        else:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    # Success
    sys.exit(0)


if __name__ == "__main__":
    main()