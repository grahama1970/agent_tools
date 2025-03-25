#!/usr/bin/env python3
"""
Repository-specific testing module for code block extraction.

This module contains functionality for testing extraction on specific repositories,
separating concerns from the main blind test framework.

Functions:
    RepositoryTester: Class for testing extraction on specific repositories

Author: Claude AI
Created: 2025-03-22
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("repository_test")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions
from extraction_blocks import extract_all_blocks


class RepositoryTester:
    """Test extraction on specific repositories."""
    
    def __init__(self):
        """Initialize test configuration."""
        # Files to verify in the blind test
        self.test_files = {
            # JavaScript files
            "arangodb_js": [
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/analyzers.js",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/arango-database.js",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/aql/functions.js",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/foxx/router/router.js"
            ],
            # Python files
            "sglang_py": [
                "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/frontend_language/quick_start/openai_example_chat.py",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/embedding.py",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/custom_server.py",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/offline_batch_inference_async.py"
            ],
            # Markdown files
            "arangodb_md": [
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/ERROR_LEVELS.md",
                "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/README.md"
            ]
        }
        
        # Expected minimum function/class counts for each file
        self.expected_counts = {
            # File path => {"functions": N, "classes": M}
            "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/analyzers.js": 
                {"functions": 1, "classes": 0},
            "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/frontend_language/quick_start/openai_example_chat.py": 
                {"functions": 3, "classes": 0},
            "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/offline_batch_inference_async.py": 
                {"functions": 1, "classes": 1},
                
            # Expected section counts for markdown files
            "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/ERROR_LEVELS.md":
                {"sections": 6},  # 6 main sections for the error levels
            "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/README.md":
                {"sections": 2}  # 2 sections detected in the README
        }
    
    def test_repository(self, repo_path: Path, repo_key: str) -> bool:
        """Test extraction on a repository."""
        # Extract blocks from the repository
        logger.info(f"Extracting blocks from {repo_path}")
        all_blocks = extract_all_blocks(repo_path)
        
        # Check if each test file was properly extracted
        success = True
        for file_path in self.test_files[repo_key]:
            file_success = self.verify_file_extraction(file_path, all_blocks)
            success &= file_success
            
            if not file_success:
                logger.error(f"❌ Failed extraction for {file_path}")
        
        logger.info(f"Extracted {len(all_blocks)} blocks from {repo_path}")
        return success
    
    def verify_file_extraction(self, file_path: str, all_blocks: list) -> bool:
        """Verify that a specific file was properly extracted."""
        # Find file block
        file_blocks = [b for b in all_blocks if b.get("file_path") == file_path and b.get("type") == "file"]
        if not file_blocks:
            logger.error(f"File block not found for {file_path}")
            return False
        
        file_block = file_blocks[0]
        logger.info(f"Verifying extraction for {Path(file_path).name}")
        
        # Count blocks by type
        block_types = {}
        file_blocks = [b for b in all_blocks if b.get("file_path") == file_path]
        
        for block in file_blocks:
            block_type = block.get("type")
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        logger.info(f"Found block types: {block_types}")
        
        # Verify against expected counts if available
        if file_path in self.expected_counts:
            expected = self.expected_counts[file_path]
            
            # Check for markdown sections
            if "sections" in expected:
                section_count = block_types.get("section", 0)
                if section_count < expected["sections"]:
                    logger.error(f"Expected at least {expected['sections']} sections, found {section_count}")
                    return False
                
                # If it's a markdown file, also check for text blocks, code blocks, etc.
                has_text_blocks = block_types.get("text", 0) > 0
                if not has_text_blocks and file_path.endswith(".md"):
                    logger.warning(f"Markdown file {file_path} has no text blocks")
                
                # Check for other markdown elements
                if file_path.endswith(".md"):
                    logger.info(f"Checking markdown elements in {Path(file_path).name}")
                    # List child UUIDs of the file block
                    child_uuids = file_block.get("child_uuids", [])
                    # Count child blocks
                    child_blocks = [b for b in all_blocks if b.get("uuid") in child_uuids]
                    logger.info(f"Found {len(child_blocks)} child blocks for {Path(file_path).name}")
            else:
                # Check function count for code files
                function_count = block_types.get("function", 0) + block_types.get("method", 0)
                if "functions" in expected and function_count < expected["functions"]:
                    logger.error(f"Expected at least {expected['functions']} functions, found {function_count}")
                    return False
                    
                # Check class count for code files
                class_count = block_types.get("class", 0)
                if "classes" in expected and class_count < expected["classes"]:
                    logger.error(f"Expected at least {expected['classes']} classes, found {class_count}")
                    return False
        
        logger.info(f"✅ Successfully verified {Path(file_path).name}")
        return True
    
    def test_arangodb_js(self) -> bool:
        """Test JavaScript extraction from ArangoDB repo."""
        logger.info("Testing JavaScript extraction")
        repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb")
        return self.test_repository(repo_path, "arangodb_js")
    
    def test_sglang_py(self) -> bool:
        """Test Python extraction from SGLang repo."""
        logger.info("Testing Python extraction")
        repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/sglang")
        return self.test_repository(repo_path, "sglang_py")
        
    def test_arangodb_md(self) -> bool:
        """Test Markdown extraction from ArangoDB repo."""
        logger.info("Testing Markdown extraction")
        repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb")
        return self.test_repository(repo_path, "arangodb_md")


def run_repository_tests():
    """Run all repository tests."""
    tester = RepositoryTester()
    success = True
    
    # Test JavaScript extraction
    success &= tester.test_arangodb_js()
    
    # Test Python extraction
    success &= tester.test_sglang_py()
    
    # Test Markdown extraction
    success &= tester.test_arangodb_md()
    
    return success


if __name__ == "__main__":
    # Run all repository tests
    if run_repository_tests():
        logger.info("✅ All repository tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some repository tests failed")
        sys.exit(1)