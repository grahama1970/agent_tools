#!/usr/bin/env python3
"""
Blind test for code block extraction across various repository files.

This script tests our extraction module on multiple repositories and files
without prior knowledge of their content, ensuring robust extraction capabilities.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("blind_test")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions
from extraction_blocks import extract_all_blocks
from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output

class BlindTest:
    """Run blind tests on repository extraction."""
    
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
        }
    
    def run_tests(self):
        """Run all blind tests."""
        success = True
        
        # Test JavaScript extraction
        logger.info("Testing JavaScript extraction")
        repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb")
        success &= self.test_repository(repo_path, "arangodb_js")
        
        # Test Python extraction
        logger.info("Testing Python extraction")
        repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/sglang")
        success &= self.test_repository(repo_path, "sglang_py")
        
        return success
    
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
            
            # Check function count
            function_count = block_types.get("function", 0) + block_types.get("method", 0)
            if function_count < expected["functions"]:
                logger.error(f"Expected at least {expected['functions']} functions, found {function_count}")
                return False
                
            # Check class count
            class_count = block_types.get("class", 0)
            if class_count < expected["classes"]:
                logger.error(f"Expected at least {expected['classes']} classes, found {class_count}")
                return False
        
        logger.info(f"✅ Successfully verified {Path(file_path).name}")
        return True

if __name__ == "__main__":
    # Run blind tests
    test = BlindTest()
    if test.run_tests():
        logger.info("✅ All blind tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some blind tests failed")
        sys.exit(1)