#!/usr/bin/env python3
"""
Blind test for code block extraction across various repository files.

This script tests our extraction module on multiple repositories and files
without prior knowledge of their content, ensuring robust extraction capabilities.
It also includes specialized blind tests for ArangoDB documentation integration.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("blind_test")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import utility modules
# Import repository test module
try:
    from repository_test import run_repository_tests
    logger.info("Successfully imported repository test module")
except ImportError:
    logger.error("Failed to import repository test module")
    sys.exit(1)


class BlindTest:
    """Run blind tests on repository extraction."""
    
    def __init__(self):
        """Initialize test configuration."""
        # ArangoDB documentation URLs to test
        self.arangodb_doc_urls = {
            "main_aql": "https://docs.arangodb.com/stable/aql/",
            "fundamentals": "https://docs.arangodb.com/stable/aql/fundamentals/",
            "operations": "https://docs.arangodb.com/stable/aql/operations/return/",
            "indexing": "https://docs.arangodb.com/stable/indexing/",
            "array_intersection": "https://docs.arangodb.com/3.12/aql/functions/array/#intersection",
            "length_function": "https://docs.arangodb.com/3.12/aql/functions/string/#length"
        }
    
    def run_tests(self):
        """Run all blind tests."""
        success = True
        
        # Test code repository extraction
        logger.info("Testing repository extraction")
        success &= run_repository_tests()
        
        # Test general ArangoDB documentation extraction
        logger.info("Testing ArangoDB documentation extraction")
        success &= self.test_arangodb_docs_extraction()
        
        # Test specific ArangoDB AQL main page extraction
        logger.info("Testing ArangoDB AQL main page extraction")
        try:
            from arangodb_aql_test import ArangoDBAQLTest
            aql_test = ArangoDBAQLTest()
            success &= aql_test.run_test()
        except ImportError:
            logger.error("Failed to import AQL test module")
            success = False
        
        # Test specific Array Intersection function page
        logger.info("Testing ArangoDB Array Intersection function extraction")
        try:
            from array_intersection_test import ArrayIntersectionTest
            intersection_test = ArrayIntersectionTest()
            success &= intersection_test.run_test()
        except ImportError:
            logger.error("Failed to import Array Intersection test module")
            success = False
            
        # Test specific LENGTH function page
        logger.info("Testing ArangoDB LENGTH function extraction")
        try:
            from length_function_test import run_test as run_length_test
            success &= (run_length_test() == 0)
        except ImportError:
            logger.error("Failed to import LENGTH function test module")
            success = False
        
        return success
    
    def test_arangodb_docs_extraction(self) -> bool:
        """
        Blind test for ArangoDB documentation extraction.
        
        This has been moved to arangodb_validator.py, we just reference it here
        to maintain API compatibility.
        """
        try:
            from arangodb_validator import ArangoDBDocTest
            doc_test = ArangoDBDocTest()
            return doc_test.run_test()
        except ImportError:
            logger.error("Failed to import ArangoDBDocTest class from arangodb_validator")
            return False


def run_blind_test():
    """Run the blind test."""
    test = BlindTest()
    if test.run_tests():
        logger.info("✅ All blind tests passed!")
        return 0
    else:
        logger.error("❌ Some blind tests failed")
        return 1

def run_arangodb_doc_test_only():
    """Run only the ArangoDB documentation test."""
    try:
        from arangodb_validator import ArangoDBDocTest
        doc_test = ArangoDBDocTest()
        if doc_test.run_test():
            logger.info("✅ ArangoDB documentation test passed!")
            return 0
        else:
            logger.error("❌ ArangoDB documentation test failed")
            return 1
    except ImportError:
        logger.error("Failed to import ArangoDBDocTest class from arangodb_validator")
        return 1

def run_aql_doc_test_only():
    """Run only the ArangoDB AQL main documentation test."""
    # Use the specialized AQL test module
    try:
        from arangodb_aql_test import run_aql_test
        return run_aql_test()
    except ImportError:
        logger.error("Failed to import AQL test module")
        return 1

def run_array_intersection_test_only():
    """Run only the ArangoDB Array Intersection function test."""
    try:
        from array_intersection_test import ArrayIntersectionTest
        test = ArrayIntersectionTest()
        if test.run_test():
            logger.info("✅ Array Intersection function test passed!")
            return 0
        else:
            logger.error("❌ Array Intersection function test failed")
            return 1
    except ImportError:
        logger.error("Failed to import Array Intersection test module")
        return 1
        
def run_length_function_test_only():
    """Run only the ArangoDB LENGTH function test."""
    try:
        from length_function_test import run_test as run_length_test
        result = run_length_test()
        if result == 0:
            logger.info("✅ LENGTH function test passed!")
            return 0
        else:
            logger.error("❌ LENGTH function test failed")
            return 1
    except ImportError:
        logger.error("Failed to import LENGTH function test module")
        return 1

if __name__ == "__main__":
    # Check if we should run only a specific test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--arangodb-docs-only":
            sys.exit(run_arangodb_doc_test_only())
        elif sys.argv[1] == "--aql-main-page-only":
            sys.exit(run_aql_doc_test_only())
        elif sys.argv[1] == "--array-intersection-only":
            sys.exit(run_array_intersection_test_only())
        elif sys.argv[1] == "--length-function-only":
            sys.exit(run_length_function_test_only())
        elif sys.argv[1] == "--repo-only":
            sys.exit(0 if run_repository_tests() else 1)
        elif sys.argv[1] == "--markdown-only":
            from repository_test import RepositoryTester
            tester = RepositoryTester()
            sys.exit(0 if tester.test_arangodb_md() else 1)
        elif sys.argv[1] == "--test-markdown-extraction":
            from test_markdown_extraction import run_test
            sys.exit(run_test())
        elif sys.argv[1] == "--ultimate-test":
            import test_ultimate_extraction
            sys.exit(test_ultimate_extraction.main())
    else:
        # Run all blind tests
        sys.exit(run_blind_test())