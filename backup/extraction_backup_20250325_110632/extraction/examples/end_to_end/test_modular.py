#!/usr/bin/env python3
"""
Simple test script to verify modular code structure works.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_modular")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

def test_imports():
    """Test that all modules can be imported correctly."""
    success = True
    
    # Test repository_test module
    try:
        from repository_test import RepositoryTester
        logger.info("Successfully imported repository_test")
    except ImportError as e:
        logger.error(f"Failed to import repository_test: {e}")
        success = False
    
    # Test arangodb_validator module
    try:
        from arangodb_validator import ArangoDBDocTest, get_expected_format, validate_arangodb_blocks
        logger.info("Successfully imported arangodb_validator")
    except ImportError as e:
        logger.error(f"Failed to import arangodb_validator: {e}")
        success = False
        
    # Test arangodb_aql_test module
    try:
        from arangodb_aql_test import ArangoDBAQLTest, run_aql_test
        logger.info("Successfully imported arangodb_aql_test")
    except ImportError as e:
        logger.error(f"Failed to import arangodb_aql_test: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    if test_imports():
        logger.info("✅ All modules imported successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some imports failed")
        sys.exit(1)