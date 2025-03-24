#!/usr/bin/env python3
"""
Run specific extraction tests for ArangoDB documentation.

This script provides options to run various specialized tests for ArangoDB
documentation extraction, including the Array Intersection function test and
LENGTH function test.

Example usage:
    python run_specific_tests.py --all
    python run_specific_tests.py --array-intersection
    python run_specific_tests.py --length-function
    python run_specific_tests.py --aql-main

Author: Claude AI
Created: 2025-03-24
"""

import os
import sys
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_specific_tests")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def run_array_intersection_test():
    """Run the ArangoDB Array Intersection function test."""
    try:
        from array_intersection_test import run_test as run_array_intersection
        logger.info("Running Array Intersection function test...")
        return run_array_intersection() == 0
    except ImportError as e:
        logger.error(f"Failed to import array_intersection_test: {e}")
        return False


def run_aql_main_test():
    """Run the ArangoDB AQL main page test."""
    try:
        from arangodb_aql_test import run_aql_test
        logger.info("Running AQL main page test...")
        return run_aql_test() == 0
    except ImportError as e:
        logger.error(f"Failed to import arangodb_aql_test: {e}")
        return False


def run_length_function_test():
    """Run the ArangoDB LENGTH function test."""
    try:
        from length_function_test import run_test as run_length_function
        logger.info("Running LENGTH function test...")
        return run_length_function() == 0
    except ImportError as e:
        logger.error(f"Failed to import length_function_test: {e}")
        return False


def run_arangodb_docs_test():
    """Run the general ArangoDB documentation test."""
    try:
        from arangodb_validator import ArangoDBDocTest
        logger.info("Running general ArangoDB documentation test...")
        test = ArangoDBDocTest()
        return test.run_test()
    except ImportError as e:
        logger.error(f"Failed to import ArangoDBDocTest: {e}")
        return False


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run specific ArangoDB documentation extraction tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--array-intersection", action="store_true", help="Run Array Intersection function test")
    parser.add_argument("--length-function", action="store_true", help="Run LENGTH function test")
    parser.add_argument("--aql-main", action="store_true", help="Run AQL main page test")
    parser.add_argument("--general-docs", action="store_true", help="Run general ArangoDB documentation test")
    
    args = parser.parse_args()
    
    # If no args provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    results = {}
    
    # Run Array Intersection test
    if args.all or args.array_intersection:
        results["array_intersection"] = run_array_intersection_test()
    
    # Run LENGTH function test
    if args.all or args.length_function:
        results["length_function"] = run_length_function_test()
        
    # Run AQL main page test
    if args.all or args.aql_main:
        results["aql_main"] = run_aql_main_test()
    
    # Run general ArangoDB documentation test
    if args.all or args.general_docs:
        results["general_docs"] = run_arangodb_docs_test()
    
    # Print summary
    logger.info("\n--- Test Results Summary ---")
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        all_passed &= result
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())