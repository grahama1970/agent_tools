#!/usr/bin/env python3
"""
CLI tool to validate code against package documentation.
Uses ArangoDB as an example but can be extended for other packages.
"""

import sys
import inspect
import argparse
from typing import Dict, Any, Set
import importlib
from loguru import logger
from arango import ArangoClient
from arango.exceptions import ArangoError


def get_documented_methods() -> Dict[str, str]:
    """Get documented ArangoDB methods from official docs."""
    methods = {
        "db.collection('name').insert_many": "Batch insert multiple documents",
        "db.collection('name').get": "Retrieve a single document by key",
        "db.aql.execute": "Execute an AQL query",
        # Add more from docs.python-arango.com
    }
    return methods


def get_documented_errors() -> Dict[str, str]:
    """Get documented ArangoDB errors from official docs."""
    errors = {
        "ArangoError": "Base class for all exceptions",
        "DocumentInsertError": "Failed to insert document",
        "DocumentGetError": "Failed to retrieve document",
        # Add more from docs.python-arango.com/errors.html
    }
    return errors


def extract_methods_from_code(file_path: str) -> Set[str]:
    """Extract method calls from a Python file."""
    with open(file_path, "r") as f:
        content = f.read()

    # This is a simple extraction - could be improved with ast
    methods = set()
    for line in content.split("\n"):
        if "db.collection" in line and "(" in line:
            # Extract method name
            method = line.split("(")[0].strip()
            methods.add(method)
        elif "db.aql.execute" in line:
            methods.add("db.aql.execute")

    return methods


def extract_errors_from_code(file_path: str) -> Set[str]:
    """Extract error handling from a Python file."""
    with open(file_path, "r") as f:
        content = f.read()

    errors = set()
    for line in content.split("\n"):
        if "except" in line:
            # Extract error type
            if "(" in line:
                error = line.split("(")[0].replace("except", "").strip()
                errors.add(error)

    return errors


def validate_code(file_path: str) -> Dict[str, Any]:
    """Validate a Python file against ArangoDB documentation."""
    # Get documented items
    doc_methods = get_documented_methods()
    doc_errors = get_documented_errors()

    # Extract from code
    code_methods = extract_methods_from_code(file_path)
    code_errors = extract_errors_from_code(file_path)

    # Find undocumented usage
    undoc_methods = {m for m in code_methods if m not in doc_methods}
    undoc_errors = {e for e in code_errors if e not in doc_errors}

    # Find unused documented features
    unused_methods = {m for m in doc_methods if m not in code_methods}
    unused_errors = {e for e in doc_errors if e not in code_errors}

    return {
        "undocumented_methods": undoc_methods,
        "undocumented_errors": undoc_errors,
        "unused_documented_methods": unused_methods,
        "unused_documented_errors": unused_errors,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate code against package documentation"
    )
    parser.add_argument("file", help="Python file to validate")
    parser.add_argument(
        "--show-docs",
        action="store_true",
        help="Show available documented methods and errors",
    )

    args = parser.parse_args()

    if args.show_docs:
        logger.info("Documented Methods:")
        for method, desc in get_documented_methods().items():
            logger.info(f"  {method}: {desc}")

        logger.info("\nDocumented Errors:")
        for error, desc in get_documented_errors().items():
            logger.info(f"  {error}: {desc}")
        return

    results = validate_code(args.file)

    if results["undocumented_methods"]:
        logger.warning("Methods used but not in documentation:")
        for method in results["undocumented_methods"]:
            logger.warning(f"  {method}")

    if results["undocumented_errors"]:
        logger.warning("Errors used but not in documentation:")
        for error in results["undocumented_errors"]:
            logger.warning(f"  {error}")

    if results["unused_documented_methods"]:
        logger.info("Available documented methods not used:")
        for method in results["unused_documented_methods"]:
            logger.info(f"  {method}")

    if results["unused_documented_errors"]:
        logger.info("Available documented errors not used:")
        for error in results["unused_documented_errors"]:
            logger.info(f"  {error}")


if __name__ == "__main__":
    main()
