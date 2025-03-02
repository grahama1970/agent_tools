#!/usr/bin/env python3
"""
Method Validator CLI - Analyze and validate Python package methods.

This tool helps AI agents verify methods and check for existing functionality BEFORE suggesting code.

AGENT WORKFLOW:
1. Draft the code first.
2. Extract methods you plan to use.
3. Verify each method using:
   --list-all to discover available methods,
   --method to verify specific methods,
   --exceptions-only to check error handling.
4. Only present verified code after confirmation.
5. Additionally, you can analyze a script to extract all third-party packages used.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from loguru import logger
import ast
import importlib.util
import click

# Add the agent_tools package to Python path
agent_tools_path = "/Users/robert/Documents/dev/workspace/agent_tools/src"
if agent_tools_path not in sys.path:
    sys.path.insert(0, agent_tools_path)

from agent_tools.method_validator.analyzer import (
    MethodAnalyzer,
    timing,
    validate_method,
)

# Standard library packages that should be skipped
STANDARD_PACKAGES = {
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "binascii",
    "builtins",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "functools",
    "glob",
    "gzip",
    "hashlib",
    "hmac",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "pprint",
    "queue",
    "re",
    "random",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "ssl",
    "statistics",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "timeit",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
    "zlib",
}

# Common utility packages that typically don't need analysis
COMMON_UTILITY_PACKAGES = {
    "requests",
    "urllib3",
    "six",
    "setuptools",
    "pip",
    "wheel",
    "pkg_resources",
    "pytest",
    "nose",
    "mock",
    "coverage",
    "tox",
    "flake8",
    "pylint",
    "mypy",
    "black",
    "isort",
    "yapf",
    "autopep8",
}

# Standard/common utility packages that should be skipped
SKIP_PACKAGES = {
    "pip",
    "setuptools",
    "pkg_resources",
    "wheel",
    "distutils",
    "venv",
    "virtualenv",
}

def should_analyze_package(package_name: str, allow_third_party: bool = False) -> bool:
    """
    Determine if a package should be analyzed based on its name.

    Args:
        package_name: Name of the package to analyze
        allow_third_party: If True, allows analysis of third-party packages
    """
    logger.debug(f"Checking if package should be analyzed: {package_name}")

    # Always skip standard library
    if package_name in STANDARD_PACKAGES or any(
        package_name.startswith(f"{pkg}.") for pkg in STANDARD_PACKAGES
    ):
        logger.debug(f"Skipping standard library package: {package_name}")
        return False

    # Skip common utilities unless explicitly allowed
    if not allow_third_party and package_name in COMMON_UTILITY_PACKAGES:
        logger.debug(f"Skipping common utility package: {package_name}")
        return False

    # Skip specific packages
    if package_name in SKIP_PACKAGES:
        logger.debug(f"Skipping package: {package_name}")
        return False

    # Skip packages that start with underscore
    if package_name.startswith("_"):
        logger.debug(f"Skipping package: {package_name}")
        return False

    logger.debug(f"Package {package_name} will be analyzed")
    return True

def should_auto_execute(command: str) -> bool:
    """
    Determine if a command should be executed automatically based on user preferences.
    Currently supports compilation and git commands.
    """
    # List of commands that are safe to auto-execute
    auto_executable_patterns = [
        # Git commands
        r"^git\s+(status|log|diff|branch|checkout|pull|push|fetch|merge|rebase)",
        # Compilation commands
        r"^(gcc|g\+\+|make|cmake|mvn|gradle|pip|npm|yarn|cargo)",
        # Python package analysis (for AI agent use)
        r"^method-validator\s+\w+\s+(--method|--list-all|--exceptions-only|--debug)",
    ]

    return any(re.match(pattern, command) for pattern in auto_executable_patterns)

def find_venv_path() -> Optional[str]:
    """Find the virtual environment path for the current project.

    Looks for virtual environment in the following order:
    1. Check if currently running in a virtual environment (sys.prefix)
    2. Look for .venv in the project root directory (uv's default location)
    3. Look for venv or .env as fallbacks
    """
    # Check if running in a virtual environment
    if sys.prefix != sys.base_prefix:
        return sys.prefix

    # Get the project root directory (where the .git directory is typically located)
    current_dir = Path.cwd()
    while current_dir.parent != current_dir:
        # Look for common project root indicators
        if (
            (current_dir / ".git").exists()
            or (current_dir / "pyproject.toml").exists()
            or (current_dir / "setup.py").exists()
        ):
            # For uv, check .venv first as it's the default
            venv_dir = current_dir / ".venv"
            if venv_dir.exists():
                # uv's structure is different - it has 'Scripts' on Windows and 'bin' on Unix
                bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
                if bin_dir.exists():
                    return str(venv_dir)

            # Fallback to other common venv names
            for venv_name in ["venv", ".env"]:
                venv_dir = current_dir / venv_name
                if venv_dir.exists() and (venv_dir / "bin").exists():
                    return str(venv_dir)
            break  # If we found project root but no venv, stop searching
        current_dir = current_dir.parent

    return None

def extract_third_party_packages(script_path: str) -> List[Tuple[str, str]]:
    """
    Given a path to a Python script, parse the file to extract all imported package names
    and determine their file system locations (if installed). It filters out standard library
    and common utility packages.

    Returns:
        A list of tuples (package_name, package_location)
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read script {script_path}: {e}")
        return []

    try:
        tree = ast.parse(script_content, filename=script_path)
    except Exception as e:
        logger.error(f"Error parsing script {script_path}: {e}")
        return []

    imported_packages: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split('.')[0]
                imported_packages.add(pkg)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                pkg = node.module.split('.')[0]
                imported_packages.add(pkg)

    # Filter out packages that are standard or common utility packages.
    third_party = [
        pkg for pkg in imported_packages
        if pkg not in STANDARD_PACKAGES and pkg not in COMMON_UTILITY_PACKAGES
    ]

    results = []
    for pkg in third_party:
        spec = importlib.util.find_spec(pkg)
        if spec and spec.origin:
            results.append((pkg, spec.origin))
        else:
            results.append((pkg, "Not Found"))

    return results

def setup_cli_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Method Validator - Analyze and validate Python package methods"
    )
    parser.add_argument(
        "package",
        nargs="?",
        help="Package name to analyze (e.g. 'requests', 'pandas')."
    )
    parser.add_argument(
        "--method",
        help="Specific method to analyze (e.g. 'DataFrame.apply')",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all available methods",
    )
    parser.add_argument(
        "--exceptions-only",
        action="store_true",
        help="Show only exception information",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help="Group methods by category",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show execution timing statistics",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--venv-path",
        help="Virtual environment path (auto-detected if not provided)",
        default=find_venv_path(),
    )
    # New flag to analyze a Python script for third-party packages.
    parser.add_argument(
        "--script",
        help="Path to a Python script from which to extract third-party package names and their locations."
    )
    return parser

def configure_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from debug logs
    logging.getLogger("agent_tools.method_validator.cache").setLevel(logging.WARNING)

def handle_exceptions_only(analyzer: MethodAnalyzer, package: str, json_output: bool) -> None:
    """Handle --exceptions-only command."""
    with timing.measure("exception_summary", "Generating exception summary"):
        # Get all methods and extract exception info
        methods = analyzer.quick_scan(package)
        exceptions: Dict[str, List[str]] = {}
        
        for name, _, _ in methods:
            method_info = analyzer.deep_analyze(package, name)
            if method_info and method_info.get("exceptions"):
                for exc in method_info["exceptions"]:
                    exc_name = exc["type"]
                    if exc_name not in exceptions:
                        exceptions[exc_name] = []
                    exceptions[exc_name].append(name)

        if json_output:
            result = {
                "exceptions": [
                    {
                        "name": name,
                        "raised_by": methods,
                    }
                    for name, methods in exceptions.items()
                ]
            }
            print(json.dumps(result))
            return

        if not exceptions:
            logger.warning("No exceptions found in package documentation")
            return

        for exc_name, raised_by in exceptions.items():
            logger.info(f"\n{exc_name}:")
            if raised_by:
                logger.info(f"  Raised by {len(raised_by)} methods:")
                for method in raised_by[:5]:
                    logger.info(f"    - {method}")
                if len(raised_by) > 5:
                    logger.info(f"    ... and {len(raised_by) - 5} more")

def handle_list_all(analyzer: MethodAnalyzer, package: str, by_category: bool, json_output: bool) -> None:
    """Handle --list-all command."""
    with timing.measure("quick_scan", "Performing quick scan"):
        methods = analyzer.quick_scan(package)
        if json_output:
            result: Dict[str, Any] = {}
            if by_category:
                by_category_dict: Dict[str, List[Dict[str, Any]]] = {}
                for name, summary, categories in methods:
                    for category in categories:
                        by_category_dict.setdefault(category, []).append(
                            {"name": name, "summary": summary}
                        )
                result["methods_by_category"] = by_category_dict
            else:
                result["methods"] = [
                    {"name": name, "summary": summary, "categories": cats} 
                    for name, summary, cats in methods
                ]
            print(json.dumps(result))
            return

        if by_category:
            category_methods: Dict[str, List[Tuple[str, str]]] = {}
            for name, summary, categories in methods:
                for category in categories:
                    category_methods.setdefault(category, []).append((name, summary))

            for category, method_list in sorted(category_methods.items()):
                logger.info(f"\n📦 {category.title()}:")
                for name, summary in sorted(method_list):
                    logger.info(f"  ✓ {name}:")
                    if summary:
                        logger.info(f"    {summary}")
        else:
            for name, summary, _ in methods:
                logger.info(f"\n{name}:")
                if summary:
                    logger.info(f"  Summary: {summary}")

def handle_method_analysis(analyzer: MethodAnalyzer, package: str, method: str, json_output: bool) -> None:
    """Handle --method command."""
    with timing.measure("deep_analysis", "Performing deep analysis"):
        method_info = analyzer.deep_analyze(package, method)
        if method_info:
            if json_output:
                print(json.dumps({"method_info": method_info}))
                return

            logger.info(f"\nDetailed analysis of '{method}':")
            logger.info(f"  Description: {method_info['summary']}")
            logger.info(f"  Signature: {method_info['signature']}")
            if method_info.get("parameters"):
                logger.info("\n  Parameters:")
                for param in method_info["parameters"]:
                    logger.info(f"    {param}")
            if method_info.get("returns"):
                logger.info(f"\n  Returns: {method_info['return_info']['type']}")
            if method_info.get("exceptions"):
                logger.info("\n  May raise:")
                for exc in method_info["exceptions"]:
                    logger.info(f"    - {exc['type']}")
            if method_info.get("examples"):
                logger.info("\n  Examples:")
                for example in method_info["examples"]:
                    logger.info(f"\n{example}")
        else:
            # Get similar methods
            similar = []
            for name, summary, _ in analyzer.quick_scan(package):
                if method.lower() in name.lower():
                    similar.append(name)
                    
            if json_output:
                result: Dict[str, Any] = {"error": f"Method '{method}' not found"}
                if similar:
                    result["similar_methods"] = similar
                print(json.dumps(result))
            else:
                logger.error(f"Method '{method}' not found")
                if similar:
                    logger.info("\nSimilar methods:")
                    for method_name in similar:
                        logger.info(f"  - {method_name}")

@click.command()
@click.argument("package_name")
@click.option("--method", help="Method name to analyze")
@click.option("--list-all", is_flag=True, help="List all methods in package")
@click.option("--quick", is_flag=True, help="Quick validation without deep analysis")
def main(package_name: str, method: Optional[str], list_all: bool, quick: bool):
    """Validate methods in a Python package."""
    logger.debug(f"Analyzing package: {package_name}")
    
    if not should_analyze_package(package_name):
        logger.info(f"Skipping package {package_name}")
        return
        
    if quick and method:
        # Quick validation without deep analysis
        is_valid, message = validate_method(package_name, method)
        if is_valid:
            logger.info(f"✓ {message}")
            sys.exit(0)
        else:
            logger.error(f"✗ {message}")
            sys.exit(1)
    
    analyzer = MethodAnalyzer()
    
    if list_all:
        # List all methods
        methods = analyzer.quick_scan(package_name)
        for name, summary, categories in methods:
            logger.info(f"{name}: {summary}")
            if categories:
                logger.info(f"  Categories: {', '.join(categories)}")
        return
        
    if method:
        # Analyze specific method
        result = analyzer.deep_analyze(package_name, method)
        if result:
            logger.info(f"Method: {method}")
            logger.info(f"Summary: {result.get('summary', '')}")
            logger.info(f"Signature: {result.get('signature', '')}")
            if result.get('parameters'):
                logger.info("Parameters:")
                for name, info in result['parameters'].items():
                    required = "required" if info.get('required') else "optional"
                    logger.info(f"  {name} ({required}): {info.get('description', '')}")
            if result.get('exceptions'):
                logger.info("Exceptions:")
                for exc in result['exceptions']:
                    logger.info(f"  {exc['type']}: {exc.get('description', '')}")
        else:
            logger.error(f"Method {method} not found in {package_name}")
            sys.exit(1)
    else:
        logger.error("Please specify a method to analyze or use --list-all")
        sys.exit(1)

if __name__ == "__main__":
    main()
