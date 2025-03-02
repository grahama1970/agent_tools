#!/usr/bin/env python3
"""Method Validator CLI.

This CLI analyzes and validates Python package methods. It is primarily used by the AI agent
to validate methods after writing code.

IMPORTANT:
- The AI agent MUST validate methods using `method-validator package --method method_name --quick`
  autonomously (without user approval) before showing any code to the user.
- If a method exists but requires parameter details, use `method-validator package --method method_name`
  to get the full signature, required parameters and return types.
- Only return complete working code after ALL methods have been validated successfully.
- Use deep analysis (without --quick flag) to understand parameters.
- Do not check standard library methods or common utilities.
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, cast
import click
from loguru import logger

from .analyzer import MethodAnalyzer, validate_method
from .utils import timing

# Standard library packages that don't need analysis
STDLIB_PACKAGES = {
    "os",
    "sys",
    "json",
    "time",
    "datetime",
    "math",
    "random",
    "collections",
    "itertools",
    "functools",
    "typing",
    "pathlib",
    "shutil",
    "tempfile",
    "io",
    "re",
}

# Common utility packages that don't need analysis
COMMON_UTIL_PACKAGES = {
    "numpy",
    "pandas",
    "requests",
    "pytest",
    "unittest",
    "logging",
    "loguru",
    "click",
    "typer",
    "rich",
}


def find_venv_path() -> Optional[str]:
    """Find the virtual environment path."""
    if "VIRTUAL_ENV" in os.environ:
        return os.environ["VIRTUAL_ENV"]

    # Check common venv locations
    cwd = os.getcwd()
    common_venv_dirs = ["venv", ".venv", "env", ".env"]
    for venv_dir in common_venv_dirs:
        venv_path = os.path.join(cwd, venv_dir)
        if os.path.isdir(venv_path):
            return venv_path

    return None


def should_analyze_package(package_name: str) -> bool:
    """Determine if a package should be analyzed."""
    return (
        package_name not in STDLIB_PACKAGES
        and package_name not in COMMON_UTIL_PACKAGES
        and not package_name.startswith("_")
    )


def extract_third_party_packages(script_path: str) -> Set[str]:
    """Extract third party package imports from a Python script."""
    packages = set()

    with open(script_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                base_package = name.name.split(".")[0]
                if should_analyze_package(base_package):
                    packages.add(base_package)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_package = node.module.split(".")[0]
                if should_analyze_package(base_package):
                    packages.add(base_package)

    return packages


@click.command()
@click.argument("package_name", required=False)
@click.option("--method", help="Method to analyze")
@click.option("--list-all", is_flag=True, help="List all methods in package")
@click.option("--quick", is_flag=True, help="Quick validation without deep analysis")
@click.option("--show-timing", is_flag=True, help="Show timing information")
@click.option("--script", help="Extract third party packages from script")
def main(
    package_name: Optional[str],
    method: Optional[str],
    list_all: bool,
    quick: bool,
    show_timing: bool,
    script: Optional[str],
) -> None:
    """Analyze and validate Python package methods."""

    timing.enabled = show_timing

    if script:
        if not os.path.isfile(script):
            click.echo(f"Script not found: {script}")
            sys.exit(1)

        packages = extract_third_party_packages(script)
        if not packages:
            click.echo("No third party packages found in script")
            sys.exit(0)

        click.echo("Found packages:")
        for pkg in sorted(packages):
            click.echo(f"- {pkg}")
        sys.exit(0)

    # Package name is required for method validation
    if not package_name:
        click.echo("Please specify a package name when not using --script")
        sys.exit(1)

    venv_path = find_venv_path()
    if not venv_path:
        click.echo("Virtual environment not found")
        sys.exit(1)

    analyzer = MethodAnalyzer(venv_path)

    try:
        if list_all:
            methods = analyzer.quick_scan(package_name)
            click.echo(json.dumps(methods, indent=2))
            sys.exit(0)

        if not method:
            click.echo("Please specify a method to analyze with --method")
            sys.exit(1)

        if quick:
            is_valid, message = validate_method(package_name, method)
            click.echo(message)
            sys.exit(0 if is_valid else 1)
        else:
            results = analyzer.deep_analyze(package_name, method)
            if results is None:
                click.echo(f"Method {method} not found in package {package_name}")
                sys.exit(1)

            click.echo(json.dumps(results, indent=2))

        if show_timing:
            click.echo("\nTiming information:")
            click.echo(json.dumps(timing.get_summary(), indent=2))

    except Exception as e:
        click.echo(f"Error analyzing method: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
