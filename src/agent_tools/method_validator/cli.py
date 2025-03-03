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
from typing import Dict, List, Optional, Set, Any, Union, cast, Tuple, Callable, TypeVar, Generic
import click
from loguru import logger
import typer
import hashlib
import functools

from .analyzer import MethodAnalyzer, validate_method
from .utils import timing
from .function_improver import FunctionValidator, _analysis_cache, ValidationResult

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

# Track validated methods in the current session
_validated_methods: Set[str] = set()
_validation_required = True

T = TypeVar('T')

class ValidationContext:
    """Context manager to ensure validation happens automatically."""
    def __init__(self) -> None:
        self.validated_methods: Set[str] = set()
        self.test_results: Dict[str, bool] = {}
        
    def __enter__(self) -> 'ValidationContext':
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
        
    def validate_method(self, package_name: str, method_name: str) -> bool:
        """Validate a single method."""
        key = f"{package_name}.{method_name}"
        if key in self.validated_methods:
            return True
            
        is_valid, message = validate_method(package_name, method_name)
        if is_valid:
            self.validated_methods.add(key)
            logger.info(f"✓ Validated {key}")
        else:
            logger.error(f"✗ Failed to validate {key}: {message}")
        return is_valid
        
    def run_tests(self) -> bool:
        """Run pytest suite."""
        result = run_pytest_automatically()
        if result:
            logger.info("✓ All tests passed")
        else:
            logger.error("✗ Tests failed")
        return result

class ValidationResult(Generic[T]):
    """Result type that ensures validation before access."""
    def __init__(self, value: T, context: ValidationContext) -> None:
        self._value = value
        self._context = context
        self._validated = False
        
    def get(self) -> T:
        """Get the value, ensuring validation has occurred."""
        if not self._validated:
            # Extract and validate all methods
            tree = ast.parse(str(self._value))
            methods_to_validate = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        methods_to_validate.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        methods_to_validate.add(node.func.attr)
            
            # Validate all methods
            all_valid = True
            for method in methods_to_validate:
                if method not in self._context.validated_methods and should_analyze_package(method):
                    if not self._context.validate_method(method, method):
                        all_valid = False
                        
            # Run tests
            if all_valid and not self._context.run_tests():
                all_valid = False
                
            if not all_valid:
                raise ValueError("Validation failed - cannot show code to user")
                
            self._validated = True
            
        return self._value

def require_validation(func: Callable) -> Callable:
    """Decorator to ensure all methods are validated before code is shown."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _validation_required:
            return func(*args, **kwargs)
            
        # Extract method names from code
        code = kwargs.get('code', '')
        if not code:
            return func(*args, **kwargs)
            
        tree = ast.parse(code)
        methods_to_validate = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    methods_to_validate.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    methods_to_validate.add(node.func.attr)
        
        # Validate all methods
        for method in methods_to_validate:
            if method not in _validated_methods and should_analyze_package(method):
                # Run quick validation
                is_valid, message = validate_method(method, method)
                if not is_valid:
                    raise ValueError(f"Method {method} failed validation: {message}")
                _validated_methods.add(method)
        
        return func(*args, **kwargs)
    return wrapper

def run_pytest_automatically() -> bool:
    """Run pytest automatically and return True if all tests pass."""
    try:
        import pytest
        result = pytest.main(['--quiet'])
        return result == 0
    except Exception as e:
        logger.error(f"Error running pytest: {e}")
        return False

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


app = typer.Typer()

@app.command()
def validate(
    package_name: Optional[str] = typer.Argument(None),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Method to analyze"),
    quick: bool = typer.Option(True, "--quick/--no-quick", help="Quick validation without deep analysis"),
    script: Optional[str] = typer.Option(None, "--script", "-s", help="Extract third party packages from script"),
    function_file: Optional[str] = typer.Option(None, "--function-file", "-f", help="Python file containing function to validate"),
) -> None:
    """Validate Python package methods or functions.
    
    By default, runs quick validation on methods and automatic validation on functions.
    Use --no-quick for deep analysis of methods when needed.
    """
    with ValidationContext() as ctx:
        try:
            # Handle function validation first - this is the primary use case
            if function_file:
                try:
                    with open(function_file, 'r') as f:
                        function_text = f.read()
                    
                    # Run automatic validation
                    is_valid, result = validate_code_automatically(function_text)
                    
                    # Display results
                    typer.echo(f"\n🔍 Validating {function_file}")
                    typer.echo("=" * 50)
                    
                    has_issues = False
                    
                    if result['type_issues']:
                        has_issues = True
                        typer.echo("\n❌ Type Issues:")
                        for issue in result['type_issues']:
                            typer.echo(f"  • {issue}")
                            
                    if result['method_issues']:
                        has_issues = True
                        typer.echo("\n❌ Method Issues:")
                        for issue in result['method_issues']:
                            typer.echo(f"  • {issue}")
                            
                    if result['quality_issues']:
                        has_issues = True
                        typer.echo("\n⚠️ Quality Issues:")
                        for issue in result['quality_issues']:
                            typer.echo(f"  • {issue}")
                            
                    if result['style_issues']:
                        has_issues = True
                        typer.echo("\n📝 Style Issues:")
                        for issue in result['style_issues']:
                            typer.echo(f"  • {issue}")
                    
                    if result['suggestions']:
                        typer.echo("\n💡 Suggestions:")
                        for suggestion in result['suggestions']:
                            typer.echo(f"  • {suggestion}")
                            
                    if not has_issues:
                        typer.echo("\n✅ No issues found!")

                    sys.exit(1 if has_issues else 0)
                    
                except Exception as e:
                    typer.echo(f"Error validating function: {e}")
                    sys.exit(1)

            # Handle script analysis
            if script:
                if not os.path.isfile(script):
                    typer.echo(f"Script not found: {script}")
                    sys.exit(1)

                packages = extract_third_party_packages(script)
                if not packages:
                    typer.echo("No third party packages found in script")
                    sys.exit(0)

                typer.echo("Found packages:")
                for pkg in sorted(packages):
                    typer.echo(f"- {pkg}")
                sys.exit(0)

            # Package validation requires package name
            if not package_name:
                typer.echo("Please specify a package name or use --function-file to validate a function")
                sys.exit(1)

            # Find virtual environment
            venv_path = find_venv_path()
            if not venv_path:
                typer.echo("Virtual environment not found")
                sys.exit(1)

            # Analyze package/method
            analyzer = MethodAnalyzer(venv_path)
            try:
                if not method:
                    # Quick scan of package methods
                    methods = analyzer.quick_scan(package_name)
                    typer.echo(json.dumps(methods, indent=2))
                    sys.exit(0)

                # Validate specific method
                if quick:
                    # Quick validation for method
                    is_valid, message = validate_method(package_name, method)
                    typer.echo(message)
                    sys.exit(0 if is_valid else 1)
                else:
                    # Deep analysis for method
                    results = analyzer.deep_analyze(package_name, method)
                    if results is None:
                        typer.echo(f"Method {method} not found in package {package_name}")
                        sys.exit(1)
                    typer.echo(json.dumps(results, indent=2))
                    sys.exit(0)

            except Exception as e:
                typer.echo(f"Error analyzing method: {e}")
                sys.exit(1)

            # If validation succeeds, run pytest automatically
            if not ctx.validate_method(package_name, method):
                sys.exit(1)
            
        except Exception as e:
            typer.echo(f"Error during validation: {e}")
            sys.exit(1)

def validate_code_automatically(function_text: str) -> Tuple[bool, ValidationResult]:
    """
    Automatically validate code before showing it to the user.
    Returns (is_valid, result) tuple.
    """
    # Try to get cached result first
    cache_key = hashlib.md5(function_text.encode()).hexdigest()
    if cache_key in _analysis_cache:
        result = _analysis_cache[cache_key][0]  # Get result from tuple
        return not any([
            result['type_issues'],
            result['method_issues'],
            result['quality_issues']
        ]), result

    # Run quick validation first
    validator = FunctionValidator(deep_analysis=False)
    quick_result = validator.validate_function(function_text=function_text)
    
    # If quick validation finds issues, run deep analysis
    if quick_result['type_issues'] or quick_result['method_issues']:
        validator = FunctionValidator(deep_analysis=True)
        result = validator.validate_function(function_text=function_text)
    else:
        result = quick_result
    
    is_valid = not any([
        result['type_issues'],
        result['method_issues'],
        result['quality_issues']
    ])
    
    return is_valid, result

@app.command()
def check_function(
    file: str = typer.Option(..., help="Python file to validate"),
    manual: bool = typer.Option(False, help="Run in manual mode without automatic validation")
) -> None:
    """Validate a Python function file automatically."""
    try:
        with open(file, 'r') as f:
            function_text = f.read()
    except Exception as e:
        typer.echo(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    # Always run automatic validation first
    is_valid, result = validate_code_automatically(function_text)
    
    # Display results
    typer.echo(f"\n🔍 Validating {file}")
    typer.echo("=" * 50)
    
    has_issues = False
    
    if result['type_issues']:
        has_issues = True
        typer.echo("\n❌ Type Issues:")
        for issue in result['type_issues']:
            typer.echo(f"  • {issue}")
            
    if result['method_issues']:
        has_issues = True
        typer.echo("\n❌ Method Issues:")
        for issue in result['method_issues']:
            typer.echo(f"  • {issue}")
            
    if result['quality_issues']:
        has_issues = True
        typer.echo("\n⚠️ Quality Issues:")
        for issue in result['quality_issues']:
            typer.echo(f"  • {issue}")
            
    if result['style_issues']:
        has_issues = True
        typer.echo("\n📝 Style Issues:")
        for issue in result['style_issues']:
            typer.echo(f"  • {issue}")
    
    if result['suggestions']:
        typer.echo("\n💡 Suggestions:")
        for suggestion in result['suggestions']:
            typer.echo(f"  • {suggestion}")
            
    if not has_issues:
        typer.echo("\n✅ No issues found!")

    # Exit with appropriate code
    if has_issues and not manual:
        raise typer.Exit(code=1)

def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
