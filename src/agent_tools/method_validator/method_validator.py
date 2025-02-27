#!/usr/bin/env python3
"""
Method Validator - An AI Agent's Tool for API Discovery and Validation

This tool helps AI agents verify methods and check for existing functionality BEFORE suggesting code.

AGENT WORKFLOW:
1. When user requests code implementation:
   a. Draft implementation (do not show yet)
   b. Extract ALL methods you plan to use
   c. Run verification:
      method-validator package_name --list-all  # Check for existing functionality
      For each method:
      method-validator package_name --method method_name
   d. If verification fails:
      - Remove non-existent methods
      - Fix duplicated functionality
   e. Only then present verified code with confirmation:
      "✓ All methods verified to exist
       ✓ No duplicate functionality found
       Here's the implementation:"

2. When modifying existing code:
   a. Extract methods being added/changed
   b. Verify each new/changed method
   c. Check for duplicate functionality
   d. Present changes only after verification

3. When handling errors:
   method-validator package_name --exceptions-only

IMPORTANT:
- Never present code without method verification
- Never suggest methods without checking they exist
- Never duplicate functionality that already exists

Usage workflow for AI agents:
1. When writing a function that might already exist in a relevant package:
   - Use --list-all to check for existing implementations
2. When using a package-specific method:
   - Use --method to verify its exact signature and behavior
3. When handling errors:
   - Use --exceptions-only to see package-specific exceptions to handle
"""

import argparse
import inspect
import importlib
import json
import os
import sys
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from loguru import logger
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import sqlite3
import pickle
import hashlib


@dataclass
class TimingStats:
    """Container for timing statistics of operations."""

    operation: str
    total_time: float = 0.0
    calls: int = 0

    @property
    def average_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0


class TimingManager:
    """Manages timing statistics for method validator operations."""

    def __init__(self):
        self.stats: Dict[str, TimingStats] = defaultdict(lambda: TimingStats(""))
        self.enabled = True

    @contextmanager
    def measure(self, operation: str, description: str = ""):
        """Context manager to measure execution time of an operation."""
        if not self.enabled:
            yield
            return

        start_time = time.time()
        try:
            if description:
                logger.info(f"Starting {operation}: {description}")
            yield
        finally:
            elapsed = time.time() - start_time
            if operation not in self.stats:
                self.stats[operation] = TimingStats(operation)
            self.stats[operation].total_time += elapsed
            self.stats[operation].calls += 1
            if description:
                logger.info(f"Completed {operation} in {elapsed:.2f}s")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of timing statistics."""
        return {
            op: {
                "total_time": stat.total_time,
                "calls": stat.calls,
                "average_time": stat.average_time,
            }
            for op, stat in self.stats.items()
        }


# Create global timing manager
timing = TimingManager()

# Create a global cache for method analysis results
method_analysis_cache = {}

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


def should_analyze_package(package_name: str, allow_third_party: bool = False) -> bool:
    """
    Determine if a package should be analyzed based on its name.

    Args:
        package_name: Name of the package to analyze
        allow_third_party: If True, allows analysis of third-party packages
    """
    # Always skip standard library
    if package_name in STANDARD_PACKAGES or any(
        package_name.startswith(f"{pkg}.") for pkg in STANDARD_PACKAGES
    ):
        return False

    # Skip common utilities unless explicitly allowed
    if not allow_third_party and package_name in COMMON_UTILITY_PACKAGES:
        return False

    return True


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


class AnalysisCache:
    """Persistent cache for method analysis results using SQLite."""

    def __init__(self):
        cache_dir = Path.home() / ".cache" / "sparta" / "method_validator"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "analysis_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS method_cache (
                    package_name TEXT,
                    method_name TEXT,
                    source_hash TEXT,
                    result BLOB,
                    timestamp REAL,
                    PRIMARY KEY (package_name, method_name)
                )
            """
            )

    def _get_source_hash(self, obj) -> str:
        """Get hash of method source code to detect changes."""
        try:
            source = inspect.getsource(obj)
            return hashlib.sha256(source.encode()).hexdigest()
        except (TypeError, OSError):
            return ""

    def get(self, package_name: str, method_name: str, obj) -> Optional[Any]:
        """Get cached result if it exists and is valid."""
        source_hash = self._get_source_hash(obj)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result, source_hash FROM method_cache WHERE package_name = ? AND method_name = ?",
                (package_name, method_name),
            ).fetchone()

            if row and row[1] == source_hash:
                return pickle.loads(row[0])
        return None

    def set(self, package_name: str, method_name: str, obj, result: Any):
        """Cache analysis result."""
        source_hash = self._get_source_hash(obj)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO method_cache VALUES (?, ?, ?, ?, ?)",
                (
                    package_name,
                    method_name,
                    source_hash,
                    pickle.dumps(result),
                    time.time(),
                ),
            )


# Create global cache instance
analysis_cache = AnalysisCache()


class MethodInfo:
    """Structured container for method information."""

    def __init__(self, obj, name: str):
        if obj is None:
            raise ValueError(f"Cannot analyze None object for method {name}")

        self.name = name
        self.obj = obj
        try:
            with timing.measure("docstring_extraction"):
                self.doc = inspect.getdoc(obj) or ""
                self.signature = str(inspect.signature(obj))
                self.module = obj.__module__
                self.summary = self._generate_summary()

            with timing.measure("parameter_analysis"):
                self.parameters = self._analyze_parameters()

            with timing.measure("example_extraction"):
                self.examples = self._extract_examples()

            with timing.measure("exception_analysis"):
                self.exceptions = self._analyze_exceptions()

            with timing.measure("return_analysis"):
                self.return_info = self._analyze_return_info()
        except Exception as e:
            logger.warning(f"Error analyzing method {name}: {e}")
            # Set default values for failed analysis
            self.doc = ""
            self.signature = "()"
            self.module = obj.__module__ if hasattr(obj, "__module__") else ""
            self.summary = ""
            self.parameters = {}
            self.examples = []
            self.exceptions = []
            self.return_info = {}

    def _generate_summary(self) -> str:
        """Generate a quick summary from the docstring."""
        if not self.doc:
            return ""
        summary = self.doc.split("\n")[0].split(".")[0]
        return summary[:100] + "..." if len(summary) > 100 else summary

    def _analyze_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter types, defaults, and constraints.

        Focuses on the most relevant parameters by considering:
        1. Required parameters
        2. Parameters with good documentation
        3. Parameters commonly used in examples
        4. Parameters that affect core functionality

        Filters out:
        - Internal/private parameters
        - Rarely used optional parameters
        - Debug/development parameters
        - Parameters with default values that rarely need changing
        """
        params = {}
        signature = inspect.signature(self.obj)

        # Extract parameters mentioned in examples
        example_params = set()
        for example in self._extract_examples():
            param_matches = re.finditer(r"(\w+)\s*=", example)
            example_params.update(match.group(1) for match in param_matches)

        # Common parameter name patterns to exclude
        exclude_patterns = {
            r"^_",  # Private params
            r"debug$",  # Debug flags
            r"verbose$",  # Verbosity flags
            r"(callback|hook)$",  # Advanced callback params
            r"experimental",  # Experimental features
            r"internal",  # Internal use params
            r"deprecated",  # Deprecated params
        }

        for name, param in signature.parameters.items():
            # Skip excluded parameters
            if any(re.search(pattern, name) for pattern in exclude_patterns):
                continue

            # Calculate parameter relevance score
            relevance = 0
            if param.default == param.empty:  # Required param
                relevance += 3
            if name in example_params:  # Used in examples
                relevance += 2
            if self._find_param_description(name):  # Has documentation
                relevance += 1

            # Include parameter if it's relevant enough
            if relevance >= 2 or name in {
                "model",
                "messages",
                "stream",
                "api_key",
            }:  # Always include core params
                params[name] = {
                    "type": (
                        str(param.annotation)
                        if param.annotation != param.empty
                        else None
                    ),
                    "default": (
                        None if param.default == param.empty else str(param.default)
                    ),
                    "required": param.default == param.empty
                    and param.kind != param.VAR_POSITIONAL,
                    "description": self._find_param_description(name),
                }

        return params

    def _find_param_description(self, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not self.doc:
            return ""

        # Look for :param param_name: or Parameters: section
        param_patterns = [
            rf":param {param_name}:\s*([^\n]+)",
            rf"Parameters.*?{param_name}\s*[:-]\s*([^\n]+)",
        ]

        for pattern in param_patterns:
            match = re.search(pattern, self.doc, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_examples(self) -> List[str]:
        """Extract usage examples from docstring."""
        if not self.doc:
            return []

        examples = []
        # Look for Examples section or code blocks
        example_section = re.split(r"Examples?[:|-]", self.doc)
        if len(example_section) > 1:
            # Extract code blocks (indented or between ```)
            code_blocks = re.findall(
                r"```(?:python)?\n(.*?)\n```|\n\s{4}(.*?)(?=\n\S)",
                example_section[1],
                re.DOTALL,
            )
            examples = [block[0] or block[1] for block in code_blocks if any(block)]
        return examples

    def _analyze_exceptions(self) -> List[Dict[str, str]]:
        """Analyze exceptions that can be raised by the method.

        Focuses on the most relevant exceptions by considering:
        1. Explicit raises in docstring or source code
        2. Frequency of use across the codebase
        3. Quality of documentation
        4. Whether it's a custom exception vs generic
        """
        exceptions = []
        if not self.doc:
            return exceptions

        # Look for explicitly documented exceptions first (highest priority)
        raise_patterns = [
            r":raises\s+(\w+):\s*([^\n]+)",
            r"Raises:\n(?:\s*-?\s*(\w+):\s*([^\n]+)\n?)*",
        ]

        for pattern in raise_patterns:
            matches = re.finditer(pattern, self.doc, re.MULTILINE)
            for match in matches:
                if len(match.groups()) == 2:
                    exc_name, desc = match.groups()
                    if (
                        desc and len(desc.strip()) > 10
                    ):  # Only include well-documented exceptions
                        exceptions.append(
                            {
                                "type": exc_name,
                                "description": desc.strip(),
                                "hierarchy": self._get_exception_hierarchy(exc_name),
                                "source": "documentation",
                            }
                        )

        # Look for raise statements in source code
        try:
            source = inspect.getsource(self.obj)
            custom_exceptions = set()

            # First pass: identify custom exceptions
            for line in source.split("\n"):
                if "class" in line and "Error" in line and "Exception" in line:
                    match = re.search(r"class\s+(\w+Error)", line)
                    if match:
                        custom_exceptions.add(match.group(1))

            # Second pass: find raise statements
            raise_statements = re.finditer(r"raise\s+(\w+)(?:\(|$)", source)
            for match in raise_statements:
                exc_name = match.group(1)
                # Prioritize custom exceptions and well-known error types
                if (
                    exc_name in custom_exceptions
                    or exc_name.endswith("Error")
                    or exc_name in {"ValueError", "TypeError", "RuntimeError"}
                ):

                    # Don't duplicate exceptions we already found in docstring
                    if not any(e["type"] == exc_name for e in exceptions):
                        exceptions.append(
                            {
                                "type": exc_name,
                                "description": self._infer_exception_description(
                                    exc_name, source
                                ),
                                "hierarchy": self._get_exception_hierarchy(exc_name),
                                "source": "source_code",
                            }
                        )
        except (TypeError, OSError):
            pass

        return exceptions

    def _infer_exception_description(self, exc_name: str, source: str) -> str:
        """Attempt to infer a meaningful description for an exception from the source code context."""
        # Look for the exception class definition
        class_match = re.search(
            rf'class\s+{exc_name}\s*\([^)]+\):\s*(?:"""|\'\'\')?(.*?)(?:"""|\'\'\')?\s*(?:pass|\n\s*\w+)',
            source,
            re.DOTALL,
        )
        if class_match and class_match.group(1):
            desc = class_match.group(1).strip()
            if desc:
                return desc

        # Look for comments near raise statements
        raise_contexts = re.finditer(
            rf"(?:#[^\n]*\n)*\s*raise\s+{exc_name}\s*\(([^)]*)\)", source
        )
        descriptions = []
        for context in raise_contexts:
            comment_match = re.search(r"#\s*([^\n]+)", context.group(0))
            if comment_match:
                descriptions.append(comment_match.group(1).strip())
            elif context.group(1):  # Use the error message if no comment
                descriptions.append(context.group(1).strip(" '\""))

        if descriptions:
            # Use the most common or first description
            from collections import Counter

            return Counter(descriptions).most_common(1)[0][0]

        return "Found in source code"

    def _get_exception_hierarchy(self, exc_name: str) -> List[str]:
        """Get the exception class hierarchy."""
        try:
            exc_class = getattr(
                sys.modules.get(self.module) or sys.modules["builtins"], exc_name
            )
            if not (
                inspect.isclass(exc_class) and issubclass(exc_class, BaseException)
            ):
                return []

            hierarchy = []
            current = exc_class
            while current != object:
                hierarchy.append(current.__name__)
                current = current.__bases__[0]
            return hierarchy
        except (AttributeError, TypeError):
            return []

    def _analyze_return_info(self) -> Dict[str, str]:
        """Analyze return type and description."""
        return_info = {
            "type": (
                str(inspect.signature(self.obj).return_annotation)
                if inspect.signature(self.obj).return_annotation
                != inspect.Signature.empty
                else None
            ),
            "description": "",
        }

        if self.doc:
            # Look for :return: or Returns: section
            return_patterns = [r":return:\s*([^\n]+)", r"Returns:\s*([^\n]+)"]
            for pattern in return_patterns:
                match = re.search(pattern, self.doc)
                if match:
                    return_info["description"] = match.group(1).strip()
                    break

        return return_info

    def _categorize_method(self) -> Set[str]:
        """Categorize method based on name and documentation."""
        categories = set()

        # Common method categories and their indicators
        categorization_rules = {
            "create": {"create", "insert", "add", "new", "init"},
            "read": {"get", "fetch", "retrieve", "find", "search", "list"},
            "update": {"update", "modify", "change", "set"},
            "delete": {"delete", "remove", "clear"},
            "bulk": {"bulk", "batch", "many", "multiple"},
            "validation": {"validate", "check", "verify", "ensure"},
            "utility": {"format", "convert", "parse", "helper"},
            "error_handling": {"raise", "except", "error", "handle"},
        }

        # Check method name and summary against categories
        method_text = f"{self.name} {self.summary}".lower()
        for category, indicators in categorization_rules.items():
            if any(indicator in method_text for indicator in indicators):
                categories.add(category)

        # Add error_handling category if method has documented exceptions
        if self.exceptions:
            categories.add("error_handling")

        return categories

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "doc": self.doc,
            "signature": self.signature,
            "module": self.module,
            "summary": self.summary,
            "parameters": self.parameters,
            "examples": self.examples,
            "exceptions": self.exceptions,
            "return_info": self.return_info,
            "categories": list(self._categorize_method()),
        }


class MethodAnalyzer:
    """Analyzes methods in a package."""

    def __init__(self):
        self._module_cache = {}

    def _get_module(self, package_name: str) -> Any:
        """Get module with caching."""
        if package_name not in self._module_cache:
            with timing.measure("package_import", f"Importing package {package_name}"):
                self._module_cache[package_name] = importlib.import_module(package_name)
        return self._module_cache[package_name]

    def quick_scan(self, package_name: str) -> List[Tuple[str, str, List[str]]]:
        """Quick scan of all methods in a package."""
        with timing.measure("method_discovery", "Discovering methods"):
            module = self._get_module(package_name)
            methods = []

            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    methods.append((name, obj))
                elif inspect.isclass(obj):
                    for method_name, method_obj in inspect.getmembers(obj):
                        if (
                            inspect.isfunction(method_obj)
                            or inspect.ismethod(method_obj)
                        ) and not method_name.startswith("_"):
                            methods.append((f"{name}.{method_name}", method_obj))

        results = []
        with timing.measure("method_analysis", "Analyzing methods"):
            progress = tqdm(methods, desc="Analyzing methods", unit="method")
            for name, obj in progress:
                try:
                    result = self._analyze_method_quick(name, obj)
                    if result:
                        results.append(result)
                    progress.set_postfix({"current": name})
                except Exception as e:
                    logger.debug(f"Error analyzing method {name}: {e}")

        return sorted(results, key=lambda x: x[0])

    def _analyze_method_quick(
        self, name: str, obj: Any
    ) -> Optional[Tuple[str, str, List[str]]]:
        """Quick analysis of a single method with caching."""
        cache_key = f"{obj.__module__}.{name}"

        # Try persistent cache first
        cached_result = analysis_cache.get(obj.__module__, name, obj)
        if cached_result is not None:
            return cached_result

        # Try memory cache
        if cache_key in method_analysis_cache:
            return method_analysis_cache[cache_key]

        try:
            info = MethodInfo(obj, name)
            result = (name, info.summary, self._categorize_method(info))

            # Cache in both memory and persistent storage
            method_analysis_cache[cache_key] = result
            analysis_cache.set(obj.__module__, name, obj, result)

            return result
        except Exception as e:
            logger.debug(f"Error in quick analysis of {name}: {e}")
            return None

    def deep_analyze(
        self, package_name: str, method_name: str
    ) -> Optional[Dict[str, Any]]:
        """Deep analysis of a specific method."""
        with timing.measure("deep_analysis", f"Deep analysis of {method_name}"):
            module = self._get_module(package_name)

            # Try to find the method
            method_parts = method_name.split(".")
            obj = module

            for part in method_parts:
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    return None

            if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
                return None

            # Check cache first
            cache_key = f"{package_name}.{method_name}"
            if cache_key in method_analysis_cache:
                return method_analysis_cache[cache_key]

            try:
                info = MethodInfo(obj, method_name)
                result = {
                    "name": method_name,
                    "signature": info.signature,
                    "doc": info.doc,
                    "summary": info.summary,
                    "parameters": info.parameters,
                    "return_info": info.return_info,
                    "exceptions": info.exceptions,
                    "examples": info.examples,
                    "categories": self._categorize_method(info),
                }
                method_analysis_cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"Error in deep analysis of {method_name}: {e}")
                return None

    def _categorize_method(self, info: MethodInfo) -> List[str]:
        """Categorize a method based on its characteristics."""
        categories = []

        # Add categories based on method characteristics
        if "async" in info.signature or info.name.startswith(("a", "async_")):
            categories.append("async")
        if "stream" in info.name.lower():
            categories.append("streaming")
        if any(word in info.name.lower() for word in ["get", "fetch", "retrieve"]):
            categories.append("getter")
        if any(word in info.name.lower() for word in ["set", "update", "modify"]):
            categories.append("setter")
        if "cache" in info.name.lower():
            categories.append("caching")
        if "validate" in info.name.lower():
            categories.append("validation")
        if any(word in info.doc.lower() for word in ["helper", "utility"]):
            categories.append("utility")

        return categories or ["general"]

    def get_exception_summary(
        self, package_name: str
    ) -> List[Tuple[str, str, List[str]]]:
        """Get a summary of exceptions used in the package."""
        exceptions = {}

        # Analyze methods to find exceptions
        with timing.measure("exception_analysis", "Analyzing exceptions"):
            module = self._get_module(package_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Exception):
                    exceptions[name] = {
                        "description": inspect.getdoc(obj)
                        or "No description available",
                        "raised_by": [],
                    }

            # Scan methods to find where exceptions are raised
            methods = self.quick_scan(package_name)
            for name, _, _ in methods:
                method_info = self.deep_analyze(package_name, name)
                if method_info and method_info.get("exceptions"):
                    for exc in method_info["exceptions"]:
                        exc_name = exc["type"]
                        if exc_name not in exceptions:
                            exceptions[exc_name] = {
                                "description": exc["description"],
                                "raised_by": [],
                            }
                        exceptions[exc_name]["raised_by"].append(name)

        # Convert to list format
        return [
            (name, info["description"], info["raised_by"])
            for name, info in exceptions.items()
        ]


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
        r".*method_validator\.py\s+\w+\s+(--method|--list-all|--exceptions-only)",
    ]

    return any(re.match(pattern, command) for pattern in auto_executable_patterns)


def main():
    """Main entry point for the method validator."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "package",
        help=(
            "Non-standard package to analyze. DO NOT use for standard library "
            "or common utility packages."
        ),
    )
    parser.add_argument("--method", help="Method name for detailed analysis")
    parser.add_argument(
        "--list-all", action="store_true", help="Quick scan of all available methods"
    )
    parser.add_argument(
        "--by-category", action="store_true", help="Group methods by category"
    )
    parser.add_argument(
        "--show-exceptions",
        action="store_true",
        help="Show detailed exception information",
    )
    parser.add_argument(
        "--exceptions-only",
        action="store_true",
        help="Show only exception information for the package",
    )
    parser.add_argument(
        "--venv-path",
        help="Virtual environment path (auto-detected if not provided)",
        default=find_venv_path(),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format for machine consumption",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show execution timing statistics",
    )

    args = parser.parse_args()

    # Enable/disable timing based on argument
    timing.enabled = args.timing

    if not should_analyze_package(args.package):
        if args.json:
            print(
                json.dumps(
                    {"error": "Package should not be analyzed", "package": args.package}
                )
            )
            sys.exit(1)
        logger.warning(
            f"Package '{args.package}' is a standard/utility package and should not be analyzed."
        )
        logger.info(
            "This tool is intended for analyzing non-standard packages relevant to your current task."
        )
        sys.exit(1)

    try:
        with timing.measure("total_execution", "Total execution time"):
            analyzer = MethodAnalyzer()
            result = {}

            if args.exceptions_only:
                with timing.measure(
                    "exception_summary", "Generating exception summary"
                ):
                    exceptions = analyzer.get_exception_summary(args.package)
                    if args.json:
                        result["exceptions"] = [
                            {
                                "name": name,
                                "description": description,
                                "raised_by": raised_by,
                            }
                            for name, description, raised_by in exceptions
                        ]
                    else:
                        if not exceptions:
                            logger.warning(
                                "No exceptions found in package documentation"
                            )
                            return

                        for exc_name, description, raised_by in exceptions:
                            logger.info(f"\n{exc_name}:")
                            logger.info(f"  Description: {description}")
                            if raised_by:
                                logger.info(f"  Raised by {len(raised_by)} methods:")
                                for method in raised_by[:5]:
                                    logger.info(f"    - {method}")
                                if len(raised_by) > 5:
                                    logger.info(
                                        f"    ... and {len(raised_by) - 5} more"
                                    )

            elif args.list_all:
                with timing.measure("quick_scan", "Performing quick scan"):
                    methods = analyzer.quick_scan(args.package)
                    if args.json:
                        result["methods"] = []
                        if args.by_category:
                            by_category = {}
                            for name, summary, categories in methods:
                                for category in categories:
                                    by_category.setdefault(category, []).append(
                                        {"name": name, "summary": summary}
                                    )
                            result["methods_by_category"] = by_category
                        else:
                            result["methods"] = [
                                {"name": name, "summary": summary}
                                for name, summary, _ in methods
                            ]
                    else:
                        if args.by_category:
                            by_category = {}
                            for name, summary, categories in methods:
                                for category in categories:
                                    by_category.setdefault(category, []).append(
                                        (name, summary)
                                    )

                            for category, methods in by_category.items():
                                logger.info(f"\n[{category.upper()}]")
                                for name, summary in methods:
                                    logger.info(f"\n{name}:")
                                    if summary:
                                        logger.info(f"  Summary: {summary}")
                        else:
                            for name, summary, _ in methods:
                                logger.info(f"\n{name}:")
                                if summary:
                                    logger.info(f"  Summary: {summary}")

            elif args.method:
                with timing.measure("deep_analysis", f"Analyzing method {args.method}"):
                    method_info = analyzer.deep_analyze(args.package, args.method)
                    if method_info:
                        if args.json:
                            result["method_info"] = method_info
                        else:
                            logger.info(f"\nDetailed analysis of '{args.method}':")
                            logger.info(f"Signature: {method_info['signature']}")
                            logger.info(
                                f"Categories: {', '.join(method_info['categories'])}"
                            )

                            if method_info["doc"]:
                                logger.info(f"\nDocumentation:\n{method_info['doc']}")

                            logger.info("\nParameters:")
                            for name, details in method_info["parameters"].items():
                                required = (
                                    "required" if details["required"] else "optional"
                                )
                                logger.info(f"  {name} ({required}):")
                                if details["type"]:
                                    logger.info(f"    Type: {details['type']}")
                                if details["description"]:
                                    logger.info(
                                        f"    Description: {details['description']}"
                                    )
                                if details["default"]:
                                    logger.info(f"    Default: {details['default']}")

                            if args.show_exceptions and method_info["exceptions"]:
                                logger.info("\nExceptions:")
                                for exc in method_info["exceptions"]:
                                    logger.info(f"  {exc['type']}:")
                                    logger.info(
                                        f"    Description: {exc['description']}"
                                    )
                                    if exc["hierarchy"]:
                                        logger.info(
                                            f"    Hierarchy: {' -> '.join(exc['hierarchy'])}"
                                        )

                            if method_info["return_info"]["type"]:
                                logger.info(
                                    f"\nReturns: {method_info['return_info']['type']}"
                                )
                                if method_info["return_info"]["description"]:
                                    logger.info(
                                        f"  {method_info['return_info']['description']}"
                                    )

                            if method_info["examples"]:
                                logger.info("\nExamples:")
                                for example in method_info["examples"]:
                                    logger.info(f"\n{example}")
                    else:
                        similar = [
                            name
                            for name, _, _ in analyzer.quick_scan(args.package)
                            if args.method.lower() in name.lower()
                        ]
                        if args.json:
                            result["error"] = f"Method '{args.method}' not found"
                            if similar:
                                result["similar_methods"] = similar
                        else:
                            logger.warning(f"\nMethod '{args.method}' not found.")
                            if similar:
                                logger.info("Similar methods found:")
                                for method in similar:
                                    logger.info(f"  - {method}")

            else:
                if args.json:
                    result["error"] = "No action specified"
                else:
                    parser.print_help()

            if args.timing:
                timing_stats = timing.get_summary()
                if args.json:
                    result["timing_stats"] = timing_stats
                else:
                    logger.info("\nTiming Statistics:")
                    for op, stats in timing_stats.items():
                        logger.info(f"\n{op}:")
                        logger.info(f"  Total time: {stats['total_time']:.3f}s")
                        logger.info(f"  Calls: {stats['calls']}")
                        logger.info(f"  Average time: {stats['average_time']:.3f}s")

            if args.json:
                print(json.dumps(result))

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
