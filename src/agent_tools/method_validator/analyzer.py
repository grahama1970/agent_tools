"""Core analysis functionality for method validator."""

import inspect
import importlib
import re
import sys
from types import ModuleType
from typing import Dict, List, Any, Optional, Tuple, Set, Match, Callable, cast, Type
from dataclasses import dataclass
from tqdm import tqdm  # type: ignore
from loguru import logger
import json
import time
import tempfile
import subprocess

from .utils import timing

# Lazy loading of cache to avoid circular imports
_analysis_cache = None

def safe_get_signature(obj: Callable[..., Any]) -> str:
    """Safely get a string signature of a function, injecting sys if needed."""
    try:
        return str(inspect.signature(obj))
    except NameError:
        # The function's globals might be missing sys – inject it
        try:
            g = getattr(obj, '__globals__', {})
            if 'sys' not in g:
                g['sys'] = sys
        except Exception:
            pass
        try:
            return str(inspect.signature(obj))
        except Exception:
            return "()"

def safe_signature_object(obj: Callable[..., Any]) -> Optional[inspect.Signature]:
    """Return the Signature object safely, injecting sys if needed."""
    try:
        return inspect.signature(obj)
    except (NameError, ValueError):
        # Try to build signature from annotations
        if hasattr(obj, "__annotations__"):
            params = []
            for name, type_hint in obj.__annotations__.items():
                if name != "return":
                    params.append(
                        inspect.Parameter(
                            name=name,
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=type_hint,
                            default=inspect.Parameter.empty
                        )
                    )
            return_annotation = obj.__annotations__.get("return", inspect.Signature.empty)
            return inspect.Signature(params, return_annotation=return_annotation)
        try:
            g = getattr(obj, '__globals__', {})
            if 'sys' not in g:
                g['sys'] = sys
        except Exception:
            pass
        try:
            return inspect.signature(obj)
        except Exception:
            return None

def get_cache() -> Any:
    global _analysis_cache
    if _analysis_cache is None:
        logger.debug("Initializing analysis cache for the first time")
        from .cache import AnalysisCache
        _analysis_cache = AnalysisCache()
    return _analysis_cache

def validate_method(package_name: str, method_name: str) -> Tuple[bool, str]:
    """Quickly validate if a method exists and is accessible.
    
    This is a fast validation that checks:
    1. If the method exists in cache
    2. If not, does a quick import check
    3. Returns immediately without deep analysis
    
    Args:
        package_name: Name of the package containing the method
        method_name: Name of the method to validate
        
    Returns:
        Tuple of (is_valid, message)
        - is_valid: True if method exists and is accessible
        - message: Description of validation result
    """
    cache = get_cache()
    
    # Check cache first
    try:
        module = cast(ModuleType, importlib.import_module(package_name))
        if '.' in method_name:
            # Handle nested methods like _Logger.add
            parts = method_name.split('.')
            obj: Any = module
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    return False, f"Method {method_name} not found in {package_name}"
        else:
            obj = getattr(module, method_name, None)
            if obj is None:
                return False, f"Method {method_name} not found in {package_name}"
                
        # Basic validation that it's callable
        if not callable(obj):
            return False, f"{method_name} exists but is not callable"
            
        return True, f"Method {method_name} exists and is callable"
        
    except ImportError:
        return False, f"Could not import package {package_name}"
    except Exception as e:
        return False, f"Error validating method: {str(e)}"

@dataclass
class MethodInfo:
    """Structured container for method information.
    
    Args:
        obj: The method object to analyze
        name: Name of the method
        
    Attributes:
        name: Method name
        doc: Method docstring
        signature: Method signature
        module: Module name
        summary: Brief description
        parameters: Parameter details
        examples: Usage examples
        exceptions: Possible exceptions
        return_info: Return type information
    """

    def __init__(self, obj: Any, name: str) -> None:
        if obj is None:
            raise ValueError(f"Cannot analyze None object for method {name}")

        self.name = name
        self.obj = obj
        try:
            with timing.measure("docstring_extraction"):
                self.doc = inspect.getdoc(obj) or ""
                self.signature = safe_get_signature(obj)
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

            with timing.measure("quality_analysis"):
                self.quality_info = self._analyze_code_quality()

            with timing.measure("type_analysis"):
                self.type_info = self._analyze_types()
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
            self.quality_info = {"issues": [], "suggestions": []}
            self.type_info = {"errors": [], "suggestions": []}

    def _generate_summary(self) -> str:
        """Generate a quick summary from the docstring."""
        if not self.doc:
            return ""
        summary = self.doc.split("\n")[0].split(".")[0]
        return summary[:100] + "..." if len(summary) > 100 else summary

    def _analyze_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter types, defaults, and constraints."""
        params: Dict[str, Dict[str, Any]] = {}
        signature = safe_signature_object(self.obj)
        if not signature:
            return params

        # Extract parameters mentioned in examples
        example_params: Set[str] = set()
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

            # Include parameter if it's relevant enough or has type annotations
            if (relevance >= 1 or  # Less strict relevance requirement
                param.annotation != param.empty or  # Has type annotation
                param.default == param.empty or  # Required parameter
                name in {  # Core parameters
                    "model", "messages", "stream", "api_key",
                    "obj", "data", "text", "file", "path",  # Common param names
                    "input", "output", "value", "key"  # More common param names
                }):
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
            code_blocks = re.finditer(
                r"(?:```(?:python)?\n(.*?)\n```)|(?:(?:^|\n)\s{4}(.*?)(?=\n\S|\Z))|(?:>>>\s*(.*?)(?:\n|$))",
                example_section[1],
                re.DOTALL | re.MULTILINE
            )
            for block in code_blocks:
                # Get the first non-None group
                code = next((g for g in block.groups() if g is not None), "").strip()
                if code:
                    examples.append(code)

        return examples

    def _analyze_exceptions(self) -> List[Dict[str, str]]:
        """Analyze exceptions that can be raised by the method.
        
        Returns:
            List of dictionaries containing exception information:
            - type: Exception class name
            - description: Exception description
            - hierarchy: Exception class hierarchy
            - source: Where the exception was found
        """
        exceptions: List[Dict[str, str]] = []
        if not self.doc:
            return exceptions

        # Look for explicitly documented exceptions first (highest priority)
        raise_patterns = [
            r":raises\s+(\w+):\s*([^\n]+)",
            r"Raises:\s*(?:\n\s*)?(?:-\s*)?(\w+):\s*([^\n]+)",
            r"Raises\s*-+\s*(\w+):\s*([^\n]+)",
            r"Raises:\s*(?:\n\s*)?(?:-\s*)?(\w+):\s*([^\n]+)(?:\n\s*(?:-\s*)?(\w+):\s*([^\n]+))*",
        ]

        for pattern in raise_patterns:
            matches = re.finditer(pattern, self.doc, re.MULTILINE | re.IGNORECASE)
            for doc_match in matches:
                groups = doc_match.groups()
                if len(groups) >= 2:
                    # Handle single exception
                    exc_name, desc = groups[0], groups[1]
                    if desc and len(desc.strip()) > 3:  # Only include exceptions with meaningful descriptions
                        exceptions.append(
                            {
                                "type": exc_name,
                                "description": desc.strip(),
                                "hierarchy": ",".join(self._get_exception_hierarchy(exc_name)),
                                "source": "documentation",
                            }
                        )
                    # Handle additional exceptions in the same block
                    for i in range(2, len(groups), 2):
                        if groups[i] and groups[i+1]:
                            exc_name, desc = groups[i], groups[i+1]
                            if desc and len(desc.strip()) > 3:
                                exceptions.append(
                                    {
                                        "type": exc_name,
                                        "description": desc.strip(),
                                        "hierarchy": ",".join(self._get_exception_hierarchy(exc_name)),
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
                    class_match = re.search(r"class\s+(\w+Error)", line)
                    if class_match:
                        custom_exceptions.add(class_match.group(1))

            # Second pass: find raise statements
            raise_statements = re.finditer(r"raise\s+(\w+)(?:\(|$)", source)
            for raise_match in raise_statements:
                exc_name = raise_match.group(1)
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
                                "hierarchy": ",".join(self._get_exception_hierarchy(exc_name)),
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

    def _analyze_return_info(self) -> Dict[str, Optional[str]]:
        """Analyze return type and description.
        
        Returns:
            Dict with keys:
                type: Return type annotation if available
                description: Return value description from docstring
        """
        return_info: Dict[str, Optional[str]] = {
            "type": None,
            "description": None,
        }

        def format_type(type_str: str) -> str:
            """Format a type string to be more readable."""
            # Handle class types like "<class 'str'>"
            if type_str.startswith("<class '") and type_str.endswith("'>"):
                return type_str[8:-2]
            # Handle typing types
            type_str = re.sub(r"typing\.", "", type_str)
            # Handle Union types
            type_str = re.sub(r"Union\[(.*?)\]", r"\1", type_str)
            return type_str

        # Get return type from signature
        signature = safe_signature_object(self.obj)
        if signature and signature.return_annotation != inspect.Signature.empty:
            return_info["type"] = format_type(str(signature.return_annotation))

        # Get return type from annotations
        if hasattr(self.obj, "__annotations__") and "return" in self.obj.__annotations__:
            return_info["type"] = format_type(str(self.obj.__annotations__["return"]))

        if self.doc:
            # Look for :return: or Returns: section
            return_patterns = [
                r":return:\s*([^\n]+)",
                r"Returns:\s*([^\n]+)",
                r"Returns\s*-+\s*([^\n]+)",
            ]
            for pattern in return_patterns:
                match = re.search(pattern, self.doc, re.IGNORECASE)
                if match:
                    desc = match.group(1).strip()
                    return_info["description"] = desc
                    # Try to extract type from description (e.g. "Returns: str: The JSON string")
                    if desc:
                        type_match = re.match(r"([^:]+):\s*.*", desc)
                        if type_match and not return_info["type"]:
                            return_info["type"] = format_type(type_match.group(1).strip())
                    break

        return return_info

    def _analyze_code_quality(self) -> Dict[str, List[str]]:
        """Analyze code quality using prospector."""
        quality_info: Dict[str, List[str]] = {"issues": [], "suggestions": []}
        try:
            # Get source code
            source = inspect.getsource(self.obj)
            
            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Add proper indentation to avoid syntax errors
                source_lines = source.split('\n')
                dedented_source = '\n'.join(line[min(len(line) - len(line.lstrip()), 4):] for line in source_lines)
                temp_file.write(dedented_source)
                temp_file.flush()
                
                # Run pycodestyle directly for formatting issues
                result = subprocess.run(
                    ['pycodestyle', '--first', temp_file.name],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    for line in result.stdout.splitlines():
                        if ':' in line:
                            # Extract the actual issue message
                            parts = line.split(':', 3)  # Split on first 3 colons
                            if len(parts) > 3:
                                issue = parts[3].strip()
                                # Only include significant formatting issues
                                if not issue.startswith('W293'):  # Ignore blank line contains whitespace
                                    quality_info["issues"].append(issue)
                
                # Also run prospector for additional issues
                result = subprocess.run(
                    ['prospector', temp_file.name, '--no-autodetect', '--no-external-config',
                     '--output-format=text', '--without-tool=pylint_django'],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    for line in result.stdout.splitlines():
                        if ('multiple statements' in line.lower() or 
                            'missing whitespace' in line.lower() or
                            'operator should be surrounded by whitespace' in line.lower()):
                            quality_info["issues"].append(line.split(':', 1)[1].strip() if ':' in line else line.strip())

                # Add suggestions based on issues
                if quality_info["issues"]:
                    if any("=" in issue or ";" in issue or 
                          "whitespace" in issue.lower() or
                          "multiple statements" in issue.lower()
                          for issue in quality_info["issues"]):
                        quality_info["suggestions"].append(
                            "Fix code formatting issues to improve readability"
                        )
                    if any("unused" in issue.lower() for issue in quality_info["issues"]):
                        quality_info["suggestions"].append(
                            "Remove unused imports or variables"
                        )
                    if any("complexity" in issue.lower() for issue in quality_info["issues"]):
                        quality_info["suggestions"].append(
                            "Consider simplifying complex code blocks"
                        )
                        
                # If no issues were found but we have poorly formatted code, add default messages
                if not quality_info["issues"] and any(
                    line.strip() and ('=' in line and not ' = ' in line) or ';' in line
                    for line in source.split('\n')
                ):
                    quality_info["issues"].append("Multiple statements on one line or missing whitespace around operators")
                    quality_info["suggestions"].append("Fix code formatting issues to improve readability")
                    
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            
        return quality_info

    def _analyze_types(self) -> Dict[str, List[str]]:
        """Analyze type hints using mypy."""
        type_info: Dict[str, List[str]] = {"errors": [], "suggestions": []}
        try:
            # Get source code
            source = inspect.getsource(self.obj)
            
            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Add proper indentation and imports
                source_lines = source.split('\n')
                dedented_source = '\n'.join(line[min(len(line) - len(line.lstrip()), 4):] for line in source_lines)
                temp_file.write("from typing import Any, Dict, List, Optional, Union\n\n")
                temp_file.write(dedented_source)
                temp_file.flush()
                
                # Run mypy with strict settings
                result = subprocess.run(
                    ['mypy', '--strict', '--show-error-codes', '--no-error-summary', temp_file.name],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    for line in result.stdout.splitlines():
                        if ": error:" in line:
                            error_msg = line.split(": error:", 1)[1].strip()
                            # Convert mypy error message to our expected format
                            if "[no-untyped-def]" in error_msg:
                                error_msg = "Missing type annotation"
                            type_info["errors"].append(error_msg)
                            
                            # Add specific suggestions based on error types
                            if "Missing type annotation" in error_msg:
                                if not any("Add type hints" in s for s in type_info["suggestions"]):
                                    type_info["suggestions"].append(
                                        "Add type hints to improve code clarity and catch type errors"
                                    )
                            elif "Incompatible type" in error_msg:
                                if not any("Fix type mismatches" in s for s in type_info["suggestions"]):
                                    type_info["suggestions"].append(
                                        "Fix type mismatches to ensure type safety"
                                    )
                            elif "union" in error_msg.lower():
                                if not any("Union types" in s for s in type_info["suggestions"]):
                                    type_info["suggestions"].append(
                                        "Consider using Union types for multiple possible types"
                                    )
        except Exception as e:
            logger.error(f"Type analysis failed: {e}")
            
        return type_info

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
            "quality_info": self.quality_info,
            "type_info": self.type_info
        }


class MethodAnalyzer:
    """Analyzes methods in a package."""

    def __init__(self, target_file: Optional[str] = None, include_builtins: bool = False):
        self.target_file = target_file
        self.include_builtins = include_builtins
        self._cache = get_cache()
        self._module_cache: Dict[str, ModuleType] = {}

    def _get_module(self, package_name: str) -> Any:
        """Get module with caching."""
        if package_name not in self._module_cache:
            with timing.measure("package_import", f"Importing package {package_name}"):
                self._module_cache[package_name] = importlib.import_module(package_name)
        return self._module_cache[package_name]

    def _should_analyze_method(self, obj: Any) -> bool:
        """Determine if a method should be analyzed based on its source file."""
        if not self.target_file:
            if self.include_builtins:
                return True
            try:
                source_file = inspect.getfile(obj)
                return True
            except (TypeError, OSError):
                return False
            
        try:
            source_file = inspect.getfile(obj)
            return self.target_file in source_file
        except (TypeError, OSError):
            return self.include_builtins

    def quick_scan(self, package_name: str) -> List[Tuple[str, str, List[str]]]:
        """Quick scan of all methods in a package."""
        with timing.measure("method_discovery", "Discovering methods"):
            module = self._get_module(package_name)
            methods = []

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isfunction(obj) or inspect.ismethod(obj)
                ) and self._should_analyze_method(obj):
                    methods.append((name, obj))
                elif inspect.isclass(obj):
                    for method_name, method_obj in inspect.getmembers(obj):
                        if (
                            (
                                inspect.isfunction(method_obj)
                                or inspect.ismethod(method_obj)
                            )
                            and not method_name.startswith("_")
                            and self._should_analyze_method(method_obj)
                        ):
                            methods.append((f"{name}.{method_name}", method_obj))

        results = []
        with timing.measure("method_analysis", "Analyzing methods"):
            # Check if running in a test environment
            is_test = 'pytest' in sys.modules
            if is_test:
                # Skip progress bar in tests
                for name, obj in methods:
                    try:
                        result = self._analyze_method_quick(name, obj)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.debug(f"Error analyzing method {name}: {e}")
            else:
                # Show progress bar in normal operation
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
        """Quick analysis of a single method with caching.
        
        Args:
            name: Method name
            obj: Method object
            
        Returns:
            Tuple of (name, summary, categories) if successful,
            None if analysis fails
        """
        cache_key = f"{obj.__module__}.{name}"
        logger.debug(f"Quick analyzing method: {cache_key}")

        # Try persistent cache first
        cached_result = self._cache.get(obj.__module__, name, obj)
        if cached_result is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cast(Tuple[str, str, List[str]], cached_result)

        try:
            info = MethodInfo(obj, name)
            result = (name, info.summary, list(info._categorize_method()))
            logger.debug(f"Generated result for {cache_key}: {result[:2]}")  # Don't log full result

            # Cache the result
            self._cache.set(obj.__module__, name, obj, result)
            logger.debug(f"Cached result for {cache_key}")

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

            try:
                info = MethodInfo(obj, method_name)
                result = info.to_dict()
                return result
            except Exception as e:
                logger.warning(f"Error in deep analysis of {method_name}: {e}")
                return None

def usage_examples() -> None:
    """Example usage of the Method Validator tool.
    
    These examples demonstrate how to use the Method Validator for both development
    and debugging purposes. The tool helps prevent common AI pitfalls like method
    hallucination while providing useful insights for developers.
    
    Key Features Demonstrated:
    1. Method Validation - Verify method existence
    2. Package Analysis - Discover available methods
    3. Deep Analysis - Get detailed method information
    4. Method Info - Extract specific method details
    5. Caching - Performance optimization
    
    Developer Notes:
    - Use quick validation (--quick flag) for rapid checks
    - Use deep analysis for detailed method information
    - Cache results are stored in ./analysis/method_analysis_cache.db
    - Standard libraries and common utilities are automatically filtered
    
    Common Use Cases:
    1. Exploring new packages: Use quick_scan() to discover methods
    2. API Validation: Use validate_method() to verify method existence
    3. Documentation: Use deep_analyze() to get detailed method info
    4. Debugging: Use MethodInfo for specific method analysis
    """
    
    def example_validate_method() -> None:
        """Quick validation of method existence.
        
        Use this when you need to:
        - Verify a method exists before using it
        - Check method accessibility
        - Validate API endpoints
        """
        exists, message = validate_method("requests", "get")
        print(f"Method validation: {message}")
        
        exists, message = validate_method("requests", "nonexistent_method")
        print(f"Invalid method: {message}")

    def example_quick_scan() -> None:
        """Scan a package for available methods.
        
        Use this when you need to:
        - Explore a new package
        - Discover available functionality
        - Understand method categories
        """
        analyzer = MethodAnalyzer()
        methods = analyzer.quick_scan("requests")
        print("\nAvailable methods in requests:")
        for name, summary, categories in methods[:5]:
            print(f"- {name}: {summary}")
            print(f"  Categories: {', '.join(categories)}")

    def example_deep_analysis() -> None:
        """Detailed analysis of a specific method.
        
        Use this when you need to:
        - Understand method parameters
        - Check return types
        - Review possible exceptions
        - Get complete documentation
        """
        analyzer = MethodAnalyzer()
        details = analyzer.deep_analyze("requests", "get")
        if details:
            print("\nDetailed analysis of requests.get:")
            print(f"Parameters: {json.dumps(details['parameters'], indent=2)}")
            print(f"Return type: {details['return_info']['type']}")
            print("Exceptions:")
            for exc in details['exceptions']:
                print(f"- {exc['type']}: {exc['description']}")

    def example_method_info() -> None:
        """Extract specific information about a method.
        
        Use this when you need to:
        - Get method signatures
        - Extract usage examples
        - Understand method categories
        - Access raw docstrings
        """
        import requests
        info = MethodInfo(requests.get, "get")
        print("\nMethod Info for requests.get:")
        print(f"Signature: {info.signature}")
        print(f"Summary: {info.summary}")
        print(f"Examples found: {len(info.examples)}")
        print(f"Categories: {info._categorize_method()}")

    def example_cache_usage() -> None:
        """Demonstrate cache functionality.
        
        Use this when you need to:
        - Understand caching behavior
        - Measure performance impact
        - Debug cache issues
        """
        cache = get_cache()
        analyzer = MethodAnalyzer()
        
        print("\nCache demonstration:")
        start = time.time()
        analyzer.deep_analyze("requests", "get")
        first_time = time.time() - start
        
        start = time.time()
        analyzer.deep_analyze("requests", "get")
        second_time = time.time() - start
        
        print(f"First call time: {first_time:.3f}s")
        print(f"Second call time (cached): {second_time:.3f}s")

    # Run all examples
    examples = [
        example_validate_method,
        example_quick_scan,
        example_deep_analysis,
        example_method_info,
        example_cache_usage
    ]
    
    print("\nMethod Validator Tool - Developer Examples")
    print("==========================================")
    print("This tool helps prevent method hallucination and validates API usage.")
    print("It's designed for both AI agents and human developers.\n")
    print("Key Features:")
    print("- Smart package analysis with filtering")
    print("- Detailed method discovery and validation")
    print("- Intelligent categorization of methods")
    print("- Exception pattern analysis")
    print("- Optimized caching system\n")
    
    for example in examples:
        print(f"\n{'='*50}")
        print(f"Running: {example.__name__}")
        if example.__doc__:
            print(f"Purpose: {example.__doc__.splitlines()[0]}")
        print(f"{'='*50}")
        example()

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("Method Validator Tool - Usage Examples")
    print("This will demonstrate various ways to use the tool")
    print("Note: These examples use the 'requests' package for demonstration")
    
    try:
        import requests
    except ImportError:
        print("\nPlease install 'requests' package to run examples:")
        print("pip install requests")
        print("\nThis tool works with any Python package, but we use")
        print("requests for these examples as it's well-known and stable.")
        sys.exit(1)
        
    usage_examples()
