"""Core analysis functionality for method validator."""

import inspect
import importlib
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

from .cache import timing, analysis_cache


@dataclass
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
        """Analyze parameter types, defaults, and constraints."""
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
        """Analyze exceptions that can be raised by the method."""
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

    def __init__(self, target_file: Optional[str] = None):
        self._module_cache = {}
        self._target_file = target_file

    def _get_module(self, package_name: str) -> Any:
        """Get module with caching."""
        if package_name not in self._module_cache:
            with timing.measure("package_import", f"Importing package {package_name}"):
                self._module_cache[package_name] = importlib.import_module(package_name)
        return self._module_cache[package_name]

    def _should_analyze_method(self, obj: Any) -> bool:
        """Determine if a method should be analyzed based on its source file."""
        if not self._target_file:
            return True

        try:
            source_file = inspect.getfile(obj)
            return self._target_file in source_file
        except (TypeError, OSError):
            return False

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

        try:
            info = MethodInfo(obj, name)
            result = (name, info.summary, list(info._categorize_method()))

            # Cache the result
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

            try:
                info = MethodInfo(obj, method_name)
                result = info.to_dict()
                return result
            except Exception as e:
                logger.warning(f"Error in deep analysis of {method_name}: {e}")
                return None
