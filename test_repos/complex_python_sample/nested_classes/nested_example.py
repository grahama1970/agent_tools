#!/usr/bin/env python3
"""
Example of complex nested class structures with decorators and docstrings.
This tests the AST extractor's ability to accurately parse nested structures.
"""

import typing
from typing import Optional, List, Dict, Any, Callable
from functools import wraps


def debug_method(func):
    """Decorator that adds debug logging to methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"DEBUG: Calling {func.__name__} with {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"DEBUG: {func.__name__} returned {result}")
        return result
    return wrapper


class OuterClass:
    """
    A parent class that contains nested classes and decorated methods.
    
    This class demonstrates complex nesting patterns and inheritance.
    """
    
    class_variable = "outer_value"
    
    def __init__(self, name: str):
        """Initialize the outer class with a name."""
        self.name = name
        self.inner_instances = []
    
    @debug_method
    def outer_method(self, param: str) -> str:
        """
        A method in the outer class with a decorator.
        
        Args:
            param: Input parameter
            
        Returns:
            Modified parameter
        """
        return f"Outer: {param}"
    
    class InnerClass:
        """
        A class nested within the outer class.
        
        This inner class has its own methods and attributes.
        """
        
        def __init__(self, value: int):
            """Initialize inner class with a value."""
            self.value = value
            self.nested_data = {}
        
        def inner_method(self) -> int:
            """Return the inner value multiplied by 2."""
            return self.value * 2
        
        class DeepNestedClass:
            """
            A deeply nested class (three levels deep).
            
            This tests the parser's ability to handle multi-level nesting.
            """
            
            def __init__(self, label: str):
                """Initialize with a label."""
                self.label = label
            
            @staticmethod
            def deep_static_method() -> str:
                """A static method in the deeply nested class."""
                return "Deep static result"
            
            @classmethod
            def deep_class_method(cls) -> str:
                """A class method in the deeply nested class."""
                return f"Class method from {cls.__name__}"
    
    class AnotherInnerClass:
        """Another inner class for testing multiple nested classes."""
        
        @property
        def computed_property(self) -> str:
            """A property decorator example."""
            return "Computed value"


# Usage example
if __name__ == "__main__":
    outer = OuterClass("test")
    inner = outer.InnerClass(42)
    deep = inner.DeepNestedClass("deep")
    
    print(outer.outer_method("hello"))
    print(inner.inner_method())
    print(deep.deep_static_method())