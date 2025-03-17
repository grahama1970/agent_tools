#!/usr/bin/env python3
"""Sample Python file with various code structures for extraction testing."""

import os
import sys
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path


def simple_function():
    """A simple function with no parameters."""
    return "Hello World"


def function_with_params(a: int, b: str, c: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A more complex function with type annotations and default values.
    
    Args:
        a: An integer parameter
        b: A string parameter
        c: An optional list of dictionaries
        
    Returns:
        A dictionary with the parameters
    """
    if c is None:
        c = []
    
    return {
        "a": a,
        "b": b,
        "c": c
    }


@dataclass
class SimpleClass:
    """A simple class with a few attributes."""
    name: str
    value: int
    
    def get_description(self) -> str:
        """Return a description of this instance."""
        return f"{self.name}: {self.value}"


class ComplexClass:
    """A more complex class with methods and properties."""
    
    def __init__(self, name: str, items: List[str] = None):
        """Initialize the class.
        
        Args:
            name: The name of the instance
            items: Optional list of items
        """
        self.name = name
        self.items = items or []
        self._private_value = 0
    
    @property
    def item_count(self) -> int:
        """Get the number of items."""
        return len(self.items)
    
    def add_item(self, item: str) -> None:
        """Add an item to the list.
        
        Args:
            item: The item to add
        """
        self.items.append(item)
    
    @classmethod
    def create_empty(cls, name: str) -> 'ComplexClass':
        """Create an instance with no items.
        
        Args:
            name: The name for the new instance
            
        Returns:
            A new instance with an empty items list
        """
        return cls(name, [])


# A decorator function
def log_calls(func):
    """A decorator that logs when a function is called."""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


@log_calls
def decorated_function(value: str) -> str:
    """A function with a decorator.
    
    Args:
        value: Input value
        
    Returns:
        Modified value
    """
    return f"Decorated: {value}"


if __name__ == "__main__":
    # Script execution code
    print("This is a sample Python file")
    simple = SimpleClass("Test", 42)
    print(simple.get_description())
    
    complex_obj = ComplexClass("Complex")
    complex_obj.add_item("Item 1")
    complex_obj.add_item("Item 2")
    
    print(f"Complex has {complex_obj.item_count} items") 