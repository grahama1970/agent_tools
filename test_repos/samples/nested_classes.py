"""
Sample file with nested classes for testing code hierarchy extraction.

This file contains examples of nested class hierarchies which are uncommon
in real-world Python code but useful for testing hierarchy extraction.
"""

class OuterClass:
    """Example of a class containing nested classes."""
    
    class_var = "outer class variable"
    
    def __init__(self, name):
        self.name = name
    
    def outer_method(self):
        """Method in the outer class."""
        return f"Outer method from {self.name}"
    
    # Nested class example
    class InnerClass:
        """First level nested class."""
        
        inner_var = "inner class variable"
        
        def __init__(self, value):
            self.value = value
        
        def inner_method(self):
            """Method in the inner class."""
            return f"Inner method with {self.value}"
        
        # Double-nested class example
        class DeepNestedClass:
            """Second level nested class (very uncommon in Python)."""
            
            def __init__(self, id):
                self.id = id
            
            def deep_method(self):
                """Method in the deeply nested class."""
                return f"Deep method with ID {self.id}"


# Another example with different structure
class Parent:
    """Parent class with nested components."""
    
    def parent_method(self):
        """Parent method."""
        return "Parent method"
    
    # Static nested class
    class StaticNested:
        """Static nested class example."""
        
        @staticmethod
        def static_method():
            """Static method in nested class."""
            return "Static nested method"


# Usage example (not typically how these would be used)
def example_usage():
    # Create instances
    outer = OuterClass("Test")
    inner = OuterClass.InnerClass(42)
    deep = OuterClass.InnerClass.DeepNestedClass(100)
    
    # Call methods
    print(outer.outer_method())  # "Outer method from Test"
    print(inner.inner_method())  # "Inner method with 42"
    print(deep.deep_method())    # "Deep method with ID 100"
    
    # Static nested example
    print(Parent.StaticNested.static_method())  # "Static nested method"


if __name__ == "__main__":
    example_usage() 