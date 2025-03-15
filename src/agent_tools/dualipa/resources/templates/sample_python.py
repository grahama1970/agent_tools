def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

class Calculator:
    """Simple calculator class."""
    def add(self, a, b):
        """Add two numbers."""
        return a + b
        
    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b

if __name__ == "__main__":
    print(greet("World")) 