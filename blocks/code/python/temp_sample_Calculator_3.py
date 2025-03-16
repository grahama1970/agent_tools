# Original file: temp_sample.py
# Block type: class
# Name: Calculator
# Docstring: A simple calculator class.

class Calculator:
    """A simple calculator class."""

    def __init__(self, initial_value: float = 0):
        """Initialize calculator with a starting value."""
        self.value = initial_value
        self._operations = []

    def add(self, x: float) -> float:
        """Add a number to the current value."""
        self.value += x
        self._operations.append(f"add({x})")
        return self.value

    def subtract(self, x: float) -> float:
        """Subtract a number from the current value."""
        self.value -= x
        self._operations.append(f"subtract({x})")
        return self.value

    def multiply(self, x: float) -> float:
        """Multiply the current value by a number."""
        self.value *= x
        self._operations.append(f"multiply({x})")
        return self.value

    def divide(self, x: float) -> float:
        """Divide the current value by a number."""
        if x == 0:
            raise MathError("Division by zero")
        self.value /= x
        self._operations.append(f"divide({x})")
        return self.value

    def get_history(self) -> list:
        """Get the history of operations."""
        return self._operations.copy()

    class Operation:
        """Represents a calculator operation."""

        def __init__(self, name: str, function):
            self.name = name
            self.function = function

        def execute(self, *args, **kwargs):
            """Execute the operation."""
            return self.function(*args, **kwargs)