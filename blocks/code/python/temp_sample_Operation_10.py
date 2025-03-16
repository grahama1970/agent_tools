# Original file: temp_sample.py
# Block type: class
# Name: Operation

class Operation:
        """Represents a calculator operation."""

        def __init__(self, name: str, function):
            self.name = name
            self.function = function

        def execute(self, *args, **kwargs):
            """Execute the operation."""
            return self.function(*args, **kwargs)