# Original file: temp_sample.py
# Block type: function
# Name: is_numeric
# Docstring: Check if a value is a number.
# Parameters: value

def is_numeric(value) -> bool:
    """Check if a value is a number."""
    return isinstance(value, (int, float, complex))