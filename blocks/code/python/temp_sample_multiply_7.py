# Original file: temp_sample.py
# Block type: function
# Name: multiply

def multiply(self, x: float) -> float:
        """Multiply the current value by a number."""
        self.value *= x
        self._operations.append(f"multiply({x})")
        return self.value