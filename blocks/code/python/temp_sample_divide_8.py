# Original file: temp_sample.py
# Block type: function
# Name: divide

def divide(self, x: float) -> float:
        """Divide the current value by a number."""
        if x == 0:
            raise MathError("Division by zero")
        self.value /= x
        self._operations.append(f"divide({x})")
        return self.value