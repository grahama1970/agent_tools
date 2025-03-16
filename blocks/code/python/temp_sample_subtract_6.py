# Original file: temp_sample.py
# Block type: function
# Name: subtract

def subtract(self, x: float) -> float:
        """Subtract a number from the current value."""
        self.value -= x
        self._operations.append(f"subtract({x})")
        return self.value