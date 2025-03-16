# Original file: temp_sample.py
# Block type: function
# Name: add

def add(self, x: float) -> float:
        """Add a number to the current value."""
        self.value += x
        self._operations.append(f"add({x})")
        return self.value