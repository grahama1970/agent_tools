# Original file: temp_sample.py
# Block type: function
# Name: advanced_calculation

def advanced_calculation(op: str, a: float, b: float) -> float:
    """Perform an advanced calculation."""
    if op == "power":
        return math.pow(a, b)
    elif op == "root":
        if a < 0 and b % 2 == 0:
            raise MathError("Cannot take even root of negative number")
        return a ** (1/b)
    elif op == "log":
        if a <= 0 or b <= 0 or b == 1:
            raise MathError("Invalid logarithm parameters")
        return math.log(a, b)
    else:
        raise ValueError(f"Unknown operation: {op}")