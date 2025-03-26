#!/usr/bin/env python3
"""
Example of advanced decorator patterns and descriptor usage.
This tests the AST extractor's ability to handle complex decorators.
"""

import functools
import time
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast


F = TypeVar('F', bound=Callable[..., Any])


def timing_decorator(func: F) -> F:
    """
    Decorator that measures execution time of a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function with timing capabilities
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return cast(F, wrapper)


def retry(attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """
    Decorator factory that creates a retry decorator with specified attempts and delay.
    
    Args:
        attempts: Number of attempts to retry
        delay: Delay between attempts in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts:
                        raise
                    print(f"Attempt {attempt} failed with {e}, retrying in {delay} seconds...")
                    time.sleep(delay)
        return cast(F, wrapper)
    return decorator


def validate_types(func: F) -> F:
    """
    Decorator that validates function arguments against type annotations.
    
    Args:
        func: Function to validate
        
    Returns:
        Wrapped function with type validation
    """
    signature = inspect.signature(func)
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_args = signature.bind(*args, **kwargs)
        for param_name, param_value in bound_args.arguments.items():
            param = signature.parameters.get(param_name)
            if param and param.annotation != inspect.Parameter.empty:
                if not isinstance(param_value, param.annotation):
                    raise TypeError(
                        f"Parameter '{param_name}' must be of type {param.annotation.__name__}, "
                        f"got {type(param_value).__name__}"
                    )
        return func(*args, **kwargs)
    return cast(F, wrapper)


class LazyProperty:
    """
    Descriptor for lazy evaluation of properties.
    
    The property value is computed on first access and then cached.
    """
    
    def __init__(self, func: Callable) -> None:
        """
        Initialize with the function to compute the property value.
        
        Args:
            func: Function that computes the property value
        """
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        
    def __get__(self, instance: Any, owner: Type) -> Any:
        """
        Get the property value, computing it if necessary.
        
        Args:
            instance: Object instance
            owner: Class of the instance
            
        Returns:
            Computed property value
        """
        if instance is None:
            return self
        
        value = self.func(instance)
        setattr(instance, self.__name__, value)
        return value


class CachedDecorator:
    """Class decorator that adds result caching to methods."""
    
    def __init__(self) -> None:
        """Initialize the cache."""
        self.cache: Dict[str, Any] = {}
    
    def __call__(self, func: F) -> F:
        """
        Decorate a function with caching.
        
        Args:
            func: Function to decorate
            
        Returns:
            Wrapped function with caching
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a cache key based on function name and arguments
            key = f"{func.__name__}:{hash(str(args))}-{hash(str(kwargs))}"
            
            if key not in self.cache:
                self.cache[key] = func(*args, **kwargs)
            
            return self.cache[key]
        
        return cast(F, wrapper)


# Usage examples
class ComplexClass:
    """Class with various decorated methods and descriptors."""
    
    def __init__(self, name: str) -> None:
        """Initialize with a name."""
        self.name = name
        self.data: Dict[str, Any] = {}
    
    @timing_decorator
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data with timing measurement."""
        time.sleep(0.1)  # Simulate processing
        self.data.update(input_data)
        return self.data
    
    @retry(attempts=3, delay=0.5)
    def external_api_call(self, endpoint: str) -> str:
        """Call an external API with retry logic."""
        # Simulate API call that might fail
        if endpoint == "failing" and not hasattr(self, "_succeeded"):
            self._succeeded = True
            raise ConnectionError("Simulated connection failure")
        return f"Response from {endpoint}"
    
    @validate_types
    def typed_method(self, value: int, name: str) -> str:
        """Method with type validation."""
        return f"{name}: {value}"
    
    @LazyProperty
    def expensive_computation(self) -> int:
        """Property that's computed only when accessed."""
        print("Computing expensive result...")
        time.sleep(0.5)  # Simulate expensive computation
        return 42
    
    cached = CachedDecorator()
    
    @cached
    def cached_method(self, param: str) -> str:
        """Method with result caching."""
        print(f"Computing result for {param}...")
        time.sleep(0.2)  # Simulate computation
        return f"Result for {param}"


# Test the class
if __name__ == "__main__":
    obj = ComplexClass("test")
    
    # Test timing decorator
    obj.process_data({"key": "value"})
    
    # Test retry decorator
    try:
        result = obj.external_api_call("failing")
        print(f"API call result: {result}")
    except Exception as e:
        print(f"API call failed: {e}")
    
    # Test type validation
    try:
        obj.typed_method(42, "test")
        obj.typed_method("wrong", "test")  # Should raise TypeError
    except TypeError as e:
        print(f"Type validation error: {e}")
    
    # Test lazy property
    print("Before accessing expensive_computation")
    value = obj.expensive_computation
    print(f"After first access: {value}")
    print(f"Second access (should be cached): {obj.expensive_computation}")
    
    # Test cached decorator
    print(obj.cached_method("param1"))
    print(obj.cached_method("param1"))  # Should use cached result
    print(obj.cached_method("param2"))  # Should compute new result