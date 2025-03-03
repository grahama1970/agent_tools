"""Unit tests for method validation functionality."""

import pytest
from agent_tools.method_validator.analyzer import validate_method

def test_validate_existing_method() -> None:
    """Test validation of an existing method."""
    # Test with a known existing method
    is_valid, message = validate_method("json", "dumps")
    assert is_valid
    assert "exists and is callable" in message

def test_validate_nonexistent_method() -> None:
    """Test validation of a non-existent method."""
    # Test with a method that doesn't exist
    is_valid, message = validate_method("json", "nonexistent_method")
    assert not is_valid
    assert "not found" in message.lower()

def test_validate_builtin_method() -> None:
    """Test validation of a builtin method."""
    # Test with a builtin method from a module instead of a type
    is_valid, message = validate_method("builtins", "len")
    assert is_valid
    assert "exists and is callable" in message

def test_validate_invalid_package() -> None:
    """Test validation with an invalid package."""
    # Test with a package that doesn't exist
    is_valid, message = validate_method("nonexistent_package", "method")
    assert not is_valid
    assert "could not import package" in message.lower() 