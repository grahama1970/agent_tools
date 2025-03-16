"""
Exception classes for the cursor_rules package.

This module contains custom exception classes used throughout the cursor_rules package.
"""


class DatabaseConnectionError(Exception):
    """Exception raised when there is an error connecting to the database."""
    
    def __init__(self, message="Error connecting to the database", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(Exception):
    """Exception raised when there is an error in the configuration."""
    
    def __init__(self, message="Error in configuration", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class SchemaError(Exception):
    """Exception raised when there is an error in the schema."""
    
    def __init__(self, message="Error in schema", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class CursorRulesError(Exception):
    """Base exception for cursor_rules-specific errors."""
    
    def __init__(self, message="Error in cursor_rules", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message) 