"""Example of how AI agents should validate methods before using them.

This example demonstrates the workflow that AI agents should follow when generating code:
1. Write code that uses third-party methods
2. Before presenting the code to users, validate that the methods exist
3. Only proceed if validation succeeds
4. If validation fails, revise the code to use valid methods

This helps prevent method hallucination and ensures correct API usage.
"""

from agent_tools.method_validator.analyzer import validate_method

def example_workflow() -> None:
    """Example workflow for AI agents to validate methods."""
    
    # After writing code that uses requests.get and response.json()
    # Validate the methods exist before presenting to user
    
    # Validate that methods exist and are callable
    validation_results = [
        validate_method("requests", "get"),
        validate_method("requests.models", "Response.json")  # Fixed import path
    ]
    
    # Print validation results
    for is_valid, message in validation_results:
        print(message)
        
    # Check if any validations failed
    if any(not is_valid for is_valid, _ in validation_results):
        print("\nSome methods failed validation!")
        print("AI should revise the code to use valid methods")
    else:
        print("\nAll methods validated successfully!")
        print("AI can proceed with presenting the code to user")

if __name__ == "__main__":
    example_workflow() 