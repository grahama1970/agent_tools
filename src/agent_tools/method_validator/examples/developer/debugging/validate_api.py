"""Example of using Method Validator for debugging API usage.

This example demonstrates how developers can use the Method Validator tool
to debug API usage issues and explore package functionality.

Features shown:
1. Deep method analysis
2. Parameter inspection
3. Exception handling patterns
4. Usage examples extraction
"""

from agent_tools.method_validator.analyzer import MethodAnalyzer

def explore_api():
    """Example of exploring and debugging an API."""
    analyzer = MethodAnalyzer()
    
    # Deep analysis of a specific method
    print("Analyzing requests.get()...")
    details = analyzer.deep_analyze("requests", "get")
    
    if details:
        # 1. Check Parameters
        print("\nRequired Parameters:")
        for name, info in details['parameters'].items():
            if info['required']:
                print(f"- {name}: {info['description']}")
                
        # 2. Look for Examples
        if details['examples']:
            print("\nUsage Examples:")
            for example in details['examples']:
                print(f"\n{example}")
                
        # 3. Check Exception Patterns
        if details['exceptions']:
            print("\nPossible Exceptions:")
            for exc in details['exceptions']:
                print(f"- {exc['type']}: {exc['description']}")
                
        # 4. Return Type Information
        if details['return_info']['type']:
            print(f"\nReturn Type: {details['return_info']['type']}")
            if details['return_info']['description']:
                print(f"Description: {details['return_info']['description']}")
    
    # Quick scan for related methods
    print("\nRelated methods in package:")
    methods = analyzer.quick_scan("requests")
    for name, summary, categories in methods:
        if 'read' in categories:  # Show only read-related methods
            print(f"- {name}: {summary}")

if __name__ == "__main__":
    explore_api() 