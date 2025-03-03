# ✅ Method Validator

A tool for AI agents to validate code suggestions against real, existing methods. This tool enables natural code generation while preventing both method hallucination and unnecessary duplication of functionality.

## 🎯 Purpose

The Method Validator helps AI agents write better code by:
- Validating methods after writing code but before showing to users
- Preventing method hallucination without constraining code generation
- Discovering existing functionality to prevent duplication
- Analyzing package capabilities before implementing new solutions
- Providing alternatives when methods don't exist
- Ensuring correct API usage

## 🔄 Workflow

The Method Validator implements a specific workflow designed to maximize natural code generation while ensuring safety:

1. **Write Code Naturally** ✍️
   - Use full training knowledge to write code
   - Generate complete solutions without constraints
   - Leverage understanding of common patterns and APIs

2. **Validate and Discover** 🔍
   - Check all methods exist AFTER writing code
   - Scan packages for existing similar functionality
   - Identify alternative methods if needed
   - Validate BEFORE showing code to users

3. **Smart Recovery** 🔄
   - Suggest alternatives if methods don't exist
   - Identify similar existing functionality
   - Help adapt code to use existing APIs
   - Prevent reinventing existing solutions

## 📝 Example Usage

```python
# 1. First, write code naturally based on training
def process_image(url: str) -> bytes:
    # AI writes this based on its understanding
    response = requests.get(url)
    image = Image.open(response.content)
    processed = image.filter('BLUR')
    return processed.tobytes()

# 2. Before showing to user, validate ALL methods:
method-validator requests --method get --quick
method-validator PIL.Image --method open --quick
method-validator PIL.Image --method filter --quick
method-validator PIL.Image --method tobytes --quick

# 3. Only if ALL validations pass, show code to user
# If any fail, revise using valid methods
```

## 🛠️ Features

- **Quick Validation**: Fast method existence checks
- **Deep Analysis**: Detailed method information when needed
- **Smart Caching**: Efficient validation through caching
- **Flexible Analysis**: Different validation levels based on need
- **Rich Information**: Complete method details available

## 🚀 Usage Guide

### Command Line Interface

```bash
# Quick method validation (auto-executed)
method-validator package --method method_name --quick

# List all methods in package
method-validator package --list-all

# Deep method analysis (when needed)
method-validator package --method method_name
```

### Python API

```python
from agent_tools.method_validator import validate_method
from agent_tools.method_validator.analyzer import MethodAnalyzer

# Quick validation
is_valid, message = validate_method("package_name", "method_name")

# Deep analysis
analyzer = MethodAnalyzer()
details = analyzer.deep_analyze("package_name", "method_name")
```

## 📦 Components

### analyzer.py
- Main class: `MethodAnalyzer`
- Key functions:
  - `validate_method()`: Quick existence check
  - `quick_scan()`: Fast package discovery
  - `deep_analyze()`: Detailed analysis

### cache.py
- Smart caching system
- SQLite-based storage
- Automatic cache management

### cli.py
- Autonomous validation commands
- Multiple analysis modes
- Machine-readable output

### utils.py
- Command execution control
- Timing utilities
- Common helper functions

## 🔍 Validation Levels

1. **Quick Validation** (`--quick`):
   - Method existence
   - Basic callable check
   - Cached results
   
2. **Deep Analysis**:
   - Full signature
   - Parameter details
   - Return types
   - Exceptions
   - Usage examples

## 🚫 Common Pitfalls

1. ❌ Validating before writing code
   - Constrains natural code generation
   - May miss better solutions
   
2. ❌ Skipping validation before showing code
   - Risk of method hallucination
   - Poor user experience
   
3. ❌ Over-validation
   - Checking standard library methods
   - Validating obvious methods

## ✅ Best Practices

1. Write code naturally using training
2. Validate before showing to user
3. Use quick validation first
4. Deep analyze only when needed
5. Cache results for performance

## 🔧 Performance Tips

1. Use `--quick` for existence checks
2. Leverage the caching system
3. Batch validate when possible
4. Use deep analysis sparingly