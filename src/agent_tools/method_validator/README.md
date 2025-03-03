# 🛠️ Cursor Agent Tools

A collection of specialized tools designed to enhance AI agent capabilities within the Cursor IDE. These tools enable more intelligent code analysis, validation, and generation while preventing common AI pitfalls like method hallucination and redundant implementations.

## 🎯 Purpose

This repository contains a growing collection of tools that help AI agents (like myself) work more effectively within Cursor. Each tool is designed to solve specific challenges in AI-assisted development:

- Preventing method hallucination
- Validating API usage
- Discovering existing functionality
- Ensuring correct implementation patterns
- Maintaining code quality and consistency

## 📦 Available Tools

### Method Validator ✅

The Method Validator is a specialized tool that helps AI agents analyze Python packages, discover existing methods, and validate APIs before implementing new solutions. This prevents redundant code creation and ensures correct API usage.

Key features:
- Smart package analysis with filtering of standard libraries
- Detailed method discovery and validation
- Intelligent categorization of methods
- Exception pattern analysis
- Optimized caching system
- Machine-readable output for automated processing

[Learn more about Method Validator](src/agent_tools/method_validator/README.md)

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agent-tools.git
cd agent-tools

# Install in development mode
pip install -e .
```

### Using with Cursor

These tools are designed to be used seamlessly within the Cursor IDE. When working with an AI agent in Cursor, you can trigger specific tools using designated prompts:

```
TOOL: method_validator - Implement [task description]
```

Example:
```
TOOL: method_validator - Write a function to extract tables from a webpage using Playwright
```

## 🏗️ Project Structure

```
agent_tools/
├── src/
│   └── agent_tools/
│       ├── method_validator/       # Method validation tool
│       │   ├── analyzer.py        # Core analysis logic
│       │   ├── cache.py          # Caching system
│       │   ├── cli.py            # Command-line interface
│       │   └── README.md         # Tool-specific documentation
│       └── [future tools...]     # Additional tools will be added here
├── tests/                        # Test suite
├── examples/                     # Usage examples
└── README.md                     # This file
```

## 🎯 Design Philosophy

1. **Prevention Over Correction**
   - Tools focus on preventing common AI mistakes rather than fixing them after the fact
   - Built-in validation and verification at every step

2. **Intelligent Caching**
   - Optimized caching systems to improve response times
   - Smart invalidation based on source changes

3. **Machine-First Design**
   - All tools provide machine-readable output
   - Structured data formats for easy parsing
   - Clear success/failure indicators

4. **Progressive Enhancement**
   - Tools work with basic functionality out of the box
   - Advanced features available for more complex use cases

## 🤝 Contributing

Contributions are welcome! If you have ideas for new tools or improvements to existing ones, please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

Please ensure your contributions maintain or enhance the tools' autonomous operation capabilities.

## 📝 License

[MIT License](LICENSE)

## 🔮 Future Tools

We plan to add more tools to this repository, including:

- Code Pattern Analyzer
- Dependency Graph Generator
- Test Case Validator
- Documentation Analyzer
- Type Inference Helper

Stay tuned for updates!

## 🚀 Next Steps

We are actively developing additional agent tools for various technologies and platforms:

### Database Tools
- **ArangoDB Agent Tools**: Smart graph database operations, query optimization, and schema validation
- **Database Migration Assistant**: Intelligent schema evolution and data transformation
- **Database Schema Creator for LLM consumption**: Intelligent schema evolution and data transformation

### Infrastructure Tools
- **Docker Agent Tools**: Container optimization, security scanning, and deployment validation
- **Infrastructure as Code Validator**: Template verification and best practices enforcement

### Development Tools
- **GitHub Integration Tools**: PR analysis, code review automation, and workflow optimization
- **Local LLM Tools**: Integration with local language models for privacy-sensitive operations

### Additional Planned Tools
- CI/CD Pipeline Validator
- Security Compliance Checker
- Performance Optimization Analyzer
- Cross-Platform Compatibility Validator
- API Integration Assistant
- Cloud Resource Optimizer

Each tool will follow our core design principles of prevention over correction, intelligent caching, and machine-first design while providing specific capabilities for its target technology.

## 🔗 Related Projects

- [Cursor IDE](https://cursor.sh/)
- [LiteLLM](https://github.com/BerriAI/litellm)

## ⚠️ Note for Human Developers

While these tools are primarily designed for AI agents, they can also be valuable for human developers:

- Use Method Validator to explore unfamiliar packages
- Leverage automated API discovery
- Benefit from intelligent caching and analysis

However, the primary focus remains on enhancing AI agent capabilities within Cursor.

# Method Validator

A tool for AI agents to validate code suggestions against real, existing methods. This tool enables natural code generation while preventing method hallucination.

## 🎯 Purpose & Design Philosophy

The Method Validator is designed around a natural code generation workflow:

1. **Write First**: AI agents use their training to write code naturally
2. **Validate Before Showing**: Verify all methods exist before presenting to users
3. **Fix if Needed**: Only modify code if validation fails

This approach:
- ✅ Leverages AI's full knowledge without artificial constraints
- ✅ Maintains natural coding flow
- ✅ Only validates methods actually used
- ✅ Provides context for better alternatives when needed

## 🔄 Workflow Example

```python
# 1. First write code naturally based on training
def chat_with_llm(prompt: str) -> str:
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 2. Validate ALL methods before showing to user
method-validator litellm --method completion --quick

# 3. Only if validation fails, revise code
# Example: If litellm.completion doesn't exist, find alternatives
```

## 🎯 Success Criteria

1. ✅ Natural Code Generation: Write code using full training knowledge
2. ✅ Pre-presentation Validation: Verify before showing to user
3. ✅ Efficient Validation: Only check methods actually used
4. ✅ Smart Recovery: Suggest alternatives if validation fails

## 🛠️ Key Features

- **Quick Validation**: Fast method existence checks
- **Deep Analysis**: Detailed method information when needed
- **Smart Caching**: Efficient validation through caching
- **Flexible Analysis**: Different validation levels based on need
- **Rich Information**: Complete method details available

## 🚀 Usage Guide

### For AI Agents

```bash
# Quick method validation (auto-executed)
method-validator package --method method_name --quick

# List all methods in package
method-validator package --list-all

# Deep method analysis (when needed)
method-validator package --method method_name
```

### For Developers

```python
from agent_tools.method_validator import validate_method
from agent_tools.method_validator.analyzer import MethodAnalyzer

# Quick validation
is_valid, message = validate_method("package_name", "method_name")

# Deep analysis
analyzer = MethodAnalyzer()
details = analyzer.deep_analyze("package_name", "method_name")
```

## 📦 Core Components

### analyzer.py - Core Analysis Engine
- Main class: `MethodAnalyzer`
- Key functions:
  - `validate_method()`: Quick existence check
  - `quick_scan()`: Fast package discovery
  - `deep_analyze()`: Detailed analysis

### cache.py - Performance Optimization
- Smart caching system
- SQLite-based storage
- Automatic cache management

### cli.py - Command Interface
- Autonomous validation commands
- Multiple analysis modes
- Machine-readable output

### utils.py - Helper Functions
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

## 🎯 Integration Guide

### Basic Integration
```python
def validate_code(code: str) -> bool:
    # Extract and validate methods
    methods = extract_methods(code)
    for package, method in methods:
        is_valid, _ = validate_method(package, method)
        if not is_valid:
            return False
    return True
```

### Error Recovery
```python
try:
    is_valid, message = validate_method(package, method)
    if not is_valid:
        # Get alternatives
        alternatives = analyzer.quick_scan(package)
        # Suggest similar methods
except ImportError:
    # Handle missing package
    pass
```

## 🔧 Performance Tips

1. Use `--quick` for existence checks
2. Leverage the caching system
3. Batch validate when possible
4. Use deep analysis sparingly

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

## 🎯 Best Practices

1. ✅ Write code naturally using training
2. ✅ Validate before showing to user
3. ✅ Use quick validation first
4. ✅ Deep analyze only when needed
5. ✅ Cache results for performance