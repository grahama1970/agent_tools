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

A tool for AI agents to verify that suggested code only uses real, existing methods.

## Purpose

As an AI assistant, I use method_validator to:
1. Verify my code suggestions don't use non-existent methods
2. Confirm method signatures and parameters after writing code
3. Prevent method hallucination by validating after writing
4. Ground my suggestions in actual package functionality

## Example Usage (For AI Agents)

WRONG Workflow ❌:
```python
# DON'T check methods before writing code
method-validator litellm --method completion
# Then write code...
```

RIGHT Workflow ✅:
1. First write the code you think should work:
```python
def chat_with_llm(prompt: str) -> str:
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

2. Then validate ALL methods you used:
```bash
# Verify each method exists
method-validator litellm --method completion --quick
```

3. Fix any methods that don't exist or were hallucinated

This ensures AI agents only suggest code using real methods.

## Success Criteria

1. Write code first based on training
2. Validate ALL methods used in the code
3. Fix any non-existent methods
4. Only suggest code using verified methods

## Implementation Notes

1. Always validate AFTER writing code
2. Check ALL methods used
3. Use --quick mode for fast validation
4. Fix or remove any non-existent methods
5. Update code based on validation results

## Features

- Quick validation of method existence and accessibility
- Deep analysis of method signatures, parameters, and documentation
- Caching of analysis results for improved performance
- Support for nested methods (e.g. `_Logger.add`)

## Usage

### Quick Validation

To quickly check if a method exists and is accessible:

```bash
method-validator package_name --method method_name --quick
```

This performs a fast check without deep analysis, ideal for:
- Verifying method existence
- Basic validation during development
- CI/CD pipelines

Example:
```bash
method-validator loguru --method _Logger.add --quick
```

### Deep Analysis

For detailed method analysis:

```bash
method-validator package_name --method method_name
```

This provides:
- Method signature
- Parameter details
- Documentation
- Exception information

Example:
```bash
method-validator loguru --method _Logger.add
```

### List All Methods

To list all methods in a package:

```bash
method-validator package_name --list-all
```

## Cache Management

Analysis results are cached to improve performance. The cache:
- Stores results in SQLite database
- Automatically cleans up old entries
- Validates against source code changes

## Exit Codes

- 0: Success
- 1: Method not found or error

## Example Test Conversation

User: "I want to use litellm to send a chat message"