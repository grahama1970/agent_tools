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

## Testing

This project uses pytest for testing. The test suite includes both unit tests and integration tests.

### Running Tests

To run the test suite:

```bash
pytest tests/
```

For test coverage report:

```bash
pytest --cov=agent_tools tests/
```

### Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for end-to-end functionality

### Writing Tests

When contributing to this project:

1. All code changes must include corresponding tests
2. Follow the AAA pattern (Arrange-Act-Assert)
3. Use pytest fixtures for common setup
4. Mock external dependencies appropriately
5. Ensure both new and existing tests pass

### Test Coverage Requirements

- Unit tests for all new functions/methods
- Integration tests for feature changes
- Edge case coverage
- Minimum 80% coverage for new code

### Continuous Integration

Tests are automatically run on:
- Every pull request
- Every merge to main branch
- Every release tag

# Cursor Patterns

A centralized repository of Cursor MDC rules and design patterns for consistent code generation across projects.

## Structure

```
cursor-patterns/
├── rules/              # All MDC rules
│   ├── core/          # Core rules (code advice, design patterns index)
│   ├── design_patterns/  # Specific design pattern implementations
│   └── project_specific/ # Language-specific patterns
├── scripts/           # Installation and update scripts
└── README.md
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cursor-patterns
```

2. Run the installation script:
```bash
cd cursor-patterns
chmod +x scripts/install.sh
./scripts/install.sh
```

## Updating

To update your patterns to the latest version:

```bash
chmod +x scripts/update.sh
./scripts/update.sh
```

This will create a backup of your current rules before updating.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Adding New Patterns

1. Create a new `.mdc` file in the appropriate directory
2. Follow the MDC format:
   - Include clear frontmatter (name, version, author, etc.)
   - Add specific glob patterns
   - Define clear triggers
   - Document the pattern thoroughly
3. Update the index file if necessary
4. Test the pattern in a real project
5. Submit a pull request

## License

MIT
