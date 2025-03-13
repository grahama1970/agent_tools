# .cursor Directory Guide

This directory contains essential resources for development, testing, and troubleshooting.

## Directory Structure

```
.cursor/
├── bin/              # Development scripts and tools
├── troubleshoot/     # Troubleshooting guides and solutions
├── rules/           # Project-specific rules and patterns
└── examples/        # Example code and usage patterns
```

## Quick Reference

### Troubleshooting Guides
- [Testing Guidelines](troubleshoot/testing-guidelines.md)
  - Mocking best practices
  - Test categories and organization
  - Common pitfalls and solutions
  - Best practices for test writing
  
- [Environment Setup](troubleshoot/environment-setup.md)
  - Virtual environment configuration
  - Required packages and versions
  - Environment variables
  
- [Test Categories](troubleshoot/test-categories.md)
  - Component vs Integration tests
  - Test organization and naming
  - Test dependencies
  
- [Pytest Debugging](troubleshoot/pytest-debugging.md)
  - Common pytest issues
  - Debugging strategies
  - Test isolation techniques

### Development Rules
- [Package Usage](rules/003-package-usage.mdc)
  - Text processing utilities
  - Caching and retries
  - LLM integration
  - Embedding utilities
  
- [Design Patterns](rules/002-design-patterns.mdc)
  - Common code patterns
  - Implementation templates
  
- [Testing Practices](rules/004-testing-practices.mdc)
  - Core testing rules
  - Test organization
  - Validation requirements

### Example Code
- Basic usage examples
- Advanced integration patterns
- Error handling examples
- Performance optimization examples

## When to Use What

1. **New to the Project?**
   - Start with `examples/` for quick understanding
   - Review `rules/` for project standards
   - Check `troubleshoot/environment-setup.md`

2. **Writing Tests?**
   - Consult `troubleshoot/testing-guidelines.md`
   - Review `rules/004-testing-practices.mdc`
   - Look at example tests in `examples/`

3. **Having Issues?**
   - Check relevant guide in `troubleshoot/`
   - Review error patterns in `examples/`
   - Consult specific rules in `rules/`

4. **Adding Features?**
   - Review design patterns in `rules/`
   - Check similar examples in `examples/`
   - Follow guidelines in relevant docs

## Best Practices

1. **Documentation**
   - Keep guides up to date
   - Add new examples for complex features
   - Document common issues and solutions

2. **Organization**
   - Follow directory structure
   - Use consistent file naming
   - Maintain clear categories

3. **Maintenance**
   - Regular review of guidelines
   - Update examples with new patterns
   - Archive outdated information 