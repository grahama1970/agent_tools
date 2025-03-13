# Method Validator: Agent Evolution Lessons

## Using method_validator as an Agent Rather Than Just for Testing

### 1. Proactive Method Validation

- **AI agents should actively validate methods before suggesting or writing code**
- Run `method-validator validate <package> --method <method_name> --quick` to verify existence
- Use `--no-quick` option to see detailed parameter information and return types
- Never suggest code using methods that haven't been validated
- If a method doesn't exist, suggest alternatives based on what does exist
- Method validation should be part of the AI's thinking process, not just testing
- Integrate validation results directly into the code generation workflow

### 2. Validation-First Development

- Start with method validation before writing code, not after
- Build code around validated methods rather than assuming method availability
- Use validation results to inform parameter choices and error handling
- For complex APIs, validate related methods to understand the overall pattern
- Include validated method signatures in docstrings for future reference
- This approach eliminates an entire class of implementation errors before they occur

### 3. Evolution Path for AI Agents

- **Dynamic environment inspection instead of static training data**
- Real-time verification of method existence and signatures
- Live discovery of package capabilities rather than memorized patterns
- Adaptability to library changes and version differences
- Self-correction based on actual runtime environment
- This represents a fundamental shift from "knew at training time" to "knows right now"

### 4. CLI Command Reference

```bash
# Basic usage - quick check if method exists
method-validator validate <package> --method <method_name> --quick

# Detailed analysis with parameter information
method-validator validate <package> --method <method_name> --no-quick

# Extract methods from a script to validate
method-validator validate --script <path-to-file.py>

# Validate a function in a file
method-validator check-function <path-to-file.py>
```

### 5. Examples from ArangoDB Validation

```bash
# Validate ArangoClient class exists
method-validator validate arango --method ArangoClient --quick

# Validate has_collection method with parameters
method-validator validate arango.database --method StandardDatabase.has_collection --no-quick

# Validate create_collection with full parameter details
method-validator validate arango.database --method StandardDatabase.create_collection --no-quick

# Validate view creation capabilities
method-validator validate arango.database --method StandardDatabase.create_arangosearch_view --no-quick
```

### 6. Integration with Other Tools

- Combined with documentation-first approach for comprehensive validation
- Used alongside real ArangoDB connections for integration testing
- Complementary to traditional unit and integration tests
- Acts as a bridge between AI code generation and runtime environment
- Enables safe exploration of unfamiliar libraries and APIs

### 7. Benefits for AI Evolution

- **Reduces hallucination of non-existent methods**
- Provides accurate, up-to-date API information
- Creates a self-correcting feedback loop during code generation
- Allows confident use of specialized libraries
- Decreases debugging time by catching errors before execution
- Builds a foundation for continuous learning from actual library interfaces
- Represents a step toward fully autonomous, self-validating AI coding

## Best Practices for AI Agents Using method_validator

1. **Always validate before writing code that uses external methods**
2. Examine parameter details for complex or unfamiliar methods
3. Document validation results as part of your thinking process
4. Check multiple related methods to understand library patterns
5. Use validation results to adjust your approach when methods don't exist
6. Consider versioning implications when validation results are surprising
7. **Think of method_validator as extending your capabilities, not just testing your output**
8. Remember that real libraries evolve - trust validation over training data
9. Use detailed validation to discover parameter defaults and requirements
10. Let validation guide your exploration of new libraries 