# Lessons Learned: Fetch Page Project

## What Went Wrong in the First Implementation

### 1. Overengineering
- The codebase became overly complex and difficult to maintain
- Too many abstraction layers and specialized components
- Excessive type validation and error handling before core functionality was solid

### 2. Test-Driven Development Misapplication
- Started with component tests rather than real-world usage tests
- Created artificial boundaries based on test categories
- Tests drove design decisions rather than user needs
- Failed to prioritize end-to-end working functionality

### 3. Lack of Iterative Development
- Tried to implement all requirements simultaneously
- Built complex models and validation before proving core functionality
- No clear MVP phase identified before adding advanced features
- Failed to ensure a working system at each development step

### 4. Excessive Specification
- The original task.md was too detailed and prescriptive
- Too many specific requirements without prioritization
- Encouraged implementation of edge cases before core functionality
- Focused on theoretical completeness over practical usability

### 5. Architectural Issues
- Functionality was scattered across many small modules
- No clear separation of concerns
- Database coupling throughout the codebase
- Complex error handling that was difficult to maintain

### 6. Insufficient Use of Documentation and Examples
- Attempted to implement features without first understanding established patterns
- Deviated from standard library usage patterns in documentation
- Failed to begin with working, minimal examples from official documentation
- Didn't create tests that first validate understanding of external libraries
- Reinvented existing solutions rather than using proven approaches
- **CRITICAL ERROR: Proceeded with implementation without consulting official documentation**
- **CRITICAL ERROR: Failed to include documentation links in code and tests**
- **CRITICAL ERROR: Relied on assumptions rather than documented behavior**

## Documentation as the Foundation of Development

### 1. Documentation Links are Non-Negotiable
- Official documentation links are MANDATORY for every package used
- No documentation = No implementation
- Documentation links must be included at the top of every test and implementation file
- Documentation links serve as the source of truth for expected behavior
- Documentation links should be versioned to match the package version being used
- Examples:
  - BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
  - Bleach: https://bleach.readthedocs.io/
  - urllib.parse: https://docs.python.org/3/library/urllib.parse.html
  - Click: https://click.palletsprojects.com/
  - aiohttp: https://docs.aiohttp.org/
  - pytest: https://docs.pytest.org/

### 2. Documentation Reading is the First Step
- Reading documentation is not optional - it is the required first step
- The entire relevant section of documentation must be read before writing any code
- Do not rely on prior knowledge, model training, or assumptions
- Understand the package's design philosophy and intended usage patterns
- If documentation is unclear, create clarifying tests to validate understanding

### 3. Documentation-Driven Development
- Start with the documentation, not the code
- Implement exactly what is documented, not what you think should work
- When documentation is incomplete, contribute back with clarifying examples
- Document your understanding of the package in tests and comments
- Reference specific sections of documentation for specific functionality

### 4. Documentation Verification
- Verify that your implementation matches the documented behavior
- When behavior seems unexpected, consult the documentation again
- Create tests that validate your understanding of the documentation
- Document any discrepancies between documented and actual behavior
- Update documentation references when package versions change

## Leveraging Validation Tools to Prevent Method Hallucination

### 1. The Problem of Method Hallucination
- AI models may "hallucinate" methods that don't exist in packages
- Even experienced developers may misremember API interfaces
- Method signatures and behavior can change between package versions
- Package names may be similar but have different interfaces
- Assumptions about package behavior can lead to subtle bugs

### 2. Using method_validator to Verify Methods
- The `method_validator` tool can verify that methods actually exist in packages
- It can check method signatures to ensure parameters are correct
- Use it when uncertain about whether a method exists or its signature
- Particularly useful for less common packages or recent updates

### 3. When to Use Method Validation
- When implementing functionality with unfamiliar libraries
- When updating code to use newer versions of packages
- When troubleshooting unexpected behavior
- When working with packages that have similar names but different interfaces
- Before spending time debugging code that might use non-existent methods

### 4. Method Validation Complements Documentation
- Documentation is still the primary source of truth
- Method validation helps verify understanding of documentation
- Use method validation to confirm that your understanding is correct
- It's a quick sanity check before implementation or debugging
- Particularly valuable when documentation is sparse or unclear

## Database Integration with ArangoDB

### 1. ArangoDB Driver is Synchronous Only
- The `python-arango` driver is primarily a synchronous library
- It does not provide native asynchronous API methods
- All database operations will block the event loop if called directly in async code
- Any async wrapper must handle this blocking nature appropriately

### 2. Integrating ArangoDB with Async Functions
- Always use `asyncio.to_thread()` when calling ArangoDB operations from async code
- This runs the synchronous operation in a separate thread, preventing blocking
- **CRITICAL: You MUST await the result of asyncio.to_thread() - it returns a coroutine**
- Example:
  ```python
  # Correct pattern - ALWAYS use this exact pattern:
  db = await asyncio.to_thread(get_db, db_url, db_name)
  result = await asyncio.to_thread(lambda: collection.get(key))
  
  # Using lambda for method calls with complex parameters:
  insert_result = await asyncio.to_thread(lambda: collection.insert(document, overwrite=True))
  
  # For complex operations with multiple steps, create a dedicated synchronous function:
  def sync_perform_operation(db, data):
      # Do synchronous database operations here
      return result
  
  # Then call it with asyncio.to_thread:
  result = await asyncio.to_thread(sync_perform_operation, db, data)
  ```

### 3. Common Pitfalls
- **Forgetting to await**: The `asyncio.to_thread()` function returns a coroutine that must be awaited
  ```python
  # WRONG - Will cause "coroutine was never awaited" warning and won't execute
  asyncio.to_thread(db.collection("documents").get, key)
  
  # CORRECT - Always await the result 
  result = await asyncio.to_thread(lambda: db.collection("documents").get(key))
  ```

- **Variable Reassignment**: Be careful not to reassign parameter variables with operation results
  ```python
  # BAD: Reassigning the 'result' parameter with insert results
  async def store_page_content(db, result):
      # ...code...
      result = await asyncio.to_thread(lambda: collection.insert(doc))  # Now 'result' is a dict, not the original object!
      # ...later code tries to access result.url...  # This will fail!
  
  # GOOD: Use a different variable name
  async def store_page_content(db, extraction_result):
      # ...code...
      insert_result = await asyncio.to_thread(lambda: collection.insert(doc))  # Use a different variable name
      # ...later code can still access extraction_result.url...
  ```

- **Declaring Functions as Async**: Don't declare ArangoDB functions as async without proper handling
  ```python
  # BAD: This appears async but the driver is synchronous
  async def get_document(db, key):
      return db.collection('docs').get(key)  # This blocks the event loop!
  
  # GOOD: Use asyncio.to_thread
  async def get_document(db, key):
      return await asyncio.to_thread(lambda: db.collection('docs').get(key))
  ```

- **Inconsistent Async Pattern**: Ensure all database operations follow the same pattern
  ```python
  # BAD: Inconsistent handling of database operations
  async def process_document(db, doc_id):
      # First operation uses to_thread correctly
      doc = await asyncio.to_thread(lambda: db.collection('docs').get(doc_id))
      
      # Second operation blocks the event loop!
      db.collection('docs').update(doc_id, {'processed': True})  
      
      # Third operation is awaited but needs lambda for complex args
      await asyncio.to_thread(db.collection('logs').insert, {'doc_id': doc_id})  # May not work as expected
  
  # GOOD: Consistent pattern for all operations
  async def process_document(db, doc_id):
      # All operations use to_thread with lambda for complex args
      doc = await asyncio.to_thread(lambda: db.collection('docs').get(doc_id))
      await asyncio.to_thread(lambda: db.collection('docs').update(doc_id, {'processed': True}))
      await asyncio.to_thread(lambda: db.collection('logs').insert({'doc_id': doc_id}))
  ```

- **Assuming Native Async Support**: Don't assume the driver has built-in async functionality
- **Direct Model Passing**: ArangoDB expects dictionaries, not Pydantic models directly

### 4. Complete Async Wrapper Pattern
For maximum clarity, here's a complete example of the correct pattern:

```python
# Synchronous database function (no async here)
def get_document_sync(db, doc_id):
    """Synchronous function to get a document"""
    try:
        return db.collection('documents').get(doc_id)
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise

# Async wrapper that correctly uses asyncio.to_thread
async def get_document_async(db, doc_id):
    """Async wrapper around the synchronous database function"""
    try:
        # ALWAYS await the result of asyncio.to_thread
        return await asyncio.to_thread(get_document_sync, db, doc_id)
    except Exception as e:
        logger.error(f"Error in async document retrieval: {e}")
        raise

# Usage in an async context
async def process_document(doc_id):
    db = await asyncio.to_thread(get_db, db_url, db_name)
    document = await get_document_async(db, doc_id)
    # Process document...
```

### 5. Best Practices
- Create clear separation between database operations and application logic
- Use dependency injection for easier testing
- Document all database operations with proper type hints
- Create specific synchronous and asynchronous versions of database functions
- Handle transactions carefully to avoid blocking issues
- Properly test both synchronous and asynchronous code paths
- Use real ArangoDB instances for integration tests when possible
- **ALWAYS AWAIT asyncio.to_thread() calls - they return coroutines**
- **Use unique, descriptive variable names for operation results**
- **Never reassign function parameters with operation results**

## Testing Async Database Operations with pytest-asyncio

### 1. Fixture Scope Management
- **CRITICAL**: Match fixture scopes carefully when using pytest-asyncio
- The default `event_loop` fixture is function-scoped
- Database fixtures should match the event_loop scope
- Example of correct fixture setup:
  ```python
  @pytest.fixture(scope="function")
  def event_loop():
      """Create an instance of the default event loop for each test case."""
      loop = asyncio.get_event_loop_policy().new_event_loop()
      yield loop
      loop.close()

  @pytest.fixture(scope="function")
  async def db():
      """Database fixture with matching scope to event_loop."""
      db = await asyncio.to_thread(get_db_connection)
      yield db
      await asyncio.to_thread(lambda: db.close())
  ```

### 2. Common pytest-asyncio Patterns
- Always use `@pytest.mark.asyncio` for async test functions
- Keep database operations in fixtures when possible
- Use separate fixtures for different collections
- Example:
  ```python
  @pytest.mark.asyncio
  async def test_document_operations(db, collection):
      # Test async database operations
      doc = {"_key": "test1", "value": "test"}
      result = await asyncio.to_thread(lambda: collection.insert(doc))
      assert result["_key"] == "test1"
  ```

### 3. Testing Best Practices
- Create test databases/collections in fixtures
- Clean up test data after each test
- Use dependency injection for database connections
- Handle both sync and async operations appropriately
- Document the expected behavior in test docstrings
- Reference official documentation in test files

### 4. Avoiding Common Test Pitfalls
- Don't mix sync and async operations without proper handling
- Don't use module-scoped fixtures with function-scoped event loops
- Don't forget to clean up test data and connections
- Don't assume database operations are async-native
- Always use `asyncio.to_thread` for database calls

### 5. Documentation References
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- python-arango: https://python-arango.readthedocs.io/
- arangodb AQL query reference: https://docs.arangodb.com/stable/aql/
- asyncio: https://docs.python.org/3/library/asyncio.html

## Script Initialization and Testing

### 1. Every Script Needs a Usage Example
- **CRITICAL**: Every script must include a simple usage function and an `if __name__ == "__main__"` block
- This allows for initial testing before formal tests are written
- Provides a clear example of how the script should be used
- Serves as documentation for new developers
- Example:
  ```python
  def demo_format_dataset():
      """Demonstrate the dataset formatting functionality with examples."""
      # Simple demonstration code here
      
  if __name__ == "__main__":
      # Run the demonstration when the module is executed directly
      demo_format_dataset()
      
      # Process command line arguments if provided
      if len(sys.argv) > 1:
          # Handle command line arguments
  ```

### 2. Benefits of Main Block Implementation
- Allows for quick verification of functionality
- Provides a starting point for debugging
- Demonstrates the intended usage pattern
- Creates a working MVP at every stage of development
- Supports the "Focus on Real-World Usage First" principle
- Ensures the script can be run directly for testing

### 3. Integration with Documentation-First Testing
- The usage example should reference official documentation
- It should demonstrate the core functionality in a minimal way
- The example should be updated as features are added
- This approach supports iterative development with working MVPs
- Aligns with the principle of ensuring a working system at each development step

## Better Approaches for the Simplified Version

### 1. Start with a Minimal Viable Product (MVP)
- Begin with core functionality: downloading and extracting page content
- Use simple data structures (e.g., dataclasses instead of complex Pydantic models)
- Add complexity only when justified by actual needs
- Ensure a working end-to-end system at every stage of development

### 2. Focus on Real-World Usage First
- Start with real-world tests that demonstrate practical usage
- Build the API based on how users would actually use it
- Create component tests only after establishing working patterns
- Validate with end-to-end user workflows before refining implementation details

### 3. Iterative Development Process with Working MVPs
- Each phase should start with a minimal working implementation
- Incrementally enhance features while maintaining a functioning system
- Test each enhancement with real-world usage before adding the next
- Ensure each phase builds on a solid foundation from the previous phase
- Clearly separate "must have" features from "nice to have" enhancements

### 4. Simplify Architecture
- Create clear layers with well-defined responsibilities
- Separate core logic from database and external integrations
- Use dependency injection to maintain testability
- Allow natural architectural boundaries to emerge from actual usage

### 5. Practical Testing Approach
- Begin with smoke tests that validate basic functionality
- Progress to integration tests that simulate real usage
- Add component tests for specific edge cases
- Use mocks sparingly and only for external dependencies
- Test the "what" (behavior) rather than the "how" (implementation)

### 6. Begin with Documentation and Examples
- Start by implementing a working example from official documentation
- Create "documentation tests" that demonstrate correct library usage before integration
- Use these examples as the foundation for further development
- Consult examples for each new library or component before implementation
- Maintain references to documentation sources in comments and tests

## New Insights from Documentation-First Testing

### 1. Benefits of Documentation Tests
- Documentation tests serve as a learning tool for developers to understand libraries before implementation
- They catch misunderstandings about library behavior early in the development process
- Tests reveal edge cases and limitations documented in library documentation but easy to miss
- They provide a reference point for how libraries should be used correctly
- Documentation tests create a foundation of understanding before building more complex functionality

### 2. Practical Application of Documentation Tests
- For Bleach HTML cleaning, tests revealed that the `styles` parameter doesn't exist as we initially thought
- For URL joining, tests helped understand edge cases like protocol-relative URLs and parent directory navigation
- For image deduplication, tests clarified different strategies and their trade-offs
- Tests serve as executable specifications that validate our understanding
- They create a shared knowledge base for the team about library usage patterns

### 3. Integration with Iterative Development
- Documentation tests fit perfectly with the MVP approach by establishing baseline functionality
- They help define the boundaries and capabilities of each component before integration
- Tests provide confidence when refactoring or enhancing existing functionality
- They reduce debugging time by catching misunderstandings early
- Documentation tests make the development process more predictable by validating assumptions upfront

### 4. Integration with Real-World Testing
- Documentation tests alone are not sufficient - they must be followed by real-world integration tests
- After implementing each feature, run non-mocked tests with actual web pages to verify integration
- Ensure that new components work correctly within the existing MVP architecture
- Detect integration issues early rather than accumulating technical debt
- Maintain a working end-to-end system at all times by validating with real-world examples
- Avoid the trap of having individually tested components that fail to work together
- **NEVER move on to a new task without verifying the current task with real-world tests**
- **Real-world verification is a hard requirement, not an optional step**
- **Integration testing with the MVP is the only way to ensure the feature actually works in practice**
- **REQUIRED FINAL STEP: Run the actual MVP tool with real-world examples and examine the output files to verify feature integration**
- **Passing tests ≠ Working MVP - always verify with the actual tool**

## Package Isolation in Documentation Tests

### 1. The Importance of Package Isolation
- When learning a new package, test it in isolation without other packages that might interfere
- Different packages may modify the same data in ways that obscure the behavior of the package being learned
- HTML sanitizers like Bleach can remove or modify elements that BeautifulSoup would otherwise detect
- Text processing libraries might normalize or alter content before it reaches the target package
- Package interactions can create subtle bugs that are difficult to diagnose

### 2. Correct Approach to Documentation Tests
- Create separate, isolated tests for each package to understand its pure behavior
- Use raw, unmodified inputs when testing a package's core functionality
- Document the expected behavior based solely on the package's documentation
- Only after understanding each package individually should they be integrated
- When integration is necessary, clearly document the order of operations and potential interactions
- **CRITICAL: Always include the official documentation link in tests and references**
- **CRITICAL: Read the official documentation thoroughly before writing any tests or implementation code**

### 3. Documentation Links as First-Class Citizens
- Documentation links are not optional - they are required for every package being tested
- The official documentation link must be included at the top of every test file (e.g., https://www.crummy.com/software/BeautifulSoup/bs4/doc/ for BeautifulSoup)
- Documentation links should also be included in all reference materials and code comments
- If a package lacks proper documentation, consider it a red flag and evaluate alternatives
- Documentation links serve as the source of truth for expected behavior
- When documentation is unclear or incomplete, create clarifying tests to validate assumptions
- Never proceed with implementation without first consulting the official documentation
- Documentation links should be versioned to

## Code Extractor Testing Lessons

### 1. Test Against Actual Implementation, Not Specifications

- **Verify Function Existence**: Before writing complex tests, verify that all functions actually exist and are importable
- **Check Function Signatures**: Directly examine function signatures in the source code before writing tests
- **Read Function Implementations**: Understand implementation details before designing tests
- Example:
  ```python
  # BAD: Writing tests assuming function signature without checking
  def test_extract_function(file_path, output_dir, stats):
      result = extract_function(file_path, output_dir, stats)
      assert result > 0
  
  # GOOD: Check actual signature first
  # From source: def extract_function(file_path, content, output_dir, stats):
  def test_extract_function(file_path, temp_dir):
      with open(file_path, "r") as f:
          content = f.read()
      output_dir = os.path.join(temp_dir, "output")
      stats = {"blocks": 0, "errors": []}
      result = extract_function(file_path, content, output_dir, stats)
      assert result > 0
  ```

### 2. Always Create Direct Verification Tests

- **Single-Function Verification**: Create standalone scripts that test just one function
- **Outside Test Framework**: Test critical functions outside the main test framework first
- **Minimal Dependencies**: Minimize dependencies to isolate the exact functionality being tested
- **Clear Output**: Design tests to print clear, unambiguous results
- Example:
  ```python
  #!/usr/bin/env python3
  """Test a single function directly."""
  import sys
  from pathlib import Path
  
  # Add src to path
  sys.path.insert(0, str(Path(__file__).parent / "src"))
  
  from my_package.module import critical_function
  
  # Test with basic inputs
  result = critical_function("input")
  print(f"Result: {result}")
  print(f"Expected: 'processed input'")
  print(f"Passed: {result == 'processed input'}")
  ```

### 3. Start with the Simplest Working Example

- **Simplify Test Cases**: Begin with the absolute minimum test case that should work
- **Example-Driven Testing**: Base initial tests on examples from documentation
- **Baseline First**: Establish a working baseline before testing edge cases
- **Real File Tests**: When testing file operations, start with real files not mocks
- Example:
  ```python
  def test_basic_language_detection():
      """Test the most basic language detection case first."""
      assert detect_language(".py") == "python"
      
  def test_complex_language_detection():
      """Only test complex cases after basic case works."""
      # More complex cases...
  ```

### 4. Python Path Management Principles

- **Explicit Path Setup**: Always set up Python path explicitly in conftest.py or test scripts
- **Print Path Debugging**: Include path debugging to diagnose import issues
- **Package Structure Awareness**: Ensure package structure supports imports correctly
- **Module-Level Initialization**: Use __init__.py files appropriately
- Example conftest.py:
  ```python
  import sys
  from pathlib import Path
  
  # Add project root and src to Python path
  project_root = Path(__file__).parent.parent
  sys.path.insert(0, str(project_root))
  sys.path.insert(0, str(project_root / "src"))
  
  # Print configuration for debugging
  print(f"Python path: {sys.path}")
  ```

### 5. Type Consistency and Validation

- **Path vs. String Objects**: Be consistent with Path objects vs. strings
- **Type Hints as Documentation**: Use type hints to document expected types
- **Runtime Type Checking**: Validate input types at runtime for critical functions
- **Type-Aware Testing**: Write tests with awareness of expected types
- Example:
  ```python
  from pathlib import Path
  
  # Function with clear type hints
  def process_file(file_path: Path, output_dir: Path) -> int:
      """Process a file and return number of items processed."""
      # Type validation at runtime
      if not isinstance(file_path, Path):
          file_path = Path(file_path)  # Coerce to Path if string
      
      # Processing logic...
      return items_processed
  
  # Test with correct types
  def test_process_file(temp_dir):
      input_file = Path(temp_dir) / "input.txt"
      output_dir = Path(temp_dir) / "output"
      # Correct usage with Path objects
      result = process_file(input_file, output_dir)
  ```

### 6. Progressive Test Execution

- **Start Small**: Begin testing with the simplest component
- **Build Up**: Gradually add complexity only after basics work
- **Isolate Issues**: Run specific tests to pinpoint problems
- **Document Command Patterns**: Record working pytest commands for reference
- Example test progression:
  ```
  # 1. Test utility functions directly
  python -m my_package.utils.test_util
  
  # 2. Test specific component with pytest
  pytest tests/unit/test_component.py::test_basic_functionality -v
  
  # 3. Test full workflow with real files
  pytest tests/integration/test_workflow.py
  ```

### 7. Avoid Module-Level Test Skipping

- **Function-Level Skips**: Use conditional skips at the function level, not module level
- **Informative Skip Messages**: Include detailed messages explaining why tests are skipped
- **Optional Dependencies**: Handle optional dependencies with clear error messages
- **Alternative Test Paths**: Provide alternative test paths when dependencies are missing
- Example:
  ```python
  @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, 
                      reason="Tree-sitter is not available")
  def test_with_tree_sitter():
      """Test functionality that requires tree-sitter."""
      # Test logic...
  
  def test_alternative_approach():
      """Test alternative functionality that doesn't require tree-sitter."""
      # Alternative test logic...
  ```
