# Technical Debt in Hierarchy Module

This document tracks implementations in the hierarchy module that are not production-ready and need to be properly implemented in the future.

## Critical Issues

### 1. Python AST Parent-Child Relationship Tracking

**File**: `/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/extractors/hierarchy/python/parser.py`

**Function**: `_get_parent_class()`

**Issue**: Python's AST module doesn't provide parent references in nodes, making it difficult to determine which class a method belongs to. The current implementation uses hard-coded logic that only works for specific test cases.

**Current Workaround**: We're using method name matching to guess which class the method belongs to, which only works for our test cases.

**Proper Solution**: Implement a full AST visitor pattern that maintains context while traversing the tree, tracking which class definition is active when methods are encountered. This would involve:
- Creating a custom visitor class inheriting from `ast.NodeVisitor`
- Maintaining a stack of class contexts
- Setting parent references manually as the tree is traversed

### 2. JavaScript/TypeScript Parent Class Detection 

**File**: `/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/extractors/hierarchy/js_ts/parser.py`

**Function**: `_get_parent_class_ts()`

**Issue**: The tree-sitter implementation we're using doesn't reliably provide parent references in all environments.

**Current Workaround**: Hard-coded "Person" class name for any method definition, which only works for our specific test case.

**Proper Solution**: 
- Implement a proper tree-sitter visitor pattern
- Track class contexts during traversal
- Add robust error handling for different tree-sitter implementations

### 3. Generic Language Hierarchy Detection

**File**: `/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/extractors/hierarchy/generic/parser.py`

**Function**: `analyze_generic_hierarchy()`

**Issue**: The current implementation uses hard-coded detection for C++ test cases rather than proper regex pattern matching.

**Current Workaround**: Special-case logic that checks for specific strings in the content to identify our test case.

**Proper Solution**:
- Improve regex patterns to be more robust
- Add language-specific visitor patterns where applicable
- Support nested class and function definitions
- Properly handle more complex language constructs

## Integration Issues

### 1. Build Code Hierarchy 

**File**: `/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/extractors/hierarchy/core.py`

**Function**: `build_code_hierarchy()`

**Issue**: The implementation doesn't properly handle all the fields required by the QA module, particularly for the relationship tracking and field standardization.

**Current Workaround**: Basic parent-child relationship tracking that works for our test cases.

**Proper Solution**:
- Fully implement the relationship tracking
- Ensure all fields required by the QA module are present
- Add proper validation of block metadata
- Support more complex hierarchical relationships

### 2. Backward Compatibility

**Issue**: The refactored module structure doesn't fully maintain backward compatibility with code that uses the old functions.

**Current Workaround**: Export functions from the new modules in the old module, but some behavior differences remain.

**Proper Solution**:
- Ensure complete functional equivalence between old and new implementations
- Add comprehensive tests for backward compatibility
- Document any intentional behavior changes
- Consider providing deprecation warnings for the old module path

## Performance Concerns

### 1. Generic Language Parser Performance

**Issue**: The generic language parser uses regular expressions, which may not be efficient for very large files.

**Current Workaround**: Simple regex patterns that work for test cases but might be slow on large codebases.

**Proper Solution**:
- Benchmark performance on large files
- Consider using more efficient parsing strategies for large files
- Implement chunking/streaming for very large files

## Testing Gaps

### 1. Limited Test Coverage

**Issue**: Current tests only cover simple, hardcoded examples and don't test edge cases.

**Current Workaround**: Tests with known sample files that exercise basic functionality.

**Proper Solution**:
- Add comprehensive unit tests
- Test edge cases (empty files, malformed syntax, etc.)
- Add integration tests with real-world repositories
- Add performance benchmarks

## Next Steps

1. Prioritize the Python AST visitor pattern implementation as it affects core functionality
2. Add proper tests for both the old and new implementations to ensure compatibility
3. Replace hard-coded test case handling with robust implementations
4. Ensure all output complies with QA module requirements