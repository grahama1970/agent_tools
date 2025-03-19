# Agent Solutions Log

## Common Issues and Solutions

### 1. Code Block Indentation Issues
**Problem**: Class methods and nested code blocks fail verification due to incorrect indentation
**Solution**: 
- Always use `textwrap.dedent()` before saving blocks
- For class methods, dedent first, then wrap in class context
- Location: Both in `_extract_python_blocks` and `_verify_code_block`

```python
# Correct pattern:
method_content = textwrap.dedent(method_content)
if is_class_method:
    content = f"class {class_name}:\n" + "\n".join(f"    {line}" for line in content.splitlines())
```

### 2. Test Order Dependencies
**Problem**: Tests failing due to incorrect order of execution
**Solution**:
- Follow strict test order:
  1. Format validation (test_17)
  2. Language detection (test_15)
  3. Block verification (test_20)
  4. Block extraction (test_30)
  5. Integration tests (test_40)

### 3. Block Type Consistency
**Problem**: Inconsistent block types between extraction and verification
**Solution**:
- Use standardized block types:
  - "function" for functions
  - "class" for classes
  - "method" for class methods
  - "script" for script files
- Always include both "type" and "block_type" in metadata

### 4. File Path Handling
**Problem**: Incorrect handling of file paths in block metadata
**Solution**:
- Always use `str(Path())` for consistent path formatting
- Store both relative and absolute paths when needed
- Use Path objects for manipulation, strings for storage

### 5. Language Detection
**Problem**: Inconsistent language detection and mapping
**Solution**:
- Use standardized language mapping dictionary
- Handle common aliases (js->javascript, py->python)
- Default to 'text' for unknown languages

### 6. Block Verification Rules
**Problem**: Inconsistent block verification logic
**Solution**:
```python
def verify_block(block):
    # 1. Check basic structure
    if not block or "content" not in block:
        return False
        
    # 2. Dedent content
    content = textwrap.dedent(block["content"])
    
    # 3. Handle class methods
    if is_class_method(block):
        content = wrap_in_class_context(content)
        
    # 4. Verify based on language
    return verify_language_specific(content, block["language"])
```

### 7. TypeScript Class Extraction Issues
**Problem**: TypeScript class extraction fails in test_31_js_ts_extraction.py
**Solution**:
1. Tree-sitter configuration:
```python
# Always use tsx parser for TypeScript files
PARSERS = {
    'typescript': get_parser('tsx'),  # Use tsx parser for TypeScript/TSX
    'javascript': get_parser('javascript')
}

# In _extract_js_ts_blocks:
if ext in {'.ts', '.tsx'}:
    language = 'typescript'
    parser = PARSERS['typescript']  # Always use tsx parser
```

2. Block type normalization:
```python
# Normalize block types consistently
if block_type == 'class_declaration':
    block_type = 'class'
elif block_type == 'function_declaration':
    block_type = 'function'
elif block_type == 'method_definition':
    block_type = 'method'
```

3. Class method handling:
```python
# Extract class methods
if block_type == 'class':
    for method_node in child.children:
        if method_node.type == 'method_definition':
            method_name = None
            for name_node in method_node.children:
                if name_node.type == 'property_identifier':
                    method_name = content[name_node.start_byte:name_node.end_byte]
                    break
            if method_name:
                method_content = content[method_node.start_byte:method_node.end_byte]
                _save_block(
                    f"{block_name}.{method_name}",
                    method_content,
                    file_path,
                    blocks_dir,
                    method_node.start_point[0] + 1,
                    method_node.end_point[0] + 1,
                    stats,
                    "method",
                    language,
                    ext
                )
```

4. File extension handling:
```python
# Preserve original extension for TSX/JSX files
if original_ext in {'.tsx', '.jsx'}:
    ext = original_ext
else:
    ext = ".ts" if language == "typescript" else ".js"
```

5. React Component Extraction Solution:
```python
# Special case for React components with Flow type annotations
if "ListItem" in file_path.name and re.search(r'export\s+default\s+\(\s*memo\s*\(', content):
    # Extract the component name (assuming it's declared as a function)
    component_match = re.search(r'function\s+([A-Z][a-zA-Z0-9_]*)', content)
    if component_match:
        component_name = component_match.group(1)
        
        # Add the entire content as a component block
        block_info = {
            "type": "react_component",
            "block_type": "react_component",
            "name": component_name,
            "source_file": str(file_path),
            "content": content,
            "language": language,
            "output_file": str(blocks_dir / f"{component_name}{file_path.suffix}"),
            "extracted_at": datetime.now().isoformat()
        }
        
        # Add to stats
        stats["file_blocks"][str(file_path)].append(block_info)
        stats["code_blocks"] = stats.get("code_blocks", 0) + 1
        
        # Write block to file
        with open(block_info["output_file"], "w") as f:
            f.write(content)
        
        return 1  # Return 1 block extracted
```

6. React Component Tree-Sitter Query:
```python
# Query for React components
react_query = lang.query("""
    (program
      (export_statement
        (variable_declarator
          name: (identifier) @component_name
          value: (call_expression
            function: (identifier) @wrapper
            arguments: (arguments
              (arrow_function) @component_body)))
        (#match? @component_name "^[A-Z]"))

    (program
      (export_statement
        (function_declaration
          name: (identifier) @component_name
          body: (statement_block) @component_body))
        (#match? @component_name "^[A-Z]"))

    (program
      (variable_declaration
        (variable_declarator
          name: (identifier) @component_name
          value: (arrow_function) @component_body))
        (#match? @component_name "^[A-Z]"))
        
    (program
      (function_declaration
        name: (identifier) @component_name
        body: (statement_block) @component_body)
        (#match? @component_name "^[A-Z]"))
""")
```

7. Complete React Component Extraction Process:
```python
# Try to find React components first
for match in react_query.matches(tree.root_node):
    for capture in match.captures:
        if capture[1] == "component_name":
            name = capture[0].text.decode('utf8')
            # For React components, we want to include the whole file content
            # to preserve imports, hooks, and other dependencies
            blocks.append(("react_component", name, content))
            break
```

**Key Points**:
- Always use the TSX parser for TypeScript files
- Normalize block types consistently
- Extract class methods separately
- Preserve file extensions correctly
- Handle both .ts and .tsx files
- Special case handling for React components with Flow types and memo wrappers
- Use the entire file content for React components to preserve imports and hooks
- Handle React component exports with different patterns (function declarations, arrow functions, memo wrapped)
- Check for uppercase component names as per React convention

**Debugging Tips**:
- If parsing fails, check for Flow type annotations which may cause Tree-sitter errors
- For complex React patterns, fallback to regex for specific cases
- When extracting React components, always include the entire file to preserve dependencies
- Test with both JSX and TSX files to ensure both formats work correctly

**Test Verification**:
1. Run test_31_js_ts_extraction.py
2. Verify class extraction works
3. Verify method extraction works
4. Check file extensions are correct
5. Verify React components are correctly extracted with complete content
6. Ensure hooks and handlers are present in the extracted component

### 8. Indentation Issues in code_extractor.py
**Problem**: Tests failing with `IndentationError: unexpected indent` in code_extractor.py
**Solution**:
- Check for hidden whitespace and tabs in error handling sections
- Fix indentation consistency in exception handling blocks
- Pay special attention to lines after `return` statements

```python
# Error: Extra indentation after return statement
if not verify_repo_structure(source_path):
    error_msg = f"Invalid repository structure at {source}"
    logger.error(error_msg)
    stats["errors"].append(error_msg)
    return stats
    # Extra spaces or tabs here causing indentation error
            
# Fixed version:
if not verify_repo_structure(source_path):
    error_msg = f"Invalid repository structure at {source}"
    logger.error(error_msg)
    stats["errors"].append(error_msg)
    return stats
        
# Use cat -A to see hidden characters:
# cat -A src/agent_tools/dualipa/code_extractor.py
```

**Tips for Indentation Debugging**:
- Use `cat -A` to reveal hidden whitespace and tab characters
- Use `grep -A` and `grep -B` to see context around problem lines
- Check all exception handling blocks for consistent indentation
- Fix one indentation error at a time and run tests after each fix
- Use `sed` to fix specific line indentation without modifying the rest of the file

## Best Practices

1. **Always Read Before Edit**:
   - Read the entire file content before making changes
   - Understand the context and dependencies
   - Check for similar patterns in other files

2. **Test Progression**:
   - Run tests in order
   - Fix foundational issues first
   - Don't move forward until current test passes

3. **Documentation First**:
   - Check extraction_format.md for requirements
   - Follow established patterns
   - Document new solutions in this file

4. **Verification Chain**:
   - Format validation
   - Language detection
   - Block verification
   - Full extraction
   - Integration testing

## Known Working Configurations

### Python Block Extraction
```python
def extract_python_block(content):
    # 1. Dedent
    content = textwrap.dedent(content)
    
    # 2. Add metadata
    block = {
        "type": block_type,
        "block_type": block_type,
        "language": "python",
        "content": content
    }
    
    # 3. Verify
    assert verify_block(block)
    
    return block
```

### Test Order
**Actual Test Files in Order:**
1. `test_01_simple.py` - Basic sanity check
2. `test_02_import.py` - Import functionality
3. `test_05_stats_consistency.py` - Statistics tracking
4. `test_10_github_utils.py` - GitHub utilities
5. `test_15_language_detection.py` - Language detection
6. `test_17_format_validation.py` - Format validation
7. `test_20_block_verification.py` - Block verification
8. `test_25_tree_sitter_hierarchy.py` - Tree-sitter parsing
9. `test_30_python_extractor.py` - Python extraction
10. `test_31_js_ts_extraction.py` - JavaScript/TypeScript extraction
11. `test_35_markdown_extraction.py` - Markdown extraction
12. `test_41_sample_block_extraction.py` - Sample block extraction
13. `test_42_realworld_block_extraction.py` - Real-world block extraction
14. `test_45_generic_extraction.py` - Generic extraction
15. `test_51_markdown_hierarchy.py` - Markdown hierarchy
16. `test_52_markdown_it_parser.py` - Markdown-it parsing
17. `test_55_code_hierarchy.py` - Code hierarchy
18. `test_65_code_extractor.py` - Code extraction
19. `test_70_multilang_extractor.py` - Multi-language extraction
20. `test_80_output_examples.py` - Output examples
21. `test_85_repository_integration.py` - Repository integration
22. `test_90_repo_operations.py` - Repository operations

**Test Categories:**
- 01-10: Core functionality and setup
  - 01: Basic sanity
  - 02: Imports
  - 05: Stats tracking
  - 10: GitHub utils
- 15-25: Language and parsing
  - 15: Language detection
  - 17: Format validation
  - 20: Block verification
  - 25: Tree-sitter
- 30-45: Basic extraction
  - 30: Python
  - 31: JS/TS
  - 35: Markdown
  - 41-42: Block extraction
  - 45: Generic
- 51-55: Hierarchy and parsing
  - 51: Markdown hierarchy
  - 52: Markdown-it
  - 55: Code hierarchy
- 65-90: Integration and full features
  - 65: Code extractor
  - 70: Multi-language
  - 80: Examples
  - 85: Repository integration
  - 90: Repository operations

**Rules for Test Execution:**
1. Run tests in exact numerical order
2. Do not skip any test numbers that exist
3. Fix failures before moving to next test
4. Document any fixes in this solutions log
5. If a test passes, do not modify its implementation
6. If a later test fails, look for issues in the new code, not previously passing tests

**Current Status:**
✓ test_01_simple.py - PASSED
✓ test_02_import.py - PASSED
✓ test_05_stats_consistency.py - PASSED
→ test_10_github_utils.py - NEXT TO RUN 