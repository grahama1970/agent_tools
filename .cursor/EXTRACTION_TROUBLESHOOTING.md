# Extraction Troubleshooting Guide

## Common Extraction Issues

### Issue: Placeholder Blocks Instead of Actual Content

**Symptoms:**
- Extraction process reports a positive count of blocks (e.g., "284 code blocks extracted")
- Output files contain placeholder blocks instead of actual code
- Warning messages about "No blocks found in file_blocks dictionary"

**Root Cause:**
- Blocks are being counted but not properly stored in the `stats["file_blocks"]` dictionary
- Extraction functions increment counters but fail to collect the actual extracted content

**Solution:**
1. Check each extraction function (`_extract_python_blocks`, `_extract_js_ts_blocks`, `_extract_with_tree_sitter`, `_extract_generic_blocks`, etc.)
2. Ensure each function properly initializes a collection for extracted blocks (e.g., `file_blocks = []`)
3. Verify each block extraction correctly appends metadata to this collection
4. Confirm the function adds the collection to `stats["file_blocks"][str(file_path)]` before returning

### Issue: Functions Not Properly Extracted

**Symptoms:**
- Missing specific functions or classes in the output
- Some files appear to be processed, but certain expected declarations are not found

**Root Cause:**
- AST parser or regex patterns failing to identify specific function/class declarations
- Incorrect language identification leading to wrong parser selection

**Solution:**
1. Check language detection logic is working correctly
2. Verify the appropriate parser is being selected for each file type
3. For Python files, ensure AST parsing is handling all function and class definitions
4. For other languages, check regex patterns or tree-sitter grammar selection

## Debugging Strategies

### 1. Verify Repository Structure

Always check that the repository is properly cloned and structured:

```python
# Verify repository and critical files exist
requests_dir = os.path.join(repo_dir, "src", "requests")
if not os.path.exists(requests_dir):
    print(f"ERROR: src/requests directory not found in cloned repository")
    # List available directories to diagnose
    
api_py_path = os.path.join(requests_dir, "api.py")
if not os.path.exists(api_py_path):
    print(f"ERROR: api.py file not found in {requests_dir}")
    # List available files to diagnose
```

### 2. Check Block Extraction and Storage

Print the number of blocks extracted and details for debugging:

```python
# After processing a file
api_py_blocks = stats["file_blocks"].get(str(api_py_path), [])
print(f"Blocks stored for api.py: {len(api_py_blocks)}")

# List extracted function names
extracted_function_names = [block.get('name', '') for block in api_py_blocks 
                         if block.get('block_type') == 'function']
print(f"Extracted functions: {extracted_function_names}")
```

### 3. Validate Specific Requirements

Always test against specific, concrete expectations:

```python
# Define expected outputs explicitly
required_functions = ["request", "get", "options", "head", "post", "put", "patch", "delete"]

# Check if all required functions were extracted
missing_functions = []
for func_name in required_functions:
    if func_name not in extracted_function_names:
        missing_functions.append(func_name)

# Fail the test if any expected function is missing
if missing_functions:
    print(f"EXTRACTION TEST FAILED: Missing required functions: {', '.join(missing_functions)}")
```

## Testing Principles

1. **Test complete flows** - Don't test extraction functions in isolation; test the entire pipeline from repository download to block extraction
2. **Verify specific outputs** - Define concrete expectations (specific functions, classes, etc.) and validate them
3. **Check data structures** - Don't just verify counts; check that the expected data is properly stored in the appropriate structures
4. **Inspect file outputs** - Examine the actual content of extracted files to ensure quality

## Common Fixes

1. **Collection Initialization**: Ensure each extraction function initializes a list to collect blocks: `file_blocks = []`

2. **Proper Storage**: Verify extraction functions add blocks to the stats dictionary:
   ```python
   if block_count > 0:
       stats["file_blocks"][str(file_path)] = file_blocks
   ```

3. **Block Metadata**: Ensure each block includes complete metadata:
   ```python
   block_data = {
       "type": "code",
       "language": language,
       "content": node_text,
       "name": node_name,
       "block_type": decl_type,
       "file": str(file_path),
       "start_line": start_row,
       "end_line": end_row,
       "output_file": str(output_file)
   }
   file_blocks.append(block_data)
   ```

4. **Error Handling**: Add robust error handling with specific error messages:
   ```python
   except Exception as e:
       error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
       logger.error(error_msg)
       stats["errors"].append(error_msg)
   ```

Remember: Just because the block counter increments doesn't mean the blocks are properly stored for later use! 