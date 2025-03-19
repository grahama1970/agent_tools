# DuaLipa Function Reference

This document provides detailed information about key functions in the DuaLipa codebase, including parameter requirements and return values. Use this as a reference when working with the codebase.

## code_extractor.py

### extract_repository
```python
def extract_repository(
    repo_path: Union[str, Path],  # Path to repository (string or Path)
    output_dir: Union[str, Path], # Directory for output (string or Path)
    extract_blocks: bool = True,  # Whether to extract code blocks
    include_patterns: Optional[List[str]] = None,  # Glob patterns to include
    exclude_patterns: Optional[List[str]] = None,  # Glob patterns to exclude
    min_tokens: int = 10,         # Minimum tokens for blocks
    max_tokens: int = 4000,       # Maximum tokens for blocks
) -> Dict[str, Any]:              # Returns stats dictionary
```

**Purpose**: Extract code files and optionally code blocks from a repository

**Critical Requirements**:
- `repo_path` must exist and be readable
- `output_dir` will be created if it doesn't exist
- Returns a dictionary with extraction statistics

### _extract_python_blocks
```python
def _extract_python_blocks(
    file_path: Path,              # MUST be Path object, not string
    content: str,                 # Content of the file
    output_dir: Path,             # Output directory (Path object)
    stats: Dict[str, Any]         # MUST contain keys: "code_blocks", "errors", "file_blocks"
) -> int:                         # Returns number of blocks extracted
```

**Purpose**: Extract functions and classes from Python code using AST

**Critical Requirements**:
- `file_path` MUST be a Path object, not a string
- `stats` MUST be initialized with required keys:
  ```python
  stats = {
      "code_blocks": 0,  # Counter for blocks
      "errors": [],      # List to store any errors
      "file_blocks": {}  # Dictionary to track blocks by file
  }
  ```
- For defensive programming, the function will initialize missing keys:
  ```python
  stats.setdefault("code_blocks", 0)  # Will be added if missing
  ```
- Internally uses a `language = "python"` variable to handle script files that don't contain functions or classes

**Script-Level Extraction**:
- For files like `setup.py`, `manage.py`, `app.py`, etc., which often don't have traditional function/class blocks, the entire file is extracted as a script-level block
- Script-level blocks are also created for files with top-level executable statements
- Script-level blocks are included in the `stats["code_blocks"]` count
- They are marked with `# Block type: script` in the output files

**Known Limitations**:
- Python's AST parser flattens nested class hierarchies. Classes defined inside other classes will be extracted as separate, top-level entities without preserving their nested relationship.

### _extract_markdown_blocks
```python
def _extract_markdown_blocks(
    file_path: Path,              # MUST be Path object, not string
    content: str,                 # Content of the file
    output_dir: Path,             # Output directory (Path object)
    stats: Dict[str, Any]         # MUST contain keys: "doc_blocks", "code_blocks", "errors", "file_blocks"
) -> int:                         # Returns number of blocks extracted
```

**Purpose**: Extract sections from markdown files based on headings

**Critical Requirements**:
- `file_path` MUST be a Path object, not a string
- `stats` MUST be initialized with these keys:
  ```python
  stats = {
      "code_blocks": 0,  # Counter for code blocks
      "doc_blocks": 0,   # Counter for documentation blocks
      "errors": [],      # List to store any errors
      "file_blocks": {}  # Dictionary to track blocks by file
  }
  ```
- For defensive programming, the function will initialize missing keys:
  ```python
  stats.setdefault("doc_blocks", 0)  # Will be added if missing
  ```

### _extract_js_ts_blocks
```python
def _extract_js_ts_blocks(
    file_path: Path,              # MUST be Path object, not string
    content: str,                 # Content of the file
    output_dir: Path,             # Output directory (Path object)
    stats: Dict[str, Any]         # MUST contain keys: "code_blocks", "errors", "file_blocks"
) -> int:                         # Returns number of blocks extracted
```

**Purpose**: Extract functions, classes, and components from JavaScript and TypeScript using tree-sitter

**Critical Requirements**:
- `file_path` MUST be a Path object, not a string
- File extension determines language (.js, .jsx, .ts, .tsx)
- `stats` MUST be initialized with these keys:
  ```python
  stats = {
      "code_blocks": 0,  # Counter for code blocks
      "errors": [],      # List to store any errors
      "file_blocks": {}  # Dictionary to track blocks by file
  }
  ```
- For defensive programming, the function will initialize missing keys:
  ```python
  stats.setdefault("code_blocks", 0)  # Will be added if missing
  ```

**Tree-Sitter Integration**:
- Uses tree-sitter for accurate parsing
- Automatically selects correct parser based on file extension
- Extracts:
  - Function declarations
  - Arrow functions
  - Class declarations
  - React components (uppercase names)
- Falls back to generic extraction for unsupported constructs

**Example Usage**:
```python
# Initialize stats
stats = initialize_stats_dict(source=file_path, output_dir=output_dir)

# Extract from JavaScript
js_file = Path("example.js")
num_blocks = _extract_js_ts_blocks(js_file, js_content, output_dir, stats)

# Extract from TypeScript
ts_file = Path("example.ts")
num_blocks = _extract_js_ts_blocks(ts_file, ts_content, output_dir, stats)

# Extract from React component
tsx_file = Path("Component.tsx")
num_blocks = _extract_js_ts_blocks(tsx_file, tsx_content, output_dir, stats)
```

**Language Detection**:
- `.js`, `.jsx` → JavaScript parser
- `.ts`, `.tsx` → TypeScript parser
- Updates `stats["languages"]` and `stats["file_types"]` automatically

**Error Handling**:
- Adds errors to `stats["errors"]` list
- Continues extraction on parser errors
- Falls back to generic extraction if tree-sitter fails

### _extract_hierarchical_structure_treesitter
```python
def _extract_hierarchical_structure_treesitter(
    code: str,           # Source code to parse
    language: str,       # Language identifier
    filename: str = None # Optional source filename
) -> Dict[str, Any]:    # Returns hierarchical structure
```

**Purpose**: Extract hierarchical structure from code using tree-sitter

**Critical Requirements**:
- Returns a dictionary with the following structure:
  ```python
  {
      "file": "path/to/file.ext",
      "language": "language_id",
      "blocks": [
          {
              "type": "interface|class|function|method",
              "name": "string",
              "content": "exact source code",
              "start_line": number,
              "end_line": number,
              "methods": [...],  # For classes
              "implementations": [...],  # For interfaces
              "decorators": [...],  # For Python/TypeScript
              "metadata": {
                  "visibility": "public|private|protected",
                  "static": boolean,
                  "async": boolean
              }
          }
      ],
      "order": ["block_names_in_declaration_order"],
      "stats": {
          "total_blocks": number,
          "by_type": {"class": number, ...}
      }
  }
  ```

**Method Counting Rules**:
1. For interfaces:
   - Count all method declarations
   - Exclude constructors from method count
   - Include getter/setter methods
   - Include async methods

2. For classes:
   - Count all methods including constructors
   - Count private and protected methods
   - Count static methods
   - Count async methods
   - Include getter/setter methods

**Stats Counting Rules**:
- `total_blocks`: Sum of all blocks (classes, interfaces, methods)
- `by_type`: Count of each block type:
  - `interface`: Number of interfaces
  - `class`: Number of classes
  - `method`: Total number of methods (following method counting rules)

**Example Usage**:
```python
# Extract from TypeScript interface
result = _extract_hierarchical_structure_treesitter(
    code='''
    interface Repository<T> {
        find(id: string): Promise<T>;
        save(entity: T): Promise<T>;
        delete(id: string): Promise<void>;
    }
    ''',
    language="typescript",
    filename="repository.ts"
)

# Result will contain:
# - 1 interface block
# - 3 method blocks
# - total_blocks = 4
# - by_type = {"interface": 1, "method": 3}

# Extract from TypeScript class
result = _extract_hierarchical_structure_treesitter(
    code='''
    class UserService {
        constructor(private db: Database) {}
        async findById(id: string): Promise<User> {
            return this.db.findUnique({ id });
        }
        private validateUser(user: User): boolean {
            return user.isValid();
        }
    }
    ''',
    language="typescript",
    filename="user.service.ts"
)

# Result will contain:
# - 1 class block
# - 3 method blocks (constructor + findById + validateUser)
# - total_blocks = 4
# - by_type = {"class": 1, "method": 3}
```

## verification/verify_code_blocks.py

### verify_code_block
```python
def verify_code_block(
    block: Dict[str, Any],        # Code block to verify (dictionary)
    language: Optional[str] = None # Language to verify against (optional)
) -> bool:                        # Returns whether the block is valid
```

**Purpose**: Verify if a code block is valid for the given language

**Critical Requirements**:
- `block` must be a dictionary with at least the keys:
  ```python
  block = {
      "language": "python",  # Language of the code block
      "content": "...",      # Content of the code block
      "file": "sample.py"    # File name or path of the code block
  }
  ```
- `language` is optional and will default to the block's language if not provided

## github_utils.py

### parse_github_url
```python
def parse_github_url(url: str) -> Dict[str, str]:
    """Parse a GitHub URL into components.
    
    Args:
        url: The GitHub URL to parse
        
    Returns:
        Dictionary with components:
            - owner: Repository owner/organization
            - repo: Repository name
            - path: Path within repository (empty if none)
            - branch: Branch name (defaults to 'main')
            - protocol: Protocol ('https' or 'ssh')
            - subdir: Subdirectory path (same as path)
            
    Raises:
        ValueError: With specific messages for different validation failures:
            - "Empty or invalid URL provided"
            - "Not a GitHub URL"
            - "Invalid GitHub SSH URL"
            - "Invalid GitHub repository path"
    """
```

**Critical Requirements**:
- URL must be a valid GitHub URL (HTTPS or SSH format)
- Returns consistent fields regardless of URL format
- Handles both repository root URLs and specific paths
- Validates URL format before parsing
- MUST include 'protocol' and 'subdir' fields in return value
- Error messages must match test expectations exactly

### clone_github_repo
```python
def clone_github_repo(url: str, temp_dir: Optional[str] = None) -> str:
    """Clone a GitHub repository to a temporary directory.
    
    Args:
        url: GitHub repository URL (HTTPS or SSH)
        temp_dir: Optional directory for cloning (creates temp dir if None)
        
    Returns:
        Path to the cloned repository
        
    Raises:
        ValueError: If URL is not a valid GitHub URL
        GitCommandError: For repository errors with specific messages:
            - "Repository not found" for non-existent or private repos
            - Original git error message for other failures
    """
```

**Critical Requirements**:
- Validates URL with is_github_url() before attempting to clone
- Creates and manages temporary directory if none provided
- Cleans up temporary directory on any failure
- Handles both HTTPS and SSH URLs
- Preserves original git error messages for debugging
- For private repositories, preserves "Repository not found" message

### is_github_url
```python
def is_github_url(url: str) -> bool:
    """Check if a URL is a GitHub repository URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a valid GitHub repository URL
    """
```

**Critical Requirements**:
- Handles both HTTPS and SSH formats
- Validates domain is exactly 'github.com'
- Checks for owner/repo path structure
- Returns False for non-GitHub URLs
- Must be used before attempting any repository operations

### Testing Requirements
- Use real repositories from test_repos/ for validation
- Test both HTTPS and SSH URL formats
- Test error cases with specific error messages:
  - "Not a GitHub URL" for invalid URLs
  - "Repository not found" for non-existent/private repos
  - "Invalid GitHub repository path" for malformed paths
- Verify repository structure after cloning
- Clean up temporary directories in all cases
- Handle network errors gracefully
- Test URL parsing with various formats:
  - Root repository URLs
  - Branch-specific URLs
  - Subdirectory URLs
  - SSH URLs with and without .git suffix

### Error Handling Best Practices
1. Validate URLs before any operations
2. Use consistent error messages that match test expectations
3. Clean up resources on any failure
4. Preserve git error messages for debugging
5. Handle private repositories as "Repository not found"
6. Log errors with appropriate context

### fetch_repo_contents_async
```python
async def fetch_repo_contents_async(
    owner: str,
    repo: str,
    path: str = ""
) -> List[Dict[str, str]]:
    """Fetch repository contents asynchronously.
    
    Args:
        owner: Repository owner or 'local' for local repos
        repo: Repository name or path for local repos
        path: Optional path within repository
        
    Returns:
        List of dictionaries with file information:
            - name: File name
            - path: File path relative to repo root
            - type: File type ('file' or 'dir')
            
    Raises:
        ValueError: If PyGithub not available or invalid parameters
    """
```

**Critical Requirements**:
- Handles both remote and local repositories
- Validates paths before access
- Returns consistent structure for both remote/local
- Requires PyGithub for remote repositories

## pipeline.py

### run_pipeline
```python
def run_pipeline(
    repo_path: Union[str, Path],  # Path to repository (string or Path)
    output_dir: Union[str, Path], # Directory for output (string or Path)
    run_extract: bool = True,     # Whether to run extraction phase
    run_format: bool = True,      # Whether to run formatting phase
    run_train: bool = True,       # Whether to run training phase
    upload_to_hf: bool = False,   # Whether to upload to HuggingFace
    model_name: Optional[str] = None # Name for the model when uploading
) -> Dict[str, Any]:              # Returns pipeline results
```

**Purpose**: Run the complete DuaLipa pipeline from repository to model

**Critical Requirements**:
- `repo_path` must be a valid repository path or URL
- `output_dir` will be created if it doesn't exist
- `model_name` is required if `upload_to_hf` is True
- Returns a dictionary with pipeline results

## format_dataset.py

### format_for_lora
```python
def format_for_lora(
    extracted_dir: Union[str, Path],  # Directory with extracted files/blocks
    output_dir: Union[str, Path],     # Directory for formatted output
    use_llm: bool = True,             # Whether to use LLM for generation
    file_types: Optional[List[str]] = None  # File types to include
) -> Dict[str, Any]:                  # Returns formatting statistics
```

**Purpose**: Format extracted code into training data for LoRA fine-tuning

**Critical Requirements**:
- `extracted_dir` must contain output from extract_repository
- `output_dir` will be created if it doesn't exist
- `use_llm` requires API key if True
- Returns a dictionary with formatting statistics 

## Best Practices

### Testing and Implementation
- **Fix Implementation, Not Tests**: When tests fail, fix the actual implementation code rather than modifying tests to accommodate broken code. Tests serve as specifications of intended behavior.
- **Script-Level Extraction**: Several special files (e.g., `setup.py`, `webpack.config.js`) don't contain traditional blocks but are extracted as entire script blocks. Ensure these are properly counted in statistics.
- **Stats Consistency**: Always increment the appropriate counters (`stats["code_blocks"]`, `stats["doc_blocks"]`) when adding blocks to maintain accurate statistics.
- **Defensive Programming**: Use `setdefault()` to ensure required dictionary keys exist before incrementing them. 

## Stats Dictionary Requirements

All extraction functions require a properly initialized stats dictionary with:

```python
stats = {
    # Core counters
    "code_blocks": 0,      # Required for all code extractors
    "doc_blocks": 0,       # Required for markdown extraction
    "errors": [],         # Required for all extractors
    "file_blocks": {},    # Required for all extractors
    
    # Language tracking
    "languages": {},      # Required for language statistics
    "file_types": {},     # Required for file type tracking
    
    # Metadata
    "source": str,        # Source path/URL
    "output_path": str,   # Output directory path
    "start_time": str,    # ISO format timestamp
    "end_time": None,     # Set when extraction completes
    "duration_seconds": 0 # Updated when extraction completes
}
```

### Critical Stats Update Points

1. **Language Statistics Updates**
   - Must be updated when processing ANY file
   - Update in language-specific extractors
   - Example:
     ```python
     stats["languages"][language] = stats["languages"].get(language, 0) + 1
     ```

2. **File Type Statistics Updates**
   - Must be updated when processing ANY file
   - Update after determining file extension
   - Example:
     ```python
     ext = file_path.suffix.lower()
     stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
     ```

3. **Block Counter Updates**
   - Must be updated when extracting ANY block
   - Different counters for code vs documentation
   - Example:
     ```python
     stats["code_blocks"] += 1  # For code blocks
     stats["doc_blocks"] += 1   # For documentation blocks
     ```

4. **Error Handling**
   - Must capture errors for ALL failure cases
   - Preserve existing stats when errors occur
   - Example:
     ```python
     try:
         # extraction code
     except Exception as e:
         stats["errors"].append(str(e))
     ```

### Stats Verification Requirements

1. **Structure Verification**
   - All required fields must exist
   - Fields must have correct types
   - Example:
     ```python
     assert isinstance(stats["languages"], dict)
     assert isinstance(stats["code_blocks"], int)
     ```

2. **Value Verification**
   - Counters must be non-negative
   - Lists and dicts must be initialized
   - Example:
     ```python
     assert stats["code_blocks"] >= 0
     assert isinstance(stats["errors"], list)
     ```

3. **Cross-Language Consistency**
   - All extractors must maintain consistent stats
   - Language-specific stats must be properly tracked
   - Example:
     ```python
     assert "python" in stats["languages"]  # After Python extraction
     assert ".py" in stats["file_types"]    # After Python file processing
     ``` 

## Linter Error Handling

### When to Disable Linters
Only disable linters when ALL of these conditions are met:
1. The error is a proven false positive
2. You have documentation supporting the code pattern
3. Restructuring the code would make it less readable or maintainable

### How to Disable Linters
```python
# For single-line disables:
# pylint: disable=specific-error  # Justification: <documentation link>
code_that_triggers_error()

# For block disables:
# pylint: disable=specific-error
"""Justification:
1. Link to documentation: <url>
2. Explanation of why this is correct
"""
def block_of_code():
    pass
# pylint: enable=specific-error
```

### Common False Positives
1. Try/Except Block Alignment
   - When the linter misinterprets complex nested try/except blocks
   - Solution: Restructure into smaller functions when possible
   - Only disable if restructuring would harm readability

2. Import Order
   - When standard library and local imports have dependencies
   - Solution: Group imports logically with comments explaining order
   - Only disable if order is required for initialization

3. Line Length
   - For long strings, URLs, or formatted strings
   - Solution: Use line continuation when possible
   - Only disable for unbreakable items like URLs

### Required Documentation
Every linter disable MUST include:
1. Link to official documentation supporting the pattern
2. Clear explanation of why it's a false positive
3. Why restructuring isn't a better solution 