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

**Purpose**: Extract functions, classes, and components from JavaScript and TypeScript

**Critical Requirements**:
- `file_path` MUST be a Path object, not a string
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
- Similar to Python extraction, also handles script-level extraction for files like `webpack.config.js`

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

### download_github_repo
```python
def download_github_repo(
    repo_url: str,               # URL to GitHub repository
    target_dir: Optional[Union[str, Path]] = None,  # Target directory (optional)
    branch: Optional[str] = None # Branch to checkout (optional)
) -> Path:                       # Returns path to downloaded repository
```

**Purpose**: Download a GitHub repository to a local directory

**Critical Requirements**:
- `repo_url` must be a valid GitHub repository URL
- `target_dir` will be created if it doesn't exist
- Returns a Path object pointing to the downloaded repository

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