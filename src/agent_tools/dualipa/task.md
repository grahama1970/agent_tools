# DuaLipa Pipeline: Updating LLMs with Current Code Knowledge

## AI Agent Task Reference

This document serves as a structured reference for AI agents working with the DuaLipa pipeline. The information is organized to facilitate rapid information retrieval and task execution.

## Core Purpose: Model Pre-training Bias Correction

DuaLipa addresses the critical limitation of frontier language models: outdated code knowledge within pre-trained weights. Models exhibit:
- Pre-training bias toward deprecated code patterns
- Strong resistance to context-provided examples
- Persistent method hallucinations despite context correction
- Default behavior favoring outdated training data over context

The correction mechanism leverages LoRA adapters with Unsloth to encode current coding patterns directly into model weights.

## Project Workflow Architecture

Sequential pipeline stages:
1. Repository Download [SOURCE ACQUISITION]
2. Code and Documentation Extraction [DATA EXTRACTION]
3. QA Pair Generation and Dataset Formatting [DATA PREPARATION]
4. Model Fine-tuning with Unsloth [MODEL TRAINING]
5. LoRA Adapter Merging [MODEL OPTIMIZATION]
6. Deployment to Hugging Face [DISTRIBUTION]
7. Usage for Current Code Generation [APPLICATION]

## Component Specifications

### Stage 1: Repository Download
**Purpose**: Acquire source code repositories
**Components**:
- `github_utils.py`: Repository acquisition (`download_github_repo()`)
**Data Flow**: URL → Local repository files

### Stage 2: Code and Documentation Extraction
**Purpose**: Extract and process code/documentation from repositories
**Components**:
- `code_extractor.py`: Content filtering and processing (`extract_repository()`)
  - Stores complete files and extracts logical blocks
  - Uses AST for Python, regex patterns for JS/TS
  - Splits markdown by headers for documentation
- `language_detection.py`: Programming language identification
**Tasks**:
- Filter files by extensions and patterns
- Extract complete files with source information
- Extract logical blocks (functions, classes, markdown sections)
- Process multiple programming languages
- Generate structured representation with context
**Data Flow**: Repository files → Complete files and logical blocks for training

### Stage 3: QA Pair Generation and Dataset Formatting
**Purpose**: Transform code into training data
**Components**:
- `format_dataset.py`: Data structuring (`format_for_lora()`)
- `llm_generator.py`: Q&A synthesis engine
- `qa_validator.py`: Quality control
**Data Flow**: Structured repo → JSONL training dataset

### Stage 4: Model Fine-tuning with Unsloth
**Purpose**: Encode current code patterns into model weights
**Components**:
- `train_lora.py`: Training orchestration (`train_lora()`)
**Data Flow**: JSONL dataset → LoRA adapter weights

### Stage 5: LoRA Adapter Merging
**Purpose**: Create deployment-ready model
**Tasks**:
- Base model + adapter integration
- Inference optimization
- Quality validation
**Data Flow**: Adapter + Base model → Updated model

### Stage 6: Deployment to Hugging Face
**Purpose**: Distribution mechanism
**Tasks**:
- Model packaging
- Metadata configuration
- Access provisioning
**Data Flow**: Updated model → HF endpoint

## Function Call Patterns

**Pipeline Execution**:
```python
from agent_tools.dualipa.pipeline import run_pipeline

run_pipeline(
    repo_path="https://github.com/username/repo",
    output_dir="./output",
    run_extract=True,
    run_format=True,
    run_train=True,
    upload_to_hf=True,
    model_name="my-updated-code-model"
)
```

**Component Testing**:
   ```python
# Stage 1: Download
   from agent_tools.dualipa.github_utils import download_github_repo
repo_path = download_github_repo("https://github.com/username/repo")

# Stage 2: Extract (Complete Files)
   from agent_tools.dualipa.code_extractor import extract_repository
   extracted_data = extract_repository(repo_path, "./extracted_output", extract_blocks=False)

# Stage 2: Extract (With Code Blocks)
   from agent_tools.dualipa.code_extractor import extract_repository
   extracted_data = extract_repository(repo_path, "./extracted_output", extract_blocks=True)

# Stage 3: Format
   from agent_tools.dualipa.format_dataset import format_for_lora
   formatted_data = format_for_lora("./extracted_output", "./formatted_output")

# Stage 4: Train
   from agent_tools.dualipa.train_lora import train_lora
adapter_path = train_lora("./formatted_output", "./model_output")

# Stage 5-6: Merge and Deploy
from agent_tools.dualipa.model_utils import merge_and_push_model
merge_and_push_model(adapter_path, "base-model-name", "my-updated-model", push_to_hub=True)
```

## Task Tracking

- [x] Fix GitHub repository download functionality
- [x] Create working smoke tests for repository download
- [x] Update task.md with complete pipeline information
- [x] Separate extraction as its own complex stage
- [x] Implement repository code and documentation extraction (Stage 2)
- [x] Store complete files in organized directory structure
- [x] Implement code block extraction using AST for Python and regex for other languages
- [x] Extract markdown sections for documentation blocks
- [x] Refactor demo_code_extractor to use separate template files for better maintainability
- [x] Create resources/templates directory for sample code files
- [x] Create utils.py module with format_string and other utility functions
- [x] Update imports across the codebase to use the utils module
- [x] Move functions from __init__.py to cli.py following best practices
- [x] Implement flexible import structure in code_extractor.py to support both package and standalone execution
- [x] Create helper scripts (__main__.py and run_extractor.py) to facilitate different execution modes
- [x] Update pyproject.toml with proper package discovery and entry points
- [x] Create comprehensive non-mocked tests for code_extractor.py
  - [x] Test Python code block extraction using AST with enhanced metadata (decorators, type hints, imports)
  - [x] Test Markdown section extraction and code block identification
  - [x] Test JavaScript/TypeScript block extraction
  - [x] Test generic code splitting by double newlines
  - [x] Verify no empty chunks are generated
  - [x] Test with resources/templates for expected results
- [x] Enhance code extraction with more advanced parsing
  - [x] Implement tree-sitter for JavaScript/TypeScript extraction
  - [x] Add support for more programming languages with tree-sitter:
    - [x] Python - fully functional
    - [x] JavaScript - fully functional
    - [x] TypeScript - fully functional
    - [x] Go - fully functional
    - [x] Rust - fully functional
    - [x] C++ - fully functional
    - [x] Java - fully functional
    - [x] Ruby - fully functional
    - [x] Bash - fully functional
    - [x] C - grammar version incompatibility (using regex fallback)
    - [x] PHP - grammar available but missing language attribute
  - [x] Improve language detection by using file extensions
  - [x] Implement proper fallback mechanisms for unsupported languages
  - [x] Simplify tree-sitter initialization (remove unnecessary checks)
  - [x] Ensure consistent stats dictionary across all extractors
  - [x] Implement progressive test verification
    - [x] Basic imports and dependencies (test_01_simple.py)
    - [x] Stats dictionary consistency (test_05_stats_consistency.py)
    - [x] Language-specific extraction (test_20_python_extractor.py, test_30_js_ts_extraction.py)
    - [x] Supporting utilities (test_10_github_utils.py)
- [ ] Generate QA pairs from extracted blocks (Stage 3)
- [ ] Implement model fine-tuning with Unsloth integration
- [ ] Create adapter merging utilities
- [ ] Implement HuggingFace deployment process
- [ ] Document end-to-end usage examples
- [ ] Optimize performance of extraction and generation
- [ ] Add comprehensive error handling and recovery

## Test-Driven Development Workflow

### 1. Tree-Sitter Integration Tasks

#### Current Focus: JavaScript/TypeScript Extraction
- [ ] Implement tree-sitter extraction for JS/TS following requirements:
  - Function: `_extract_js_ts_blocks()` in code_extractor.py
  - Test: `tests/dualipa/extraction/final_order/test_30_js_ts_extraction.py`
  
  **Input Specification**:
  ```python
  def _extract_js_ts_blocks(
      file_path: Path,  # Source file path
      content: str,     # File content
      output_dir: Path, # Output directory for blocks
      stats: Dict[str, Any]  # Stats dictionary
  ) -> int:  # Returns number of blocks extracted
  ```

  **Expected Results**:
  1. File Structure:
     ```
     output_dir/
     └── blocks/
         └── code/
             └── {javascript|typescript}/
                 └── {block_name}_{hash}.{js|ts}
     ```
  
  2. Block Content:
     ```javascript
     // Complete, self-contained code blocks like:
     function add(a, b) {
         return a + b;
     }
     
     // Or TypeScript interfaces/classes:
     interface User {
         name: string;
         age: number;
     }
     ```
  
  3. Stats Updates:
     ```python
     stats["code_blocks"] += 1
     stats["languages"].add("javascript")  # or "typescript"
     stats["file_types"].add(".js")  # or ".ts", ".tsx"
     ```

#### Tree-Sitter Hierarchy Implementation
- [ ] Implement hierarchical extraction:
  - Function: `_extract_hierarchical_structure_treesitter()` in code_hierarchy.py
  - Test: `tests/dualipa/extraction/final_order/test_31_tree_sitter_hierarchy.py`
  
  **Input Specification**:
  ```python
  def _extract_hierarchical_structure_treesitter(
      code: str,           # Source code to parse
      language: str,       # Language identifier
      filename: str = None # Optional source filename
  ) -> Dict[str, Any]:    # Returns hierarchical structure
  ```

  **Expected Results**:
  1. For TypeScript Interface:
     ```python
     # Input:
     """
     interface User {
         name: string;
         age: number;
     }
     """
     
     # Expected Output:
     {
         "file": "example.ts",
         "language": "typescript",
         "blocks": [{
             "type": "interface",
             "name": "User",
             "content": "interface User {\n    name: string;\n    age: number;\n}",
             "start_line": 1,
             "end_line": 4,
             "methods": [],
             "implementations": [],
             "metadata": {"visibility": "public"}
         }],
         "order": ["User"],
         "stats": {
             "total_blocks": 1,
             "by_type": {"interface": 1}
         }
     }
     ```

  2. For TypeScript Class with Methods:
     ```python
     # Input:
     """
     class UserService {
         private users: User[];
         
         constructor() {
             this.users = [];
         }
         
         async addUser(user: User): Promise<void> {
             this.users.push(user);
         }
     }
     """
     
     # Expected Output:
     {
         "file": "example.ts",
         "language": "typescript",
         "blocks": [{
             "type": "class",
             "name": "UserService",
             "content": "<complete class code>",
             "start_line": 1,
             "end_line": 10,
             "methods": [
                 {
                     "type": "method",
                     "name": "constructor",
                     "content": "constructor() {\n    this.users = [];\n}",
                     "start_line": 4,
                     "end_line": 6,
                     "metadata": {"visibility": "public"}
                 },
                 {
                     "type": "method",
                     "name": "addUser",
                     "content": "async addUser(user: User): Promise<void> {\n    this.users.push(user);\n}",
                     "start_line": 8,
                     "end_line": 10,
                     "metadata": {
                         "visibility": "public",
                         "async": true
                     }
                 }
             ],
             "implementations": [],
             "metadata": {"visibility": "public"}
         }],
         "order": ["UserService"],
         "stats": {
             "total_blocks": 3,  # Class + 2 methods
             "by_type": {
                 "class": 1,
                 "method": 2
             }
         }
     }
     ```

  **Verification Steps**:
  1. Run hierarchy test with specific examples:
     ```bash
     pytest tests/dualipa/extraction/final_order/test_31_tree_sitter_hierarchy.py::test_typescript_interface_hierarchy -v
     pytest tests/dualipa/extraction/final_order/test_31_tree_sitter_hierarchy.py::test_class_hierarchy_with_nested_structure -v
     ```
  2. Verify each component matches expected output exactly
  3. Ensure no code blocks are broken/partial
  4. Confirm stats are updated correctly

### 2. Stats Tracking Requirements

For all extraction functions, maintain consistent stats:
```python
# Required Stats Initialization
stats.setdefault("code_blocks", 0)
stats.setdefault("errors", [])
stats.setdefault("file_blocks", {})
stats.setdefault("languages", set())
stats.setdefault("file_types", set())

# Required Stats Updates
stats["code_blocks"] += 1  # For each extracted block
stats["languages"].add(language)  # When processing a file
stats["file_types"].add(file_extension)  # When processing a file
```

### 3. Error Handling Requirements

Implement defensive programming patterns:
```python
try:
    # Extraction logic
    pass
except Exception as e:
    stats["errors"].append({
        "file": str(file_path),
        "error": str(e),
        "type": "extraction_error"
    })
    return 0  # Indicate no blocks extracted
```

### 4. Testing Best Practices

1. **Progressive Testing**:
   - Start with basic functionality
   - Add edge cases
   - Verify error handling
   - Check stats consistency

2. **Test Isolation**:
   - Each test should be independent
   - Use temporary directories
   - Clean up resources
   - Mock external dependencies

3. **Assertion Guidelines**:
   - Be specific about failures
   - Check both positive and negative cases
   - Verify all required fields
   - Compare complete structures

## Project Motivation

The goal is to create models that produce current, accurate code that:
1. References up-to-date APIs and standards
2. Follows modern best practices
3. Utilizes the latest framework features
4. Avoids deprecated methods and patterns

By directly encoding these patterns in model weights rather than context, the model's default behavior becomes generating modern code without relying on additional prompt engineering.

## Critical Testing Rules

### 1. NEVER Modify Tests to Make Them Pass
- ❌ NEVER change file extensions in tests to make them pass
- ❌ NEVER modify test assertions to match incorrect implementation
- ❌ NEVER alter test data to work around implementation bugs
- ✅ Instead: Fix the implementation to handle the test cases correctly

Example of what NOT to do:
```python
# WRONG: Changing test to look for wrong extension
block_files = list(blocks_dir.glob("*.ts"))  # Don't change this to match implementation!

# WRONG: Modifying test data to avoid implementation issues
tsx_file = temp_dir_path / "ListItem.ts"  # Don't change extensions to bypass problems!
```

### 2. Test Integrity Principles
1. Tests are the source of truth
2. Tests document expected behavior
3. Tests verify real-world use cases
4. Implementation must adapt to tests, not vice versa

### 3. Lessons Learned from Tree-Sitter Integration
1. **Read Tests First**:
   - Tests document requirements and expectations
   - Test docstrings contain critical information
   - Test data (like file extensions) is chosen deliberately

2. **Understand Test Structure**:
   ```python
   # Example: Progressive test complexity
   test_js_function_extraction()  # Basic JS handling
   test_ts_class_extraction()     # TypeScript features
   test_tsx_component_extraction() # React/TSX complexity
   ```

3. **Debug Properly**:
   - Start by verifying test inputs exist
   - Check if implementation handles file types correctly
   - Validate parser selection logic
   - Verify output paths and extensions match real use

4. **Maintain Context**:
   - Document lessons learned
   - Reference similar issues/solutions
   - Build on previous experiences
   - Don't repeat solved problems

### 4. Implementation Checklist
Before modifying code:
- [ ] Read and understand ALL test requirements
- [ ] Verify test data exists and is correct
- [ ] Check test progression (simple → complex)
- [ ] Understand why specific file types/extensions are used
- [ ] Document any assumptions or dependencies

After modifying code:
- [ ] Verify ALL tests pass without modification
- [ ] Check if changes maintain backward compatibility
- [ ] Validate against real-world examples
- [ ] Update documentation with new insights 