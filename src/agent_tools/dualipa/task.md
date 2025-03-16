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
- [ ] Create comprehensive non-mocked tests for code_extractor.py
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
  - [ ] Improve language detection for better handling of edge cases
  - [x] Implement proper fallback mechanisms for unsupported languages
- [ ] Generate QA pairs from extracted blocks (Stage 3)
- [ ] Implement model fine-tuning with Unsloth integration
- [ ] Create adapter merging utilities
- [ ] Implement HuggingFace deployment process
- [ ] Document end-to-end usage examples
- [ ] Optimize performance of extraction and generation
- [ ] Add comprehensive error handling and recovery

## Project Motivation

The goal is to create models that produce current, accurate code that:
1. References up-to-date APIs and standards
2. Follows modern best practices
3. Utilizes the latest framework features
4. Avoids deprecated methods and patterns

By directly encoding these patterns in model weights rather than context, the model's default behavior becomes generating modern code without relying on additional prompt engineering. 