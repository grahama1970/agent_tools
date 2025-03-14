# DuaLipa: Dual LLM-Informed Python Automation

DuaLipa is a toolkit for generating high-quality question-answer pairs from code repositories and documentation. It uses LLMs to enhance the depth and quality of generated QA pairs, which can be used for fine-tuning language models or creating knowledge bases.

## Features

- **Repository Processing**: Extract code and documentation from GitHub repositories or local directories
- **Language Detection**: Automatically detect programming languages and categorize files
- **Q&A Generation**: Create high-quality question-answer pairs using LLMs or basic extraction
- **Q&A Validation**: Validate, deduplicate, and enhance generated QA pairs
- **Format Conversion**: Convert extracted data into formats suitable for various downstream tasks
- **CLI Interface**: Easy-to-use command-line interface for all functionality

## Installation

### Prerequisites

- Python 3.8+
- pip or pipenv for package management

### Basic Installation

```bash
# From PyPI (when published)
pip install dualipa

# From source
git clone https://github.com/yourusername/dualipa.git
cd dualipa
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/dualipa.git
cd dualipa
pip install -e ".[dev]"
```

## Usage

### CLI Interface

DuaLipa provides a simple command-line interface for basic operations:

```bash
# Extract code from a repository
dualipa extract https://github.com/username/repo extracted_data/

# Format the extracted data into QA pairs
dualipa format extracted_data/extraction.json qa_dataset.json --use-llm

# Run the complete pipeline (extract, format, and optionally train)
dualipa pipeline https://github.com/username/repo output_dir/ --use-llm --max-pairs 10

# Run the pipeline with training enabled
dualipa pipeline https://github.com/username/repo output_dir/ --use-llm --run-train
```

### Python API

DuaLipa can also be used as a Python library:

```python
from agent_tools.dualipa import code_extractor, format_dataset, llm_generator

# Extract code from a repository
stats = code_extractor.extract_repository(
    "https://github.com/username/repo",
    "extracted_data/",
    max_files=500,
    extract_documentation=True,
    extract_code=True
)

# Format the extracted data into QA pairs
qa_stats = format_dataset.format_for_lora(
    "extracted_data/data.json",
    "formatted_data.json",
    use_llm=True,
    max_pairs_per_item=10
)

# Generate QA pairs directly for a specific file
if llm_generator.check_litellm_available():
    with open("sample.py", "r") as f:
        content = f.read()
    
    qa_pairs = llm_generator.generate_code_related_qa_pairs(
        content,
        "calculate_average",
        entity_type="function"
    )
    print(qa_pairs)
```

## Project Structure

The DuaLipa package follows a functional, modular design with proper error handling and documentation:

```
src/agent_tools/dualipa/
├── __init__.py              # Package initialization and CLI interface
├── code_extractor.py        # Extract code from repositories
├── format_dataset.py        # Convert extracted data to QA pairs
├── github_utils.py          # GitHub repository interactions
├── language_detection.py    # Programming language detection
├── llm_generator.py         # LLM-based QA pair generation
├── markdown_parser.py       # Markdown content parsing
├── qa_validator.py          # QA pair validation and enhancement
├── LESSONS_LEARNED.md       # Documentation of lessons and improvements
└── tests/                   # Test files
    ├── __init__.py
    ├── test_basic_dependencies.py
    ├── test_code_extractor.py
    ├── test_enhanced_qa_generation.py
    └── test_llm_integration.py
```

## Code Organization Best Practices

DuaLipa follows these code organization practices:

1. **Module Documentation**: Each module begins with a comprehensive docstring including:
   - Module purpose and functionality
   - Official documentation references
   - Dependencies and their links

2. **Error Handling**: Proper error handling using loguru with:
   - Descriptive error messages
   - Appropriate logging levels
   - Exception catching and reporting

3. **Type Hints**: Comprehensive type hints for all functions and parameters

4. **Demonstration Functions**: Each module includes a `demo_*` function that:
   - Shows practical examples of the module's functionality
   - Creates sample data for demonstration
   - Handles errors properly
   - Provides clear output

5. **Command-Line Interface**: Each module can be run directly with:
   - Demonstration mode when run without arguments
   - Practical CLI functionality when run with arguments

6. **Testing**: Comprehensive test coverage using pytest:
   - Unit tests for individual functions
   - Integration tests for component interaction
   - Smoke tests for dependency verification

## Enhanced QA Generation Features

DuaLipa includes several enhancements to QA generation:

1. **Temperature Variation**: Random temperature variation for diverse responses:
   - Code content: 0.1-0.5 temperature range
   - Markdown content: 0.3-0.7 temperature range
   - Reverse QA: 0.3-0.6 temperature range

2. **RapidFuzz Integration**: Improved QA pair validation:
   - Detection and elimination of near-duplicate questions
   - Enhanced function-related questions
   - Better categorization of QA pairs

3. **Function Code Completeness**: Ensuring complete function code in answers

4. **Validation Pipeline**: Structured validation process for QA pairs

## LLM Integration

DuaLipa supports multiple LLM providers through LiteLLM:

1. Configure your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. Or use a LiteLLM configuration file at `~/.litellm.config.yaml`

## Development Guidelines

When contributing to DuaLipa, please follow these guidelines:

1. **Follow the Module Template**: Use existing modules as templates for new ones
2. **Add Demonstration Functions**: Every module should have a `demo_*` function
3. **Ensure Error Handling**: Use loguru for consistent error reporting
4. **Include Type Hints**: All functions should have proper type hints
5. **Write Tests**: Add tests for all new functionality
6. **Update Documentation**: Keep the README and docstrings up to date
7. **Document Lessons Learned**: Add to LESSONS_LEARNED.md for future reference

## License

[MIT License](LICENSE)

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

## Complete Pipeline

The pipeline feature automates the entire process from repository extraction to model training:

```python
from agent_tools.dualipa.pipeline import run_pipeline

# Run full pipeline
stats = run_pipeline(
    repo_path="https://github.com/username/repo",
    output_dir="output_directory",
    extract_kwargs={"max_files": 1000},
    format_kwargs={"use_llm": True, "max_pairs_per_item": 10},
    run_extract=True,
    run_format=True,
    run_train=False  # Set to True for model training
)

# Print statistics
print(f"Extracted {stats['extract'].get('total_files', 0)} files")
print(f"Generated {stats['format'].get('total_qa_pairs', 0)} QA pairs")
```
