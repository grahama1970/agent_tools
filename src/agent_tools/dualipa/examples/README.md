# DuaLipa Examples

This directory contains working examples demonstrating how to use the DuaLipa code extraction functionality correctly.

## Directory Structure

- `basic_extraction/`: Contains examples for each extraction function
  - `extract_repository_example.py`: Shows how to extract files and blocks from a repository
  - `extract_python_blocks_example.py`: Shows how to extract Python functions and classes
  - `extract_markdown_blocks_example.py`: Shows how to extract Markdown sections
  - `extract_js_ts_blocks_example.py`: Shows how to extract JavaScript/TypeScript functions and classes
- `sample_files/`: Contains sample files for use with the examples
  - `sample_python.py`: Example Python file with various code structures
  - `sample_markdown.md`: Example Markdown file with sections and code blocks
  - `sample_javascript.js`: Example JavaScript file with functions and classes

## Running the Examples

Run any example using Python:

```bash
cd /path/to/agent_tools
python -m src.agent_tools.dualipa.examples.basic_extraction.extract_python_blocks_example
```

## Common Requirements

All extraction functions require:

1. `file_path` parameter must be a `Path` object (not a string)
2. `stats` dictionary must be initialized with required keys:
   ```python
   stats = {
       "code_blocks": 0,
       "errors": [],
       "file_blocks": {}
   }
   ```

## More Documentation

For more detailed information:

- See `docs/module_relationships.md` for information about module relationships
- See `docs/function_reference.md` for detailed function parameter information 