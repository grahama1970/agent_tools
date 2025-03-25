# Extraction Validation and Frictionless Collaboration

This document provides guidelines for validating extraction results and collaborating effectively when working with the DuaLipa extraction system.

## Core Principles

1. **Complete Information**: Always present complete extraction structures
2. **Consistency**: Maintain consistent format across all extractions
3. **Easy Verification**: Provide simple ways to verify extraction results
4. **Reference Implementation**: Use standardized examples as reference

## Extraction Output Format

All extraction outputs must follow this standardized format:

### Code Block Format

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "type": "function",
  "name": "process_data",
  "content": "def process_data(input):\n    return input.upper()",
  "language": "python",
  "file_path": "src/main.py",
  "start_line": 10,
  "end_line": 12,
  "parent_uuid": "550e8400-e29b-41d4-a716-446655441111",
  "child_uuids": [],
  "metadata": {
    "language": "python",
    "doc_string": "Process the input data by converting to uppercase",
    "has_docstring": true,
    "arguments": ["input"],
    "returns": ["processed_string"]
  }
}
```

### Documentation Block Format

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655442222",
  "type": "doc_section",
  "name": "API Reference",
  "content": "# API Reference\n\nThis section describes...",
  "language": "markdown",
  "file_path": "docs/api.md",
  "parent_uuid": "550e8400-e29b-41d4-a716-446655443333",
  "child_uuids": ["550e8400-e29b-41d4-a716-446655444444"],
  "metadata": {
    "doc_type": "markdown",
    "section_hierarchy": ["Documentation", "API Reference"],
    "header_level": 1,
    "source_url": "https://example.com/docs/api"
  }
}
```

## Frictionless Collaboration Tools

### 1. Quick Extraction for Validation

Validate any code snippet by using the `extract_snippet` function:

```bash
python extract_snippet.py --language python --snippet "def hello():\n    return 'world'"
```

This will output a complete JSON object with all required fields that you can verify.

### 2. Markdown Extraction Validation

For markdown content:

```bash
python extract_markdown.py --content "# Title\n\nContent here" --output extraction.json
```

### 3. Documentation Extraction Validation

For documentation URLs:

```bash
python extract_docs.py --url "https://docs.example.com/api" --output docs_extraction.json
```

## Extraction Conversion Functions

### Python Code Example

If you provide the Python code:

```python
def calculate_area(length, width):
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
        
    Returns:
        The area as length * width
    """
    return length * width
```

The extraction system should return:

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655445555",
  "type": "function",
  "name": "calculate_area",
  "content": "def calculate_area(length, width):\n    \"\"\"Calculate the area of a rectangle.\n    \n    Args:\n        length: The length of the rectangle\n        width: The width of the rectangle\n        \n    Returns:\n        The area as length * width\n    \"\"\"\n    return length * width",
  "language": "python",
  "file_path": "snippet.py",
  "start_line": 1,
  "end_line": 10,
  "parent_uuid": null,
  "child_uuids": [],
  "metadata": {
    "language": "python",
    "doc_string": "Calculate the area of a rectangle.\n\nArgs:\n    length: The length of the rectangle\n    width: The width of the rectangle\n    \nReturns:\n    The area as length * width",
    "has_docstring": true,
    "arguments": ["length", "width"],
    "returns": ["area"]
  }
}
```

### Markdown Example

If you provide the markdown:

```markdown
# Usage Instructions

## Installation

Install using pip:

```bash
pip install dualipa
```

## Configuration

Configure using a config file:

```python
from dualipa import config

config.setup('config.json')
```
```

The extraction system should return:

```json
[
  {
    "uuid": "550e8400-e29b-41d4-a716-446655446666",
    "type": "section",
    "name": "Usage Instructions",
    "content": "# Usage Instructions",
    "language": "markdown",
    "file_path": "snippet.md",
    "start_line": 1,
    "end_line": 1,
    "parent_uuid": null,
    "child_uuids": ["550e8400-e29b-41d4-a716-446655447777", "550e8400-e29b-41d4-a716-446655448888"],
    "metadata": {
      "doc_type": "markdown",
      "section_hierarchy": ["Usage Instructions"],
      "header_level": 1
    }
  },
  {
    "uuid": "550e8400-e29b-41d4-a716-446655447777",
    "type": "section",
    "name": "Installation",
    "content": "## Installation\n\nInstall using pip:",
    "language": "markdown",
    "file_path": "snippet.md",
    "start_line": 3,
    "end_line": 5,
    "parent_uuid": "550e8400-e29b-41d4-a716-446655446666",
    "child_uuids": ["550e8400-e29b-41d4-a716-446655449999"],
    "metadata": {
      "doc_type": "markdown",
      "section_hierarchy": ["Usage Instructions", "Installation"],
      "header_level": 2
    }
  },
  {
    "uuid": "550e8400-e29b-41d4-a716-446655449999",
    "type": "code_block",
    "name": "Installation code",
    "content": "pip install dualipa",
    "language": "bash",
    "file_path": "snippet.md",
    "start_line": 7,
    "end_line": 9,
    "parent_uuid": "550e8400-e29b-41d4-a716-446655447777",
    "child_uuids": [],
    "metadata": {
      "doc_type": "markdown",
      "code_type": "bash"
    }
  },
  {
    "uuid": "550e8400-e29b-41d4-a716-446655448888",
    "type": "section",
    "name": "Configuration",
    "content": "## Configuration\n\nConfigure using a config file:",
    "language": "markdown",
    "file_path": "snippet.md",
    "start_line": 11,
    "end_line": 13,
    "parent_uuid": "550e8400-e29b-41d4-a716-446655446666",
    "child_uuids": ["550e8400-e29b-41d4-a716-446655450000"],
    "metadata": {
      "doc_type": "markdown",
      "section_hierarchy": ["Usage Instructions", "Configuration"],
      "header_level": 2
    }
  },
  {
    "uuid": "550e8400-e29b-41d4-a716-446655450000",
    "type": "code_block",
    "name": "Configuration code",
    "content": "from dualipa import config\n\nconfig.setup('config.json')",
    "language": "python",
    "file_path": "snippet.md",
    "start_line": 15,
    "end_line": 19,
    "parent_uuid": "550e8400-e29b-41d4-a716-446655448888",
    "child_uuids": [],
    "metadata": {
      "doc_type": "markdown",
      "code_type": "python"
    }
  }
]
```

## Implementation for Frictionless Collaboration

To implement frictionless validation in your workflow:

1. **Use the Conversion API**: Convert any code or documentation snippet:
   ```python
   from dualipa.extraction.convert import code_to_json, markdown_to_json
   
   # Convert code snippet
   json_output = code_to_json("def hello(): return 'world'", language="python")
   print(json_output)
   
   # Convert markdown snippet
   json_output = markdown_to_json("# Title\n\nContent")
   print(json_output)
   ```

2. **Implement the Quick Extract CLI**: For command-line usage:
   ```bash
   # Extract from code snippet
   python -m dualipa.extraction.quick_extract --code "def hello(): return 'world'" --language python
   
   # Extract from file
   python -m dualipa.extraction.quick_extract --file path/to/file.py
   
   # Extract from URL
   python -m dualipa.extraction.quick_extract --url https://example.com/docs
   ```

3. **Use the Validation Tools**: Validate the extraction output:
   ```bash
   python -m dualipa.extraction.validate --input extraction.json
   ```

## Command-Line Reference

### Extraction Commands

```bash
# Extract from code snippet
python -m dualipa.extraction.quick_extract --code "def hello(): return 'world'" --language python

# Extract from file
python -m dualipa.extraction.quick_extract --file path/to/file.py

# Extract from URL with Playwright support
python -m dualipa.extraction.quick_extract --url https://example.com/docs --playwright
```

### Validation Commands

```bash
# Validate extraction output
python -m dualipa.extraction.validate --input extraction.json

# Compare to expected format
python -m dualipa.extraction.validate --input extraction.json --expected expected.json
```

### Visualization Commands

```bash
# Generate HTML visualization
python -m dualipa.extraction.visualize --input extraction.json --output visualization.html
```

## Conclusion

By following these guidelines and using the provided tools, you can ensure frictionless collaboration when working with the DuaLipa extraction system. The standardized format and validation tools make it easy to verify extraction results and share them with others.