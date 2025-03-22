# Markdown Extraction Module

This module extracts structured content from markdown files, maintaining hierarchical relationships between sections and properly handling various content elements like tables, code blocks, and images.

## Input and Output Formats

### Input
The module processes markdown files (`.md`) and extracts structured content including:
- Section hierarchies (based on heading levels #, ##, ###, etc.)
- Tables (formatted with pipe syntax)
- Code blocks (fenced with triple backticks)
- Images (with markdown image syntax)
- Text content between other elements

### Output
Output is a JSON array of section objects where each section contains:
- `uuid`: Unique identifier for the section
- `section_hierarchy_depth`: Array showing the nested path of sections
- `title`: Section heading title
- `content`: Plain text content within the section
- `tables`: Array of table objects, each with:
  - `uuid`: Unique identifier
  - `content`: Object containing:
    - `headers`: Array of header cell values
    - `rows`: Array of arrays, each representing a row of cell values
- `images`: Array of image objects, each with:
  - `uuid`: Unique identifier
  - `src`: Image URL/path
  - `alt`: Alternative text for the image
- `code`: Array of code block objects, each with:
  - `uuid`: Unique identifier
  - `language`: Programming language (extracted from code fence)
  - `content`: Code content
- `tests`: Array of test blocks (optional, for specific documentation)

## Example

### Input (markdown file)
```markdown
# DeepSeek Usage

SGLang provides several optimizations specifically designed for the DeepSeek model to boost its inference speed. This document outlines current optimizations for DeepSeek.

## Launch DeepSeek V3 with SGLang

SGLang is recognized as one of the top engines for [DeepSeek model inference](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3). To run DeepSeek V3/R1 models, the requirements are as follows:

| Weight Type | Configuration |
|-------------|---------------|
| **Full precision FP8**<br>*(recommended)* | 8 x H200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |

### Download Weights

If you encounter errors when starting the server, ensure the weights have finished downloading.
```

### Output (JSON)
```json
[
  {
    "uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "section_hierarchy_depth": ["DeepSeek Usage"],
    "title": "DeepSeek Usage",
    "content": "SGLang provides several optimizations specifically designed for the DeepSeek model to boost its inference speed. This document outlines current optimizations for DeepSeek.",
    "images": [],
    "tests": [],
    "tables": [],
    "code": []
  },
  {
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
    "section_hierarchy_depth": [
      "DeepSeek Usage",
      "Launch DeepSeek V3 with SGLang"
    ],
    "title": "Launch DeepSeek V3 with SGLang",
    "content": "SGLang is recognized as one of the top engines for [DeepSeek model inference](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3). To run DeepSeek V3/R1 models, the requirements are as follows:",
    "images": [],
    "tests": [],
    "tables": [
      {
        "uuid": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "content": {
          "headers": ["Weight Type", "Configuration"],
          "rows": [
            ["**Full precision FP8**<br>*(recommended)*", "8 x H200"],
            ["", "8 x MI300X"],
            ["", "2 x 8 x H100/800/20"]
          ]
        }
      }
    ],
    "code": []
  },
  {
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "section_hierarchy_depth": [
      "DeepSeek Usage",
      "Launch DeepSeek V3 with SGLang",
      "Download Weights"
    ],
    "title": "Download Weights",
    "content": "If you encounter errors when starting the server, ensure the weights have finished downloading.",
    "images": [],
    "tests": [],
    "tables": [],
    "code": []
  }
]
```

## File Structure

- `extraction_blocks.py`: Core extraction logic for identifying markdown elements
- `qa_formatter.py`: Formats extracted blocks into the expected output structure
- `real_world_test.py`: Test framework for running extraction on sample repositories
- `validation.py`: Validates the extraction output structure

## Processing Flow

1. **File Identification**: Locate markdown files in the repository
2. **Section Extraction**: Parse headings to identify sections and their hierarchical relationships
3. **Element Extraction**: Extract tables, code blocks, images, and text from each section
4. **Position Tracking**: Maintain element order using character positions
5. **Hierarchy Mapping**: Create section hierarchy paths for nested sections
6. **Format Transformation**: Convert internal block representation to the expected output format
7. **Validation**: Verify the output structure meets the requirements

## Usage

```python
from pathlib import Path
from extraction_blocks import extract_all_blocks
from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output

# Path to repository or directory containing markdown files
repo_path = Path("./test_repos/sglang")

# Extract blocks from all files
blocks = extract_all_blocks(repo_path)

# Convert to QA-compatible format
qa_blocks = create_qa_compatible_blocks(blocks)

# Create final output
output = create_qa_compatible_output(qa_blocks)

# Write to JSON file
import json
with open("output.json", "w") as f:
    json.dump(output, f, indent=2)
```

## Implementation Notes

### 1. Section Hierarchy
- Sections are identified by heading markers (#, ##, ###, etc.)
- Nested sections are determined by heading level (e.g., ### is child of ##)
- Full hierarchy path is stored in `section_hierarchy_depth` for each section

### 2. Table Extraction
- Tables are identified using markdown pipe syntax
- Header row is separated from data rows
- Empty cells are preserved as empty strings
- Markdown formatting within cells is preserved

### 3. Code Block Extraction
- Code blocks are identified using triple backtick fences
- Language is extracted from the opening fence if specified
- Code content is preserved including indentation and empty lines

### 4. Image Extraction
- Images use standard markdown syntax: `![alt text](image_url)`
- Both the URL and alt text are extracted
- Images are associated with their parent section

### 5. Position Tracking
- Character positions are tracked to maintain original element order
- Ensures output elements appear in the same order as in the source document