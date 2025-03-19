# DuaLipa Extraction Output Format

This document defines the standardized output format for the DuaLipa extraction process, specifically designed for converting cloned GitHub repository content into structured question-answer pairs used for training Unsloth LoRA adapters.

## Extraction Workflow

The DuaLipa extraction module follows these steps:

1. **Repository Cloning:**  
   Takes a user-provided GitHub repository URL as input and clones the repository.

2. **Content Parsing:**  
   Processes the cloned repository by parsing both code and documentation files.

3. **Structured Extraction:**  
   Converts parsed content into a structured JSON format, explicitly capturing hierarchical relationships, context, and metadata.

4. **Q&A Pair Generation:**  
   Utilizes the structured JSON to generate accurate and context-aware question-answer pairs, including explicit reasoning steps for clarity.

## Purpose

The extraction format is explicitly designed to:

1. **Preserve Hierarchical Relationships**  
   Clearly indicate section relationships using explicit UUID identifiers (`uuid`, `parent_uuid`, `child_uuids`).

2. **Provide Rich Context**  
   Mark content characteristics explicitly (`content_flags`) and clearly define extraction focus (`extraction_focus`) to support accurate Q&A generation.

3. **Enable Accurate Cross-referencing**  
   Use breadcrumb paths (`breadcrumb`) and hierarchical indicators (`depth`, `header_depth`) to accurately cross-reference related content sections.

4. **Support Multi-language Codebases**  
   Facilitate extraction and summarization from repositories containing Python, TypeScript, JavaScript, Markdown, and Rust content.

5. **Facilitate Intelligent Summarization and Question Generation**  
   Provide explicit summarization instructions (`summary_instructions`) for parent sections and structured reasoning steps (`reasoning_steps`) in question-answer pairs, clearly marking reconsideration moments ("Oh wait?!") to illustrate reasoning clarity and depth.


## Example Output Format

```json
{
  "repo_stats": {
    "source": "example-repo",
    "description": "Repository content extraction intended for generating structured question-answer pairs to train a Unsloth LoRA adapter.",
    "extraction_timestamp": {
      "start": "2023-03-20T10:00:00.000Z",
      "end": "2023-03-20T10:05:00.000Z",
      "duration_seconds": 300
    },
    "total_files": 13,
    "file_breakdown": {
      "documentation_files": 3,
      "code_files": 10
    },
    "languages": {
      "python": 5,
      "typescript": 3,
      "javascript": 1,
      "markdown": 3,
      "rust": 1
    },
    "file_types": {
      ".py": 5,
      ".ts": 2,
      ".tsx": 1,
      ".js": 1,
      ".md": 3,
      ".rs": 1
    }
  },
  "sections": [
    {
      "uuid": "6d2f5c78-9eb4-4c28-927a-f0b32c2e7649",
      "id": "readme_project_overview",
      "type": "documentation",
      "language": "markdown",
      "title": "Project Overview",
      "content": "# Project Overview\n\nThis project is a code extractor for multiple languages.",
      "file_path": "src/docs/readme.md",
      "breadcrumb": ["# Project Overview"],
      "parent_uuid": null,
      "child_uuids": ["93e7517c-78f7-44ce-90bc-4a32c25ad877"],
      "depth": 0,
      "header_depth": [1],
      "content_flags": {
        "has_code_block": false,
        "has_table": false,
        "has_links": false,
        "has_image": false,
        "has_list": false
      },
      "toc_format": "Project Overview",
      "section_role": "parent_section",
      "extraction_focus": "feature_summary",
      "summary_instructions": "Summarize child sections, clearly highlighting supported languages and extraction capabilities.",
      "qa_example": {
        "question": "What is the main purpose of this project?",
        "reasoning_steps": [
          "Check content: explicitly describes the project as 'a code extractor for multiple languages.'",
          "Confirm no other purpose described in child sections. Oh wait?! Double-check child sections again—no further purpose listed.",
          "Conclude confidently based on available content."
        ],
        "answer": "It is a code extractor for multiple languages."
      }
    },
    {
      "uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
      "id": "readme_features",
      "type": "documentation",
      "language": "markdown",
      "title": "Features",
      "content": "## Features\n- Supports Python, JavaScript, and TypeScript\n- Extracts code blocks and documentation",
      "file_path": "src/docs/readme.md",
      "breadcrumb": ["# Project Overview", "## Features"],
      "parent_uuid": "6d2f5c78-9eb4-4c28-927a-f0b32c2e7649",
      "child_uuids": [],
      "depth": 1,
      "header_depth": [1, 2],
      "content_flags": {
        "has_list": true
      },
      "toc_format": "    Features",
      "section_role": "child_section",
      "extraction_focus": "feature_summary",
      "qa_example": {
        "question": "What features does this project offer?",
        "reasoning_steps": [
          "Examine bullet points explicitly mentioning supported languages: Python, JavaScript, TypeScript.",
          "Explicit mention of 'extracts code blocks and documentation' also provided clearly.",
          "No additional context found, confirming features listed."
        ],
        "answer": "It supports Python, JavaScript, and TypeScript and extracts code blocks and documentation."
      }
    },
    {
      "uuid": "7b9e12d4-3f8a-4c91-b76d-8f3c9b9a2e5d",
      "id": "extractor_extract_blocks",
      "type": "code",
      "language": "python",
      "title": "extract_blocks Function",
      "content": "def extract_blocks(content: str) -> List[Block]:\n    '''Extract code blocks from content.'''\n    blocks = []\n    # Implementation\n    return blocks",
      "file_path": "src/extractor.py",
      "breadcrumb": ["src/extractor.py", "extract_blocks"],
      "parent_uuid": null,
      "child_uuids": [],
      "dependencies": {
        "imports": ["from typing import List"],
        "referenced_types": ["Block"]
      },
      "test_coverage": {
        "test_file": "tests/test_extractor.py",
        "coverage_percentage": 100
      },
      "version_history": {
        "last_modified": "2024-03-19T15:30:00Z",
        "last_modified_by": "developer@example.com",
        "commit_hash": "abc123def456",
        "commit_message": "Add extract_blocks function"
      },
      "qa_generation": {
        "difficulty_levels": ["basic", "intermediate"],
        "knowledge_prerequisites": ["Python", "Type hints"],
        "focus_areas": ["function signature", "docstring"],
        "qa_examples": [
          {
            "difficulty": "basic",
            "question": "What does the extract_blocks function do?",
            "reasoning_steps": ["Function name implies block extraction", "Docstring explicitly confirms extraction of code blocks"],
            "answer": "It extracts code blocks from given content and returns a list of Block objects."
          }
        ]
      }
    }
  ],
  "extraction_metadata": {
    "version": "1.0.0",
    "purpose": "Convert repository content into structured Q&A pairs for training a Unsloth LoRA adapter.",
    "instructions_to_agent": "Leverage UUID hierarchical relationships explicitly to accurately summarize content from child sections. Generate clear question-answer pairs, explicitly stating reasoning steps with reconsideration ('Oh wait?!') clearly marked to illustrate thought pivots.",
    "supported_languages": ["python", "typescript", "javascript", "markdown", "rust"],
    "extraction_focus_options": ["feature_summary", "usage_details", "implementation_logic", "configuration_details", "interface_specification"],
    "expected_output_structure": {
      "question": "string",
      "reasoning_steps": ["array of reasoning steps"],
      "answer": "string"
    }
  }
}
```

## Field Descriptions

### Repository Statistics (`repo_stats`)
- `source`: Repository identifier
- `description`: Purpose of the extraction
- `extraction_timestamp`: Timing information
- `file_breakdown`: File type statistics
- `languages`: Language distribution
- `file_types`: File extension statistics

### Sections
Each section represents a discrete unit of code or documentation:

#### Common Fields
- `uuid`: Unique identifier
- `id`: Human-readable identifier
- `type`: "code" or "documentation"
- `language`: Programming or markup language
- `title`: Section title
- `content`: Actual content
- `file_path`: Source file location
- `breadcrumb`: Path in document hierarchy
- `parent_uuid`/`child_uuids`: Hierarchical relationships

#### Documentation-Specific Fields
- `content_flags`: Content type indicators
- `toc_format`: Table of contents representation
- `section_role`: Role in document hierarchy
- `extraction_focus`: Content emphasis for Q&A
- `summary_instructions`: Guidance for summarization

#### Code-Specific Fields
- `dependencies`: Import and reference information
- `complexity_metrics`: Code complexity measures
- `referenced_by`: Usage tracking
- `test_coverage`: Testing information
- `version_history`: Git metadata
- `equivalent_implementations`: Cross-language implementations
- `protocol_conformance`: Design pattern adherence

#### Q&A Generation Support
- `qa_generation`: Structured guidance for Q&A generation
  - `difficulty_levels`: Question complexity categories
  - `knowledge_prerequisites`: Required background
  - `focus_areas`: Key aspects to cover
  - `qa_examples`: Sample questions and answers
  - `follow_up_questions`: Related questions to consider

### Metadata
- `version`: Format version
- `purpose`: Extraction goal
- `instructions_to_agent`: Q&A generation guidance
- `supported_languages`: Supported languages
- `extraction_settings`: Configuration parameters

## Usage Notes

1. Always preserve hierarchical relationships using UUIDs
2. Include sufficient context for Q&A generation
3. Maintain cross-references between related sections
4. Provide clear extraction focus and summary instructions
5. Include example Q&A pairs where possible 