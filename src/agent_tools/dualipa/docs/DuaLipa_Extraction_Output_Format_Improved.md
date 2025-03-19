
# DuaLipa Extraction Output Format

This document defines the standardized output format for the DuaLipa extraction process, specifically optimized for generating high-quality question-answer pairs for training Unsloth LoRA adapters.

## Purpose

The extraction format is designed to:
1. Preserve hierarchical relationships between code and documentation sections using UUIDs.
2. Provide rich and explicit context for LLM-based Q&A generation.
3. Enable accurate cross-referencing between related components.
4. Support extraction from multi-language codebases.
5. Facilitate intelligent summarization and question generation.

## Updated Example Output Format

The JSON format has been optimized to include fields specifically crafted for maximum LLM comprehension, explicit hierarchy representation, reasoning steps, and extraction context.

Key Improvements:
- Explicit section roles (`section_role`)
- Clear summarization instructions (`summary_instructions`)
- Consistent hierarchical relationship UUIDs (`parent_uuid`, `child_uuids`)
- Structured reasoning in Q&A (`reasoning_steps` with "Oh wait?!" pivot moments)
- Defined `extraction_focus` to guide specific content summarization

## Example JSON snippet

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
      "type": "documentation",
      "language": "markdown",
      "title": "Project Overview",
      "section_role": "parent_section",
      "content": "# Project Overview\nThis project is a code extractor for multiple languages.",
      "breadcrumb": ["# Project Overview"],
      "parent_uuid": null,
      "child_uuids": ["93e7517c-78f7-44ce-90bc-4a32c25ad877"],
      "extraction_focus": "feature_summary",
      "summary_instructions": "Summarize child sections clearly, highlighting supported languages and extraction capabilities.",
      "qa_example": {
        "question": "What is the main purpose of this project?",
        "reasoning_steps": [
          "Review the content in the 'Project Overview' section.",
          "Identify explicit purpose stated.",
          "Confirm no additional context elsewhere. Oh wait?! Confirmed, no extra context.",
          "Final answer derived confidently."
        ],
        "answer": "It is a code extractor for multiple languages."
      }
    }
  ],
  "extraction_metadata": {
    "version": "1.0.0",
    "purpose": "Convert repository content into structured Q&A pairs to train a Unsloth LoRA adaptor.",
    "instructions_to_agent": "Precisely summarize content leveraging hierarchical and contextual metadata to generate accurate, context-aware Q&A pairs. Include explicit reasoning steps with pivots marked by 'Oh wait?!'.",
    "supported_languages": ["python", "typescript", "javascript", "markdown", "rust"],
    "extraction_focus_options": ["feature_summary", "usage_details", "implementation_logic", "configuration_details", "interface_specification"],
    "expected_output_structure": {
      "question": "string",
      "reasoning_steps": ["array of strings detailing logical reasoning and pivot points"],
      "answer": "string"
    }
  }
}
```

## Field Descriptions
- **uuid**: Universally unique identifier for querying hierarchical relations in databases.
- **parent_uuid, child_uuids**: Track hierarchical relationships explicitly.
- **section_role**: Explicitly indicates the role of the section within the hierarchy.
- **extraction_focus**: Indicates the main area to guide LLM extraction tasks.
- **summary_instructions**: Explicit summarization guidance for parent nodes.
- **qa_example**: Clearly defined example question-answer pairs with reasoning steps.

## Usage Guidance
- Always preserve explicit hierarchical relationships using UUIDs.
- Generate summaries for parent sections leveraging child content clearly.
- Maintain explicit reasoning steps, including "Oh wait?!" pivots to demonstrate reasoning depth.
- Align extraction tasks explicitly according to defined extraction_focus.

