"""
Configuration for QA system integration.

This module contains configuration options for QA system integration.
"""

# QA format options
QA_FORMAT_OPTIONS = {
    "include_metadata": True,
    "flatten_hierarchy": False,
    "max_content_length": 10000
}

# Question generation options
QUESTION_GENERATION_OPTIONS = {
    "question_types": ["definition", "explanation", "example", "comparison"],
    "max_questions_per_section": 3,
    "min_content_length": 100,
    "excluded_section_types": ["image", "table", "code_block"]
}
