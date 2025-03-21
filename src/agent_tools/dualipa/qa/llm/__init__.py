"""LLM integration for QA generation.

This module provides LLM-powered functionality for generating QA pairs.

Official documentation:
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

from .generation import (
    generate_qa_pairs_with_temperature,
    iterate_temperatures,
    generate_code_qa_pairs,
    generate_markdown_qa_pairs
)
from .reversal import (
    generate_reversed_pair,
    generate_reversed_qa_pairs
)
from .retry_llm_call import retry_llm_call

__all__ = [
    "generate_qa_pairs_with_temperature",
    "iterate_temperatures",
    "generate_code_qa_pairs",
    "generate_markdown_qa_pairs",
    "generate_reversed_pair",
    "generate_reversed_qa_pairs",
    "retry_llm_call"
]