"""Utilities for QA generation.

This module provides various utility functions for QA generation, including:
- Security utilities for input sanitization
- Validation functions for QA pairs and responses
- Deduplication tools for removing duplicate QA pairs
- Method validation for ensuring function correctness
- Caching mechanisms for LLM requests

Official documentation:
- bleach: https://bleach.readthedocs.io/
- sentence-transformers: https://sbert.net/docs/
- inspect: https://docs.python.org/3/library/inspect.html
- typing: https://docs.python.org/3/library/typing.html
- diskcache: https://grantjenks.com/docs/diskcache/
- hashlib: https://docs.python.org/3/library/hashlib.html
"""

from .security import sanitize_input, sanitize_input_json, check_pii_in_content
from .validation import (
    validate_qa_pair, validate_input_json, 
    validate_qa_response, validate_temperature_range
)
from .deduplication import deduplicate_qa_pairs, exact_deduplicate, semantic_deduplicate
from .method_validator import verify_async_function, validate_dedent_usage, verify_methods
from .cache import (
    initialize_cache, get_from_cache, add_to_cache, 
    compute_cache_key, get_cache_stats, cache_hit_rate, 
    clear_cache
)

# Initialize cache on import
initialize_cache()

__all__ = [
    "sanitize_input", "sanitize_input_json", "check_pii_in_content",
    "validate_qa_pair", "validate_input_json", "validate_qa_response", 
    "validate_temperature_range", "deduplicate_qa_pairs", "exact_deduplicate",
    "semantic_deduplicate",
    "verify_async_function", "validate_dedent_usage", "verify_methods",
    "initialize_cache", "get_from_cache", "add_to_cache", "compute_cache_key", 
    "get_cache_stats", "cache_hit_rate", "clear_cache"
]