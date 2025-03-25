"""
Configuration for validation integration.

This module contains configuration options for validation.
"""

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "min_content_length": 50,
    "max_content_length": 10000,
    "max_empty_blocks_percent": 0.05,
    "max_orphaned_blocks_percent": 0.02,
    "max_missing_metadata_percent": 0.1
}

# Required fields by block type
REQUIRED_FIELDS = {
    "doc_section": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "parent_uuid"],
    "code_block": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "parent_uuid"],
    "documentation": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "child_uuids"]
}

# Required metadata by block type
REQUIRED_METADATA = {
    "doc_section": ["doc_type", "section_hierarchy", "source_url"],
    "code_block": ["language", "source_file"],
    "documentation": ["doc_type", "source_url"]
}
