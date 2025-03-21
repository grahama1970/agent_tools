"""
GitHub utilities module for DuaLipa.

This module provides functionality for interacting with GitHub repositories,
including repository downloading and content extraction.

Key Features:
1. Repository cloning and management
2. Metadata extraction
3. File operations
4. Error handling and retries

Dependencies:
- git: For repository operations
- requests: For GitHub API calls
- loguru: For logging

Related Files:
- repo_utils.py: Core repository operations
- api_utils.py: GitHub API interactions
"""

from .repo_utils import (
    clone_repository,
    verify_repo_structure,
    extract_repository
)
from .api_utils import fetch_repo_metadata, get_file_content

__all__ = [
    'clone_repository',
    'verify_repo_structure',
    'extract_repository',
    'fetch_repo_metadata',
    'get_file_content'
]
