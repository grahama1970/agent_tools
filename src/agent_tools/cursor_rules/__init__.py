"""
Cursor Rules - A tool for managing and searching coding rules and patterns.
"""
import sys
from pathlib import Path

# Import logger from the utils package
from agent_tools.cursor_rules.utils.logging import logger, configure_logger

# Configure the logger with default settings
configure_logger()

from agent_tools.cursor_rules.core import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    bm25_keyword_search,
    semantic_search,
    hybrid_search,
    query_by_rule_number,
    query_by_title,
    query_by_description,
    EMBEDDING_AVAILABLE,
    # New imports from reorganization
    CursorRules,
    CursorRulesDatabase,
    Rule,
    RuleExample,
    SearchResult,
    QueryOptions,
    RuleFilter,
    get_cursor_rules
)

from agent_tools.cursor_rules.cli import cli

# Import embedding functionality
from agent_tools.cursor_rules.embedding import (
    create_embedding,
    create_embedding_sync,
    create_embedding_with_sentence_transformer,
    ensure_text_has_prefix
)

__version__ = "0.1.0"

__all__ = [
    # Core functionality
    'setup_cursor_rules_db',
    'get_all_rules',
    'get_examples_for_rule',
    'bm25_keyword_search',
    'semantic_search',
    'hybrid_search',
    'query_by_rule_number',
    'query_by_title',
    'query_by_description',
    'EMBEDDING_AVAILABLE',
    'cli',
    
    # Add new exports from reorganization
    'CursorRules',
    'CursorRulesDatabase',
    'Rule',
    'RuleExample',
    'SearchResult',
    'QueryOptions',
    'RuleFilter',
    'get_cursor_rules',
    
    # Logger
    'logger',
    
    # Embedding functionality
    'create_embedding',
    'create_embedding_sync',
    'create_embedding_with_sentence_transformer',
    'ensure_text_has_prefix'
]
