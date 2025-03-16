"""
Utility functions for the cursor_rules package.
"""

from .rule_search import (
    search_for_user_query,
    search_for_reasoning_task,
    get_related_rules,
    RuleSearchResult,
    format_rules_for_agent
)

from .logging import (
    logger,
    configure_logger,
    get_module_logger,
    log_db_operation
)

__all__ = [
    # Rule search utilities
    'search_for_user_query',
    'search_for_reasoning_task',
    'get_related_rules',
    'RuleSearchResult',
    'format_rules_for_agent',
    
    # Logging utilities
    'logger',
    'configure_logger',
    'get_module_logger',
    'log_db_operation'
]
