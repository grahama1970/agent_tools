"""
Core functionality for the Cursor Rules package.

This module provides the core database and rule management functionality
for the Cursor Rules system.
"""

from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    bm25_keyword_search,
    semantic_search,
    hybrid_search,
    query_by_rule_number,
    query_by_title,
    query_by_description,
    EMBEDDING_AVAILABLE
)

from agent_tools.cursor_rules.core.db import get_db

from agent_tools.cursor_rules.core.ai_knowledge_db import (
    setup_ai_knowledge_db,
    load_schema,
    get_schema_doc,
)

from agent_tools.cursor_rules.core.enhanced_db import (
    setup_enhanced_cursor_rules_db,
    multi_hop_rule_discovery,
    knowledge_path_between_resources,
    rule_complete_context,
    hybrid_cross_collection_search,
    contextual_recommendation,
)

from agent_tools.cursor_rules.core.pydantic_models import (
    CursorRules,
    CursorRulesDatabase,
    Rule,
    RuleExample,
    SearchResult,
    QueryOptions,
    RuleFilter
)

from agent_tools.cursor_rules.core.model_connector import (
    get_cursor_rules,
    setup_cursor_rules,
    search_rules,
    get_rule
)

__all__ = [
    # From cursor_rules.py
    "setup_cursor_rules_db",
    "get_all_rules",
    "get_examples_for_rule",
    "bm25_keyword_search",
    "semantic_search",
    "hybrid_search",
    "query_by_rule_number",
    "query_by_title",
    "query_by_description",
    "EMBEDDING_AVAILABLE",
    
    # From db.py
    "get_db",
    
    # From ai_knowledge_db.py
    "setup_ai_knowledge_db",
    "load_schema",
    "get_schema_doc",
    
    # From enhanced_db.py
    "setup_enhanced_cursor_rules_db",
    "multi_hop_rule_discovery",
    "knowledge_path_between_resources",
    "rule_complete_context",
    "hybrid_cross_collection_search",
    "contextual_recommendation",
    
    # From pydantic_models.py
    "CursorRules",
    "CursorRulesDatabase",
    "Rule",
    "RuleExample",
    "SearchResult",
    "QueryOptions",
    "RuleFilter",
    
    # From model_connector.py
    "get_cursor_rules",
    "setup_cursor_rules",
    "search_rules",
    "get_rule"
] 