"""
Connector module to bridge between Pydantic models and functional API.

This module provides helper functions that connect the object-oriented Pydantic
models with the functional implementation in cursor_rules.py.
"""

import asyncio
from typing import Dict, Any, List, Union, Optional, Tuple

from agent_tools.cursor_rules.core.pydantic_models import (
    CursorRules,
    CursorRulesDatabase,
    Rule,
    RuleExample,
    SearchResult,
    QueryOptions
)

from agent_tools.cursor_rules.core.cursor_rules import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    bm25_keyword_search,
    semantic_search,
    hybrid_search,
    query_by_rule_number,
    query_by_title,
    query_by_description
)


def get_cursor_rules(
    hosts: Union[str, List[str]] = ["http://localhost:8529"],
    username: str = "root",
    password: str = "openSesame",
    database_name: str = "cursor_rules",
    rules_dir: Optional[str] = None
) -> CursorRules:
    """
    Create and initialize a CursorRules instance.
    
    Args:
        hosts: ArangoDB host(s)
        username: ArangoDB username
        password: ArangoDB password
        database_name: Name of the database
        rules_dir: Path to directory containing rule files
        
    Returns:
        Configured CursorRules instance
    """
    db_config = CursorRulesDatabase(
        hosts=hosts,
        username=username,
        password=password,
        database_name=database_name
    )
    db_config.connect()
    
    rules = CursorRules(
        database=db_config,
        rules_dir=rules_dir
    )
    
    # Attach function implementations to the model
    rules.setup = lambda: setup_cursor_rules(
        hosts, username, password, database_name, rules_dir
    )
    
    rules.search = lambda query, search_type="hybrid", limit=5, verbose=False: search_rules(
        rules.database.db, query, search_type, limit, verbose
    )
    
    rules.get_rule = lambda rule_id: get_rule(rules.database.db, rule_id)
    
    return rules


def setup_cursor_rules(
    hosts: Union[str, List[str]] = ["http://localhost:8529"],
    username: str = "root",
    password: str = "openSesame",
    database_name: str = "cursor_rules",
    rules_dir: Optional[str] = None
) -> Any:
    """
    Set up cursor rules database and return the connection.
    
    Args:
        hosts: ArangoDB host(s)
        username: ArangoDB username
        password: ArangoDB password
        database_name: Name of the database
        rules_dir: Path to directory containing rule files
        
    Returns:
        Database connection
    """
    config = {
        "arango": {
            "hosts": hosts,
            "username": username,
            "password": password
        }
    }
    return setup_cursor_rules_db(config, rules_dir, database_name)


async def search_rules(
    db: Any,
    query: str,
    search_type: str = "hybrid",
    limit: int = 5,
    verbose: bool = False
) -> List[SearchResult]:
    """
    Search for rules based on query text.
    
    Args:
        db: Database connection
        query: Search query
        search_type: Type of search ('bm25', 'semantic', or 'hybrid')
        limit: Maximum number of results
        verbose: Whether to print detailed information
        
    Returns:
        List of search results
    """
    if search_type == "bm25":
        results = await bm25_keyword_search(db, query, limit=limit, verbose=verbose)
    elif search_type == "semantic":
        results = semantic_search(db, query, limit=limit, verbose=verbose)
    else:  # hybrid search (default)
        results = await hybrid_search(db, query, limit=limit, verbose=verbose)
    
    # Convert to SearchResult objects
    if not results:
        return []
    
    search_results = []
    for result in results:
        # Different search types return different result formats
        if isinstance(result, tuple) and len(result) == 2:
            rule_data, score = result
        elif isinstance(result, dict) and "rule" in result and "score" in result:
            rule_data, score = result["rule"], result["score"]
        else:
            continue
        
        # Create Rule object if we got valid data
        try:
            rule = Rule(**rule_data)
            search_result = SearchResult(rule=rule, score=score)
            search_results.append(search_result)
        except Exception as e:
            print(f"Error creating SearchResult: {e}")
            continue
    
    return search_results


async def get_rule(db: Any, rule_id: str) -> Optional[Rule]:
    """
    Get a rule by its ID or rule number.
    
    Args:
        db: Database connection
        rule_id: Rule ID, number, title, or description
        
    Returns:
        Rule object if found, None otherwise
    """
    # Try as rule number first
    rule_data = await query_by_rule_number(db, rule_id)
    
    if not rule_data:
        # Try as title
        rule_data = await query_by_title(db, rule_id)
    
    if not rule_data:
        # Try as description
        rule_data = await query_by_description(db, rule_id)
    
    if rule_data:
        try:
            return Rule(**rule_data)
        except Exception as e:
            print(f"Error creating Rule object: {e}")
    
    return None 