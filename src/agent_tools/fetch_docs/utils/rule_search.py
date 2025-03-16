#!/usr/bin/env python3
"""
Rule Search Module

This module provides functions to search for rules related to user queries or reasoning tasks.
It uses hybrid search (combining BM25 and semantic search) to find the most relevant rules.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleSearchResult:
    """Class to store the results of a rule search."""
    
    def __init__(self, query: str, rules: List[Dict[str, Any]]):
        """
        Initialize a RuleSearchResult.
        
        Args:
            query: The query used to find the rules
            rules: The rules found
        """
        self.query = query
        self.rules = rules
        self.timestamp = asyncio.get_event_loop().time()
    
    def get_rule_titles(self) -> List[str]:
        """Get the titles of all rules found."""
        return [rule.get("title", "Untitled Rule") for rule in self.rules]
    
    def get_rule_descriptions(self) -> List[str]:
        """Get the descriptions of all rules found."""
        return [rule.get("description", "No description") for rule in self.rules]
    
    def get_rule_content(self) -> List[str]:
        """Get the content of all rules found."""
        return [rule.get("content", "") for rule in self.rules]
    
    def get_rule_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get a rule by its title."""
        for rule in self.rules:
            if rule.get("title") == title:
                return rule
        return None
    
    def get_rule_by_number(self, rule_number: str) -> Optional[Dict[str, Any]]:
        """Get a rule by its number."""
        for rule in self.rules:
            if rule.get("rule_number") == rule_number:
                return rule
        return None
    
    def __str__(self) -> str:
        """String representation of the search result."""
        return f"RuleSearchResult(query='{self.query}', rules={len(self.rules)})"
    
    def __repr__(self) -> str:
        """Representation of the search result."""
        return self.__str__()

async def search_related_rules(db, query: str, limit: int = 5) -> RuleSearchResult:
    """
    Search for rules related to a query using hybrid search.
    
    Args:
        db: Database handle
        query: The search query
        limit: Maximum number of results
        
    Returns:
        RuleSearchResult: The search results
    """
    from agent_tools.cursor_rules.core.cursor_rules import hybrid_search
    
    try:
        # Perform hybrid search
        logger.info(f"Searching for rules related to: {query}")
        results = await hybrid_search(db, query, collection_name="rules", limit=limit)
        
        # Extract rules from results
        rules = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                # Format from cursor_rules.py: (rule, score)
                rule = result[0]
            else:
                # Format from other search functions: {'rule': doc, 'score': score}
                rule = result.get('rule', result)
            
            rules.append(rule)
        
        logger.info(f"Found {len(rules)} rules related to: {query}")
        return RuleSearchResult(query, rules)
    
    except Exception as e:
        logger.error(f"Error searching for rules: {e}")
        return RuleSearchResult(query, [])

def format_rules_for_agent(search_result: RuleSearchResult) -> str:
    """
    Format rules for presentation to the agent.
    
    Args:
        search_result: The search results
        
    Returns:
        str: Formatted rules
    """
    if not search_result.rules:
        return "No relevant rules found."
    
    formatted = f"Rules related to '{search_result.query}':\n\n"
    
    for i, rule in enumerate(search_result.rules, 1):
        title = rule.get("title", "Untitled Rule")
        description = rule.get("description", "No description")
        rule_number = rule.get("rule_number", f"Unknown-{i}")
        
        formatted += f"Rule {rule_number}: {title}\n"
        formatted += f"Description: {description}\n"
        
        # Add content if available
        content = rule.get("content")
        if content:
            # Truncate content if it's too long
            if len(content) > 500:
                content = content[:500] + "..."
            formatted += f"Content:\n{content}\n"
        
        # Add examples if available
        examples = rule.get("examples", [])
        if examples:
            formatted += "Examples:\n"
            for example in examples:
                example_title = example.get("title", "Untitled Example")
                formatted += f"- {example_title}\n"
        
        formatted += "\n"
    
    return formatted

class RuleSearchCache:
    """Cache for rule search results."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of search results to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.timestamps = {}
    
    def add(self, query: str, result: RuleSearchResult):
        """
        Add a search result to the cache.
        
        Args:
            query: The query
            result: The search result
        """
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_query = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_query]
            del self.timestamps[oldest_query]
        
        # Add new entry
        self.cache[query] = result
        self.timestamps[query] = asyncio.get_event_loop().time()
    
    def get(self, query: str) -> Optional[RuleSearchResult]:
        """
        Get a search result from the cache.
        
        Args:
            query: The query
            
        Returns:
            RuleSearchResult or None: The search result if found, None otherwise
        """
        result = self.cache.get(query)
        if result:
            # Update timestamp
            self.timestamps[query] = asyncio.get_event_loop().time()
        return result
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.timestamps = {}

# Global cache instance
cache = RuleSearchCache()

async def get_related_rules(db, query: str, use_cache: bool = True, limit: int = 5) -> RuleSearchResult:
    """
    Get rules related to a query, using cache if available.
    
    Args:
        db: Database handle
        query: The search query
        use_cache: Whether to use the cache
        limit: Maximum number of results
        
    Returns:
        RuleSearchResult: The search results
    """
    # Check cache first if enabled
    if use_cache:
        cached_result = cache.get(query)
        if cached_result:
            logger.info(f"Using cached results for query: {query}")
            return cached_result
    
    # Perform search
    result = await search_related_rules(db, query, limit)
    
    # Cache result if enabled
    if use_cache:
        cache.add(query, result)
    
    return result

async def search_for_user_query(db, user_query: str, limit: int = 5) -> str:
    """
    Search for rules related to a user query and format them for presentation.
    
    Args:
        db: Database handle
        user_query: The user's query
        limit: Maximum number of results
        
    Returns:
        str: Formatted rules
    """
    search_result = await get_related_rules(db, user_query, limit=limit)
    return format_rules_for_agent(search_result)

async def search_for_reasoning_task(db, reasoning_task: str, limit: int = 5) -> str:
    """
    Search for rules related to a reasoning task and format them for presentation.
    
    Args:
        db: Database handle
        reasoning_task: The reasoning task
        limit: Maximum number of results
        
    Returns:
        str: Formatted rules
    """
    search_result = await get_related_rules(db, reasoning_task, limit=limit)
    return format_rules_for_agent(search_result) 