#!/usr/bin/env python
"""
Enhanced Database Module for Cursor Rules

!!! TECHNICAL DEBT WARNING !!!
This module currently coexists with core/db.py, which violates our architectural principles
from LESSONS_LEARNED.md. This situation needs to be addressed in a dedicated refactoring effort.

Current Status:
- This module (enhanced_db.py) is the more complete implementation with graph features
- core/db.py provides basic database connection functionality
- Both are actively used in the codebase

This Module's Features:
1. Graph-based knowledge system for cursor rules
2. Multiple collections and edge relationships
3. Multi-hop traversals
4. Enhanced search capabilities
5. Schema management
6. View creation and management

For new development:
- Use this module for new features requiring graph or advanced search capabilities
- Follow the patterns in LESSONS_LEARNED.md for database integration
- Ensure proper async/await usage with asyncio.to_thread() as documented

Future Plan:
1. This technical debt should be addressed in a dedicated refactoring sprint
2. The functionality should be consolidated into a single implementation
3. Until then, this is the preferred module for new development

Related Documentation:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB Graph Features: https://www.arangodb.com/docs/stable/graphs.html
- Project LESSONS_LEARNED.md: Directory Structure and Database Integration sections
"""

import os
import sys
import json
import textwrap
import asyncio
from typing import Dict, Any, List, Tuple, Union, Optional, Set
from pathlib import Path
import datetime

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import CollectionCreateError, GraphCreateError, ViewCreateError, ArangoError
from loguru import logger

# Import embedding generation utilities
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding, EMBEDDING_AVAILABLE
from agent_tools.cursor_rules.core.db import get_db
from agent_tools.cursor_rules.schemas import load_schema, DB_SCHEMA

# Load the schema from schemas module
schema = DB_SCHEMA
collections = schema.get("collections", {})
edge_collections = schema.get("edge_collections", {})

# Define collection names
DOCUMENT_COLLECTIONS = [
    'rules',
    'rule_examples',
    'troubleshooting_guides',
    'lessons_learned',
    'prompt_templates'
]

EDGE_COLLECTIONS = [
    'rule_has_example',
    'rule_references_rule',
    'rule_resolves_problem',
    'lesson_references_resource',
    'prompt_uses_resource',
    'resource_applies_to_language'
]

def setup_enhanced_cursor_rules_db(
    config: Dict[str, Any], 
    db_name: str = "cursor_rules", 
    reset: bool = False
) -> StandardDatabase:
    """
    Set up the enhanced cursor rules database with all collections and views.
    
    Args:
        config: Configuration dictionary with arango_config
        db_name: Name of the database to create
        reset: Whether to reset the database if it exists
        
    Returns:
        Database handle
    """
    # Extract ArangoDB configuration
    arango_config = config.get("arango_config", {})
    hosts = arango_config.get("hosts", ["http://localhost:8529"])
    username = arango_config.get("username", "root")
    password = arango_config.get("password", "openSesame")
    
    # Connect to ArangoDB
    logger.info(f"Connecting to ArangoDB at {hosts[0]}")
    logger.info(f"Using username: {username}")
    client = ArangoClient(hosts=hosts)
    sys_db = client.db("_system", username=username, password=password)
    logger.info(f"Connected to _system database")
    
    # Create or get the database
    if reset and sys_db.has_database(db_name):
        sys_db.delete_database(db_name)
        logger.info(f"Reset: Deleted existing database '{db_name}'")
    
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        logger.info(f"Created database '{db_name}'")
    else:
        logger.info(f"Database '{db_name}' already exists")
    
    # Connect to the database
    db = client.db(db_name, username=username, password=password)
    
    # Create document collections
    for collection_name in DOCUMENT_COLLECTIONS:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            logger.info(f"Created collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
    
    # Create edge collections
    for collection_name in EDGE_COLLECTIONS:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name, edge=True)
            logger.info(f"Created edge collection '{collection_name}'")
        else:
            logger.info(f"Edge collection '{collection_name}' already exists")
    
    # Create unified search view
    create_unified_search_view(db)
    
    # Create named graph for traversals
    create_knowledge_graph(db)
    
    return db

def create_unified_search_view(db: StandardDatabase) -> bool:
    """
    Create a unified ArangoSearch view across all collections.
    
    Args:
        db: Database handle
        
    Returns:
        True if the view was created or already exists, False otherwise
    """
    view_name = "unified_search_view"
    
    try:
        # Check if the view already exists
        view_exists = False
        try:
            for view in db.views():
                if view['name'] == view_name:
                    view_exists = True
                    break
        except Exception as e:
            logger.error(f"Error checking if view exists: {e}")
        
        if view_exists:
            logger.info(f"ArangoSearch view '{view_name}' already exists")
            return True
        
        # Get view definition from schema
        view_schema = schema["views"][view_name]
        
        # Ensure we're working with plain dictionaries
        links = {}
        view_properties = {}
        
        # Copy the basic properties
        if "properties" in view_schema:
            view_properties = {**view_schema["properties"]}
        
        # Prepare links for each collection
        if "links" in view_schema:
            links_dict = {}
            for collection_name, fields_config in view_schema["links"].items():
                if db.has_collection(collection_name):
                    # Create a proper link configuration
                    links_dict[collection_name] = {
                        "fields": {**fields_config.get("fields", {})},
                        "includeAllFields": False
                    }
            
            # Add the links to the properties
            view_properties["links"] = links_dict
        
        # Create the view with combined properties
        try:
            db.create_arangosearch_view(
                name=view_name,
                properties=view_properties
            )
            logger.info(f"Created unified ArangoSearch view '{view_name}'")
            return True
        except Exception as e:
            logger.error(f"Error creating unified search view: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error creating unified search view: {e}")
        return False

def create_knowledge_graph(db: StandardDatabase) -> bool:
    """
    Create a named graph for traversals.
    
    Args:
        db: Database handle
        
    Returns:
        True if the graph was created successfully, False otherwise
    """
    graph_name = "knowledge_graph"
    
    try:
        # Check if graph exists
        if db.has_graph(graph_name):
            logger.info(f"Graph '{graph_name}' already exists")
            return True
        
        # Create the graph with all edge definitions
        edge_definitions = []
        
        # rule_has_example
        edge_definitions.append({
            'edge_collection': 'rule_has_example',
            'from_vertex_collections': ['rules'],
            'to_vertex_collections': ['rule_examples']
        })
        
        # rule_references_rule
        edge_definitions.append({
            'edge_collection': 'rule_references_rule',
            'from_vertex_collections': ['rules'],
            'to_vertex_collections': ['rules']
        })
        
        # rule_resolves_problem
        edge_definitions.append({
            'edge_collection': 'rule_resolves_problem',
            'from_vertex_collections': ['rules'],
            'to_vertex_collections': ['troubleshooting_guides']
        })
        
        # lesson_references_resource
        edge_definitions.append({
            'edge_collection': 'lesson_references_resource',
            'from_vertex_collections': ['lessons_learned'],
            'to_vertex_collections': ['rules', 'rule_examples', 'troubleshooting_guides', 'prompt_templates']
        })
        
        # prompt_uses_resource
        edge_definitions.append({
            'edge_collection': 'prompt_uses_resource',
            'from_vertex_collections': ['prompt_templates'],
            'to_vertex_collections': ['rules', 'rule_examples', 'troubleshooting_guides', 'lessons_learned']
        })
        
        # Create the graph
        graph = db.create_graph(graph_name, edge_definitions)
        logger.info(f"Created graph '{graph_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        return False

def multi_hop_rule_discovery(
    db: StandardDatabase, 
    rule_number: str, 
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Discover rules connected through multiple relationships.
    
    Args:
        db: Database handle
        rule_number: Starting rule number
        max_depth: Maximum traversal depth
        
    Returns:
        Dictionary with start rule and related rules
    """
    try:
        # AQL query using graph traversal
        aql_query = textwrap.dedent(f"""
        FOR start_rule IN rules 
            FILTER start_rule.rule_number == @rule_number 
            LET related_rules = (
                FOR related, edge, path IN 1..@max_depth 
                    OUTBOUND start_rule 
                    GRAPH 'knowledge_graph'
                    RETURN DISTINCT {{
                        rule: related,
                        relationship_type: edge.relationship_type,
                        collection: PARSE_IDENTIFIER(related._id).collection,
                        depth: LENGTH(path.edges),
                        path: path.vertices[*]._key
                    }}
            )
            RETURN {{ 
                start_rule: start_rule, 
                related_rules: related_rules 
            }}
        """)
        
        # Execute query
        bind_vars = {
            "rule_number": rule_number,
            "max_depth": max_depth
        }
        
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        if results:
            return results[0]
        else:
            return {"start_rule": None, "related_rules": []}
        
    except Exception as e:
        logger.error(f"Error in multi-hop rule discovery: {e}")
        return {"start_rule": None, "related_rules": [], "error": str(e)}

def knowledge_path_between_resources(
    db: StandardDatabase, 
    from_id: str, 
    to_id: str
) -> Dict[str, Any]:
    """
    Find the shortest path between two resources.
    
    Args:
        db: Database handle
        from_id: Starting resource ID
        to_id: Target resource ID
        
    Returns:
        Dictionary with path information
    """
    try:
        # AQL query for shortest path
        aql_query = textwrap.dedent("""
        FOR path IN ANY SHORTEST_PATH @from_id TO @to_id
            GRAPH 'knowledge_graph'
            RETURN {
                vertices: path.vertices,
                edges: path.edges,
                distance: LENGTH(path.edges)
            }
        """)
        
        # Execute query
        bind_vars = {
            "from_id": from_id,
            "to_id": to_id
        }
        
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        if results:
            return results[0]
        else:
            return {"vertices": [], "edges": [], "distance": -1, "error": "No path found"}
        
    except Exception as e:
        logger.error(f"Error finding path between resources: {e}")
        return {"vertices": [], "edges": [], "distance": -1, "error": str(e)}

def rule_complete_context(db: StandardDatabase, rule_key: str) -> Dict[str, Any]:
    """
    Get a rule with all its related resources.
    
    Args:
        db: Database handle
        rule_key: Rule key
        
    Returns:
        Dictionary with rule and all related resources
    """
    try:
        # AQL query to get complete context
        aql_query = textwrap.dedent("""
        FOR rule IN rules
            FILTER rule._key == @rule_key
            LET examples = (
                FOR e IN OUTBOUND rule rule_has_example
                RETURN e
            )
            LET related_rules = (
                FOR r, edge IN OUTBOUND rule rule_references_rule
                RETURN {
                    rule: r,
                    relationship_type: edge.relationship_type,
                    strength: edge.strength
                }
            )
            LET troubleshooting = (
                FOR t, edge IN OUTBOUND rule rule_resolves_problem
                RETURN {
                    guide: t,
                    resolution_type: edge.resolution_type
                }
            )
            LET lessons = (
                FOR l, edge IN INBOUND rule lesson_references_resource
                RETURN {
                    lesson: l,
                    reference_type: edge.reference_type
                }
            )
            LET prompts = (
                FOR p, edge IN INBOUND rule prompt_uses_resource
                RETURN {
                    prompt: p,
                    usage_type: edge.usage_type
                }
            )
            RETURN {
                rule: rule,
                examples: examples,
                related_rules: related_rules,
                troubleshooting: troubleshooting,
                lessons: lessons,
                prompts: prompts
            }
        """)
        
        # Execute query
        bind_vars = {"rule_key": rule_key}
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        if results:
            return results[0]
        else:
            return {"rule": None, "error": "Rule not found"}
        
    except Exception as e:
        logger.error(f"Error getting rule context: {e}")
        return {"rule": None, "error": str(e)}

def hybrid_cross_collection_search(
    db: StandardDatabase, 
    query_text: str, 
    limit: int = 10, 
    verbose: bool = False,
    semantic: bool = True,
    bm25: bool = True,
    glossary: bool = True,
    threshold: float = 0.4,
    boost_recency: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search across all collections.
    
    Args:
        db: Database handle
        query_text: Search query text
        limit: Maximum number of results
        verbose: Whether to print detailed information
        semantic: Whether to include semantic search component
        bm25: Whether to include BM25 text search component
        glossary: Whether to include glossary matching component
        threshold: Minimum hybrid score threshold (0-1)
        boost_recency: Whether to boost recently accessed items
        
    Returns:
        List of results with resource and relevance score
    """
    try:
        if verbose:
            logger.info(f"\nPerforming cross-collection hybrid search for: '{query_text}'")
            enabled = []
            if semantic:
                enabled.append("semantic")
            if bm25:
                enabled.append("bm25")
            if glossary:
                enabled.append("glossary")
            logger.info(f"Enabled components: {', '.join(enabled)}")
        
        # Validate at least one component is enabled
        if not any([semantic, bm25, glossary]):
            raise ValueError("At least one search component must be enabled")
        
        # Generate embedding if semantic search is enabled
        query_embedding = None
        if semantic and EMBEDDING_AVAILABLE:
            embedding_result = generate_embedding(query_text)
            if embedding_result and "embedding" in embedding_result:
                query_embedding = embedding_result["embedding"]
            elif verbose:
                logger.info("Failed to generate embedding for semantic search")
        
        # If semantic is the only enabled component and embedding failed, return empty
        if semantic and not bm25 and not glossary and not query_embedding:
            return []
        
        # Calculate component weights based on enabled components
        total_components = sum([semantic, bm25, glossary])
        base_weight = 0.8 / total_components  # Reserve 0.2 for importance and recency
        semantic_weight = base_weight if semantic else 0
        bm25_weight = base_weight if bm25 else 0
        glossary_weight = base_weight if glossary else 0
        importance_weight = 0.1
        recency_weight = 0.1 if boost_recency else 0
        
        # Construct the hybrid search query
        aql_query = textwrap.dedent("""
        // Initialize result arrays
        LET semantic_results = {semantic_subquery}
        
        LET bm25_results = {bm25_subquery}
        
        LET glossary_results = {glossary_subquery}
        
        // Merge and deduplicate results
        LET merged_results = (
            FOR result IN UNION_DISTINCT(semantic_results, bm25_results, glossary_results)
                COLLECT key = result._key INTO group
                LET doc = FIRST(group[*].result.doc)
                LET similarity_score = MAX(group[*].result.similarity_score)
                LET bm25_score = MAX(group[*].result.bm25_score)
                LET glossary_score = MAX(group[*].result.glossary_score)
                
                // Calculate recency boost if enabled
                LET days_since_access = DATE_DIFF(doc.last_accessed, @current_time, "d")
                LET days_factor = 1.0 / (1.0 + ABS(days_since_access))
                LET recency_boost = @boost_recency ? (days_factor * @recency_weight) : 0
                
                // Calculate hybrid score with dynamic weights
                LET hybrid_score = (
                    (similarity_score * @semantic_weight) + 
                    (bm25_score * @bm25_weight) +
                    (glossary_score * @glossary_weight) +
                    (doc.importance * @importance_weight) +
                    recency_boost
                )
                
                FILTER hybrid_score >= @threshold
                RETURN {{
                    resource: doc,
                    collection: PARSE_IDENTIFIER(doc._id).collection,
                    relevance: hybrid_score,
                    components: {{
                        semantic_score: similarity_score,
                        bm25_score: bm25_score,
                        glossary_score: glossary_score,
                        importance_boost: doc.importance,
                        recency_boost: recency_boost
                    }}
                }}
        )
        
        // Sort and limit final results
        FOR result IN merged_results
            SORT result.relevance DESC
            LIMIT @limit
            RETURN result
        """.format(
            semantic_subquery="""(
                FOR doc IN unified_search_view
                    FILTER @semantic AND doc.embedding != null
                    LET similarity = COSINE_SIMILARITY(doc.embedding, @query_vector)
                    FILTER similarity >= 0.5
                    SORT similarity DESC
                    LIMIT @limit
                    RETURN {
                        doc: doc,
                        _key: doc._key,
                        similarity_score: similarity,
                        bm25_score: 0,
                        glossary_score: 0
                    }
            )""" if semantic else "[]",
            
            bm25_subquery="""(
                FOR doc IN unified_search_view
                    FILTER @bm25
                    SEARCH ANALYZER(
                        doc.title IN TOKENS(@search_text, 'text_en') OR
                        doc.description IN TOKENS(@search_text, 'text_en'),
                        'text_en'
                    )
                    LET bm25_score = BM25(doc)
                    FILTER bm25_score > 0.1
                    SORT bm25_score DESC
                    LIMIT @limit
                    RETURN {
                        doc: doc,
                        _key: doc._key,
                        similarity_score: 0,
                        bm25_score: bm25_score,
                        glossary_score: 0
                    }
            )""" if bm25 else "[]",
            
            glossary_subquery="""(
                FOR doc IN unified_search_view
                    FILTER @glossary AND doc.glossary_terms != null
                    LET matches = (
                        FOR term IN doc.glossary_terms
                            FILTER LOWER(term) IN TOKENS(LOWER(@search_text), "text_en")
                            RETURN 1
                    )
                    LET glossary_score = LENGTH(matches) > 0 
                        ? MIN(LENGTH(matches) / LENGTH(TOKENS(@search_text, "text_en")))
                        : 0
                    FILTER glossary_score >= 0.3
                    SORT glossary_score DESC
                    LIMIT @limit
                    RETURN {
                        doc: doc,
                        _key: doc._key,
                        similarity_score: 0,
                        bm25_score: 0,
                        glossary_score: glossary_score
                    }
            )""" if glossary else "[]"
        ))
        
        if verbose:
            logger.info("\nAQL QUERY:")
            logger.info("-" * 80)
            logger.info(aql_query)
            logger.info("-" * 80)
        
        # Execute query
        bind_vars = {
            "search_text": query_text,
            "query_vector": query_embedding,
            "limit": limit,
            "semantic": semantic,
            "bm25": bm25,
            "glossary": glossary,
            "threshold": threshold,
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "glossary_weight": glossary_weight,
            "importance_weight": importance_weight,
            "recency_weight": recency_weight,
            "boost_recency": boost_recency,
            "current_time": datetime.datetime.now().isoformat()
        }
        
        if verbose:
            logger.info(f"Bind variables: {list(bind_vars.keys())}")
        
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        if verbose:
            logger.info(f"\nFound {len(results)} results across all collections")
            
            # Group results by collection
            results_by_collection = {}
            for result in results:
                collection = result["collection"]
                if collection not in results_by_collection:
                    results_by_collection[collection] = []
                results_by_collection[collection].append(result)
            
            for collection, items in results_by_collection.items():
                logger.info(f"\n{collection.upper()} ({len(items)} results):")
                logger.info("-" * 80)
                for i, item in enumerate(items, 1):
                    resource = item["resource"]
                    score = item["relevance"]
                    components = item["components"]
                    logger.info(f"{i}. {resource.get('title', 'Untitled')} (Score: {score:.4f})")
                    logger.info(f"   Component Scores:")
                    if semantic:
                        logger.info(f"     - Semantic: {components['semantic_score']:.4f}")
                    if bm25:
                        logger.info(f"     - BM25: {components['bm25_score']:.4f}")
                    if glossary:
                        logger.info(f"     - Glossary: {components['glossary_score']:.4f}")
                    logger.info(f"     - Importance: {components['importance_boost']:.4f}")
                    if boost_recency:
                        logger.info(f"     - Recency: {components['recency_boost']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in hybrid cross-collection search: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def contextual_recommendation(
    db: StandardDatabase, 
    file_pattern: str, 
    language: str = None, 
    limit: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recommend resources based on current file context.
    
    Args:
        db: Database handle
        file_pattern: Current file pattern (e.g., *.py)
        language: Programming language
        limit: Maximum number of results per category
        
    Returns:
        Dictionary with recommended resources by category
    """
    try:
        # AQL query for contextual recommendations
        aql_query = textwrap.dedent("""
        LET context_rules = (
            FOR rule IN rules
                FILTER rule.glob_pattern == @file_pattern
                SORT rule.priority ASC
                LIMIT @limit
                RETURN rule
        )
        
        LET relevant_examples = (
            FOR rule IN context_rules
                FOR example IN OUTBOUND rule rule_has_example
                    FILTER @language == null OR example.language == @language
                    LIMIT @limit
                    RETURN example
        )
        
        LET relevant_troubleshooting = (
            FOR problem IN troubleshooting_guides
                FILTER @language == null OR problem.language == @language
                FOR rule IN INBOUND problem rule_resolves_problem
                    FILTER rule.glob_pattern == @file_pattern
                    LIMIT @limit
                    RETURN problem
        )
        
        LET related_lessons = (
            FOR rule IN context_rules
                FOR lesson, edge IN INBOUND rule lesson_references_resource
                    SORT edge.importance DESC
                    LIMIT @limit
                    RETURN lesson
        )
        
        RETURN {
            rules: context_rules,
            examples: relevant_examples,
            troubleshooting: relevant_troubleshooting,
            lessons: related_lessons
        }
        """)
        
        # Execute query
        bind_vars = {
            "file_pattern": file_pattern,
            "language": language,
            "limit": limit
        }
        
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        if results:
            return results[0]
        else:
            return {
                "rules": [],
                "examples": [],
                "troubleshooting": [],
                "lessons": []
            }
        
    except Exception as e:
        logger.error(f"Error getting contextual recommendations: {e}")
        return {
            "rules": [],
            "examples": [],
            "troubleshooting": [],
            "lessons": [],
            "error": str(e)
        }

def create_views(db: StandardDatabase) -> List[str]:
    """Create views based on the database schema."""
    created_views = []
    
    # Get all views defined in the schema
    views = schema.get("views", {})
    
    for view_name, view_schema in views.items():
        # Check if view already exists
        try:
            db.view(view_name)
            logger.info(f"View {view_name} already exists.")
            created_views.append(view_name)
            continue
        except Exception:
            # View doesn't exist, continue to create it
            pass
        
        try:
            # Extract the schema for this specific view from the overall schema
            view_schema = schema["views"][view_name]
            
            # Ensure we're working with plain dictionaries
            links = {}
            view_properties = {}
            
            # Copy the basic properties
            if "properties" in view_schema:
                view_properties = {**view_schema["properties"]}
            
            # Prepare links for each collection
            if "links" in view_schema:
                links_dict = {}
                for collection_name, fields_config in view_schema["links"].items():
                    if db.has_collection(collection_name):
                        # Create a proper link configuration
                        links_dict[collection_name] = {
                            "fields": {**fields_config.get("fields", {})},
                            "includeAllFields": False
                        }
            
                # Add the links to the properties
                view_properties["links"] = links_dict
            
            # Create the view with combined properties
            try:
                db.create_arangosearch_view(
                    name=view_name,
                    properties=view_properties
                )
                logger.info(f"Created view {view_name}")
                created_views.append(view_name)
            except Exception as e:
                logger.error(f"Error creating view {view_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing view {view_name}: {e}")
    
    return created_views

# Example usage
if __name__ == "__main__":
    config = {
        "arango_config": {
            "hosts": ["http://localhost:8529"],
            "username": "root",
            "password": "openSesame"
        }
    }
    
    db = setup_enhanced_cursor_rules_db(config, db_name="cursor_rules_enhanced")
    
    # Add a test rule
    rule = {
        "_key": "test-rule",
        "rule_number": "999",
        "title": "Test Rule",
        "description": "A test rule for graph traversal",
        "content": "This is a test rule for testing the graph traversal capabilities.",
        "rule_type": "test",
        "glob_pattern": "*.py",
        "priority": 999
    }
    
    db.collection("rules").insert(rule)
    logger.info(f"Added test rule with key '{rule['_key']}'")
    
    # Test graph traversal
    result = multi_hop_rule_discovery(db, "999", max_depth=2)
    logger.info(f"Multi-hop discovery results: {json.dumps(result, indent=2)}")
    
    # Test hybrid search
    results = hybrid_cross_collection_search(db, "test rule", limit=5, verbose=True)
    logger.info(f"Found {len(results)} results with hybrid search") 