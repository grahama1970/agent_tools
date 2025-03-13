#!/usr/bin/env python
"""
Simplified Cursor Rules Database Manager

A straightforward implementation that follows our working example.
"""
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError
from typing import Dict, Any, List, Union, Optional
import os
import json
from pathlib import Path
import re
from datetime import datetime, timezone
import numpy as np
import textwrap
import asyncio

# Add tabulate for pretty printing results
from tabulate import tabulate

# For embeddings
try:
    from agent_tools.cursor_rules.embedding import create_embedding_sync, ensure_text_has_prefix
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    from loguru import logger
    logger.warning("Embedding utilities not available. Vector search will be disabled.")

# Schema Definition
CURSOR_RULES_SCHEMA = {
    "database_name": "cursor_rules",
    "collections": {
        "rules": {
            "type": "document",
            "description": "Stores cursor rules"
        },
        "pattern_examples": {
            "type": "document", 
            "description": "Code examples demonstrating rule patterns"
        }
    }
}

# Schema Definition
RULE_SCHEMA = {
    "type": "object",
    "required": ["title", "description", "key", "content"],
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "key": {"type": "string"},
        "content": {"type": "string"},
        "glob_pattern": {"type": "string"},
        "priority": {"type": "integer"}
    }
}

# Sample rule for testing
SAMPLE_RULE = {
    "key": "000-core-rules",
    "title": "000: Core Rules - ALWAYS READ AND FOLLOW THESE",
    "description": "These are the core rules that must be followed at all times.",
    "content": "# Core Rules\n\nThese rules must be followed at all times:\n\n1. Always use type hints\n2. Always include docstrings\n3. Follow PEP 8 style guidelines",
    "glob_pattern": "*.py",
    "priority": 0
}

# Sample pattern example
SAMPLE_EXAMPLE = {
    "_key": "001_ex1",
    "rule_key": "001-code-advice-rules",
    "title": "Type Hints Example",
    "description": "Demonstrates proper use of type hints",
    "language": "python",
    "good_example": "def add(a: int, b: int) -> int:\n    return a + b",
    "bad_example": "def add(a, b):\n    return a + b"
}

def generate_embedding(text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]:
    """
    Generate embeddings using Nomic ModernBert model.
    
    Args:
        text: Text to embed
        
    Returns:
        Dictionary with embedding vector and metadata
    """
    if not EMBEDDING_AVAILABLE:
        # Return empty embedding if embedding utilities aren't available
        return {
            "embedding": [],
            "metadata": {
                "embedding_model": "none",
                "embedding_timestamp": datetime.now(timezone.utc).isoformat(),
                "embedding_method": "none",
                "embedding_dim": 0
            }
        }
    
    # Ensure text has the required prefix for Nomic Embed
    text = ensure_text_has_prefix(text)
    
    # Generate embedding
    embedding_result = create_embedding_sync(text)
    
    return embedding_result

def load_rules_from_directory(rules_dir: str) -> List[Dict[str, Any]]:
    """
    Load rules from a directory.
    
    Args:
        rules_dir: Path to directory containing rule files
        
    Returns:
        List of rules
    """
    rules = []
    rules_dir = Path(rules_dir)
    
    # Ensure directory exists
    if not rules_dir.exists():
        print(f"Rules directory {rules_dir} does not exist")
        return rules
    
    # Load each rule file
    for rule_file in rules_dir.glob("*.mdc"):
        try:
            print(f"Reading rule file: {rule_file.name}")
            
            # Extract rule number from filename (e.g., "001-test-rule.mdc" -> "001")
            rule_number = rule_file.stem.split("-")[0]
            
            # Read rule content
            with open(rule_file, "r") as f:
                    content = f.read()
                
            # Check if the file has YAML frontmatter (starts with ---)
            if content.startswith("---"):
                # Skip the frontmatter section
                content_parts = content.split("---", 2)
                if len(content_parts) >= 3:
                    # The actual content starts after the second "---"
                    content = content_parts[2].strip()
            
            # Parse title from first line (assumes markdown format)
            title_line = content.split("\n")[0].strip()
            if title_line.startswith("# "):
                title = title_line[2:]  # Remove the "# " prefix
            else:
                title = title_line
            
            # Extract description - everything after the title until the next heading
            description = ""
            content_lines = content.split("\n")[1:]
            for i, line in enumerate(content_lines):
                if line.strip() and not line.startswith("#"):
                    # Found the start of the description
                    description_start = i
                    break
            else:
                description_start = 0
            
            for i, line in enumerate(content_lines[description_start:]):
                if line.startswith("## "):
                    # Found the end of the description
                    description_end = description_start + i
                    break
            else:
                description_end = len(content_lines)
            
            description = "\n".join(content_lines[description_start:description_end]).strip()
            
        # Create rule object
            rule = {
                "rule_number": rule_number,
                "title": title,
                "description": description,
                "content": content,
                "file_path": str(rule_file)
            }
            
            rules.append(rule)
        
        except Exception as e:
            print(f"Error loading rule from {rule_file}: {e}")
            continue
        
    print(f"Loaded {len(rules)} rules from {rules_dir}")
    return rules

def setup_cursor_rules_db(config=None, rules_dir=None, db_name="cursor_rules_test"):
    """
    Set up the cursor rules database with sample data.
    
    Args:
        config: Dictionary with database configuration. If None, uses default config.
        rules_dir: Optional path to rules directory
        db_name: Name of the database to create/connect to
        
    Returns:
        Database handle if successful, None otherwise
    """
    try:
        # Use default config if none provided
        if config is None:
            config = {
                "arango": {
                    "hosts": ["http://localhost:8529"],
                    "username": "root",
                    "password": "openSesame"
                }
            }

        # Connect to ArangoDB
        arango_config = config.get("arango", {})
        hosts = arango_config.get("hosts", ["http://localhost:8529"])
        if isinstance(hosts, str):
            hosts = [hosts]
        
        username = arango_config.get("username", "root")
        password = arango_config.get("password", "openSesame")
        
        print(f"Connecting to ArangoDB at {hosts[0]}")
        print(f"Using username: {username}")
        
        client = ArangoClient(hosts=hosts[0])
        sys_db = client.db("_system", username=username, password=password)
        
        print(f"Connected to _system database")
        
        # Check if database exists
        if sys_db.has_database(db_name):
            print(f"Database {db_name} already exists")
        else:
            sys_db.create_database(db_name)
            print(f"Created database {db_name}")
        
        # Connect to the database
        db = client.db(db_name, username=username, password=password)
        
        # Create collections if they don't exist
        if db.has_collection("rules"):
            print("Collection 'rules' already exists")
        else:
            db.create_collection("rules")
            print("Created collection 'rules'")
            
        if db.has_collection("rule_examples"):
            print("Collection 'rule_examples' already exists")
        else:
            db.create_collection("rule_examples")
            print("Created collection 'rule_examples'")
        
        # Load rules if directory provided
        if rules_dir:
            rules = load_rules_from_directory(rules_dir)
            if rules:
                # Bulk insert rules
                db.collection("rules").import_bulk(rules, on_duplicate="update")
                print(f"Inserted {len(rules)} rules")
            else:
                # Insert sample rule if no rules loaded
                sample_embedding = generate_embedding(f"RULE: {SAMPLE_RULE['title']}\nDESCRIPTION: {SAMPLE_RULE['description']}\nCONTENT: {SAMPLE_RULE['content']}")
                SAMPLE_RULE["embedding"] = sample_embedding["embedding"]
                SAMPLE_RULE["embedding_metadata"] = sample_embedding["metadata"]
                
                rules = db.collection("rules")
                rules.insert(SAMPLE_RULE, overwrite=True)
                print("Inserted sample rule")
        else:
            # Insert sample rule
            sample_embedding = generate_embedding(f"RULE: {SAMPLE_RULE['title']}\nDESCRIPTION: {SAMPLE_RULE['description']}\nCONTENT: {SAMPLE_RULE['content']}")
            SAMPLE_RULE["embedding"] = sample_embedding["embedding"]
            SAMPLE_RULE["embedding_metadata"] = sample_embedding["metadata"]
            
            rules = db.collection("rules")
            rules.insert(SAMPLE_RULE, overwrite=True)
            print("Inserted sample rule")
        
        # Insert sample example
        examples = db.collection("rule_examples")
        examples.insert(SAMPLE_EXAMPLE, overwrite=True)
        print("Inserted sample example")
        
        return db
        
    except ServerConnectionError as e:
        print(f"Failed to connect to ArangoDB: {e}")
    except ArangoError as e:
        print(f"ArangoDB error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

def get_all_rules(db):
    """
    Get all rules from the database.
    
    Args:
        db: Database handle
        
    Returns:
        List of rule dictionaries
    """
    try:
        cursor = db.aql.execute("FOR rule IN rules RETURN rule")
        return list(cursor)
    except Exception as e:
        print(f"Error getting rules: {e}")
        return []

def get_examples_for_rule(db, rule_key: str) -> List[Dict[str, Any]]:
    """
    Get examples for a specific rule.
    
    Args:
        db: Database handle
        rule_key: Key of the rule to get examples for
        
    Returns:
        List of example dictionaries
    """
    query = textwrap.dedent("""
    FOR example IN rule_examples
        FILTER example.rule_key == @rule_key OR example.rule_id == @rule_key
            RETURN example
    """)
    
    try:
        cursor = db.aql.execute(query, bind_vars={"rule_key": rule_key})
        return list(cursor)
    except Exception as e:
        print(f"Error getting examples for rule {rule_key}: {e}")
        return []

def create_arangosearch_view(db, collection_name, view_name):
    """
    Create an ArangoSearch view for the specified collection.
    
    This function creates an ArangoSearch view that can be used for full-text search
    and BM25 ranking.
    
    Args:
        db: Database handle
        collection_name: Name of the collection to create the view for
        view_name: Name of the view to create
        
    Returns:
        True if the view was created or already exists, False otherwise
    """
    try:
        # Check if the view already exists
        view_exists = False
        try:
            # Try to get the view
            for view in db.views():
                if view['name'] == view_name:
                    view_exists = True
                    break
        except Exception as e:
            print(f"Error checking if view exists: {e}")
            # Continue anyway, we'll try to create it
        
        if view_exists:
            print(f"ArangoSearch view '{view_name}' already exists")
            return True
        
        # Create the view
        view_properties = {
            "primarySort": [
                {"field": "rule_number", "direction": "asc"}
            ],
            "primarySortCompression": "lz4",
            "storedValues": [
                {"fields": ["rule_number", "title"], "compression": "lz4"}
            ],
            "links": {
                collection_name: {
                    "includeAllFields": False,
                    "fields": {
                        "title": {
                            "analyzers": ["text_en"],
                            "includeAllFields": False
                        },
                        "description": {
                            "analyzers": ["text_en"],
                            "includeAllFields": False
                        },
                        "content": {
                            "analyzers": ["text_en"],
                            "includeAllFields": False
                        }
                    }
                }
            }
        }
        
        # Create the view
        try:
            db.create_view(
                name=view_name,
                view_type="arangosearch",
                properties=view_properties
            )
            print(f"Created ArangoSearch view '{view_name}'")
            return True
        except Exception as e:
            print(f"Error creating ArangoSearch view: {e}")
            # Try an alternative method
            try:
                db.create_arangosearch_view(view_name, properties=view_properties)
                print(f"Created ArangoSearch view '{view_name}' using alternative method")
                return True
            except Exception as e2:
                print(f"Error creating ArangoSearch view using alternative method: {e2}")
                return False
    except Exception as e:
        print(f"Error creating ArangoSearch view: {e}")
        return False

async def bm25_keyword_search(db, query_text, collection_name="rules", limit=5, verbose=False):
    """
    Perform BM25 keyword search on the query_scenarios collection.
    """
    try:
        # Ensure the view exists
        view_name = f"{collection_name}_view"
        await asyncio.to_thread(create_arangosearch_view, db, collection_name, view_name)
        
        # Construct the AQL query for BM25 search
        aql = f"""
        FOR doc IN {view_name}
        SEARCH ANALYZER(
            BOOST(doc.title IN TOKENS(@query, "text_en"), 1.5) OR
            BOOST(doc.description IN TOKENS(@query, "text_en"), 1.0) OR
            BOOST(doc.content IN TOKENS(@query, "text_en"), 0.8),
            "text_en"
        )
        LET bm25_score = BM25(doc)
        FILTER bm25_score > 0.0
        SORT bm25_score DESC
        LIMIT @limit
        RETURN {{
            rule: doc,
            score: bm25_score
        }}
        """
        
        if verbose:
            print("\nBM25 Search Query:")
            print("-"*80)
            print(aql)
            print("-"*80)
            print(f"Search text: '{query_text}'")
            print(f"Collection: {collection_name}")
            print(f"View: {view_name}")
            print(f"Limit: {limit}")
        
        # Execute the query
        cursor = await asyncio.to_thread(
            db.aql.execute,
            aql,
            bind_vars={'query': query_text, 'limit': limit}
        )
        
        # Convert cursor to list
        results = await asyncio.to_thread(list, cursor)
        
        if verbose:
            print(f"\nFound {len(results)} results for query: '{query_text}'")
            try:
                from tabulate import tabulate
                headers = ["Score", "Title", "Description"]
                data = [
                    [
                        f"{result['score']:.2f}",
                        result['rule'].get('title', 'N/A'),
                        result['rule'].get('description', 'N/A')[:50] + "..." if len(result['rule'].get('description', 'N/A')) > 50 else result['rule'].get('description', 'N/A')
                    ]
                    for result in results
                ]
                print(tabulate(data, headers=headers))
            except ImportError:
                for i, result in enumerate(results):
                    print(f"{i+1}. Score: {result['score']:.2f} | Title: {result['rule'].get('title', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"Error performing BM25 search: {e}")
        import traceback
        traceback.print_exc()
        return []

def semantic_search(db, query_text, limit=5, verbose=False):
    """
    Perform semantic search using vector embeddings.
    
    Args:
        db: Database handle
        query_text: Search query
        limit: Maximum number of results
        verbose: Whether to print detailed information about the search
        
    Returns:
        List of matching rules with scores
    """
    if not EMBEDDING_AVAILABLE:
        print("Semantic search not available - embedding utilities not found")
        return []
    
    try:
        if verbose:
            print("\n" + "="*80)
            print(f"DATABASE: {db.name}")
            print("="*80)
            print(f"COLLECTION: rules")
            print("-"*80)
        
        # Generate embedding for the query
        # Using search_query prefix for queries
        prefixed_query = "search_query: " + query_text
        if verbose:
            print(f"Generating embedding for query: '{query_text}'")
            print(f"Using prefixed query: '{prefixed_query}'")
        
        query_embedding = create_embedding_sync(prefixed_query)
        
        # Perform vector similarity search using AQL
        aql_query = """
        FOR rule IN rules
            LET similarity = COSINE_SIMILARITY(rule.embedding, @query_embedding)
            FILTER similarity > 0.5
            SORT similarity DESC
            LIMIT @limit
            RETURN {
                "rule": rule,
                "similarity": similarity
            }
        """
        
        if verbose:
            print("\nAQL QUERY:")
            print("-"*80)
            print(aql_query)
            print("-"*80)
            print(f"Bind variables: limit={limit}, query_embedding=[{len(query_embedding['embedding'])} dimensions]")
        
        cursor = db.aql.execute(
            aql_query, 
            bind_vars={
                "query_embedding": query_embedding["embedding"],
                "limit": limit
            }
        )
        results = list(cursor)
        
        if verbose:
            try:
                from tabulate import tabulate
            except ImportError:
                print("WARNING: tabulate not installed. Install with: pip install tabulate")
                # Simple tabulate replacement if not available
                def tabulate(data, headers, tablefmt="grid"):
                    if not data:
                        return "No data"
                    result = []
                    # Add headers
                    result.append(" | ".join(headers))
                    result.append("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
                    # Add rows
                    for row in data:
                        result.append(" | ".join(str(cell) for cell in row))
                    return "\n".join(result)
            
            print("\nSEARCH RESULTS:")
            print("-"*80)
            
            if results:
                # Prepare data for tabulate
                table_data = []
                for i, result in enumerate(results):
                    rule = result["rule"]
                    similarity = result["similarity"]
                    table_data.append([
                        i+1,
                        rule.get("rule_number", "N/A"),
                        rule.get("title", "Untitled"),
                        rule.get("description", "")[:50] + "..." if len(rule.get("description", "")) > 50 else rule.get("description", ""),
                        f"{similarity:.4f}"
                    ])
                
                # Display table
                headers = ["#", "Rule #", "Title", "Description", "Similarity"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                print("No results found")
        
        print(f"Found {len(results)} relevant rules via semantic search")
        return results
    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

def validate_embeddings(db, collection_name, dimension=768):
    """Validate that embeddings in the collection are valid vectors of the specified dimension."""
    try:
        # Collection name interpolation (must be sanitized)
        query = f"""
            RETURN COUNT(
                FOR doc IN {collection_name}
                FILTER 
                    NOT HAS(doc, 'embedding') OR
                    NOT IS_LIST(doc.embedding) OR
                    LENGTH(doc.embedding) != @dimension OR
                    LENGTH(
                        FOR e IN doc.embedding 
                        FILTER NOT IS_NUMBER(e) 
                        LIMIT 1 RETURN true
                    ) > 0
                RETURN 1
            ) == 0
        """
        result = db.aql.execute(query, bind_vars={"dimension": dimension})
        valid = next(result, False)
        
        print(f"{'✅' if valid else '❌'} Embedding validation result: {valid}")
        return valid
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

async def hybrid_search(db, query_text, collection_name="rules", limit=5, verbose=False, **kwargs):
    """
    Perform hybrid search combining BM25 and vector similarity.
    
    This function combines the results of BM25 keyword search and vector similarity
    search to provide more comprehensive results.
    
    Documentation:
    - ArangoDB Vector Search: https://docs.arangodb.com/3.11/search/vector-search/
    - ArangoDB BM25: https://docs.arangodb.com/3.11/aql/functions/arangosearch/#bm25
    - ArangoDB Hybrid Search: https://docs.arangodb.com/3.11/search/hybrid-search/
    
    Args:
        db: Database handle
        query_text: Search query text
        collection_name: Name of the collection to search in
        limit: Maximum number of results
        verbose: Whether to print detailed information about the search
        **kwargs: Additional parameters for testing purposes
        
    Returns:
        List of tuples (rule, score) where score is a combined score
    """
    try:
        # Handle empty queries early to avoid unnecessary processing
        if not query_text or query_text.strip() == "":
            if verbose:
                print("Empty query provided. Returning empty results.")
            return []
            
        if verbose:
            print("\n" + "="*80)
            print(f"HYBRID SEARCH: {query_text}")
            print("="*80)
            print(f"DATABASE: {db.name}")
            print(f"COLLECTION: {collection_name}")
            print("-"*80)
        
        # Generate embedding for the query text
        query_embedding = generate_embedding(query_text)
        if not query_embedding or "embedding" not in query_embedding:
            if verbose:
                print("Failed to generate embedding for query. Falling back to BM25 search only.")
            return await bm25_keyword_search(db, query_text, collection_name=collection_name, limit=limit, verbose=verbose)
        
        # Ensure ArangoSearch view exists
        view_name = f"{collection_name}_view"
        view_created = await asyncio.to_thread(create_arangosearch_view, db, collection_name, view_name)
        
        if not view_created:
            print("Failed to create ArangoSearch view")
            return []
        
        # BM25 parameters
        k1 = 1.2  # Term frequency saturation
        b = 0.75  # Length normalization factor
        
        # Get thresholds from kwargs or use defaults
        initial_bm25_threshold = kwargs.get("_force_bm25_threshold", 0.1)
        initial_embedding_threshold = kwargs.get("_force_embedding_threshold", 0.5)
        initial_hybrid_score_threshold = kwargs.get("_test_hybrid_score_threshold", 0.15)
        
        # Create a list of progressively lower thresholds to try
        bm25_thresholds = [initial_bm25_threshold, 0.05, 0.01]
        embedding_thresholds = [initial_embedding_threshold, 0.3, 0.2, 0.1]
        hybrid_thresholds = [initial_hybrid_score_threshold, 0.1, 0.05, 0.01]
        
        results = []
        
        # Try different thresholds until we get results or exhaust options
        for threshold_index in range(len(bm25_thresholds)):
            if results:
                break
                
            bm25_threshold = bm25_thresholds[min(threshold_index, len(bm25_thresholds)-1)]
            embedding_threshold = embedding_thresholds[min(threshold_index, len(embedding_thresholds)-1)]
            hybrid_threshold = hybrid_thresholds[min(threshold_index, len(hybrid_thresholds)-1)]
            
            if verbose:
                print(f"\nAttempting search with thresholds - BM25: {bm25_threshold}, " +
                      f"Embedding: {embedding_threshold}, Hybrid: {hybrid_threshold}")
        
            # Specific handling for nonexistent terms - for nonexistent terms, the BM25 score is often low
            # but still barely above zero, while embeddings can still return spurious matches
            # Adding an additional filter to check if any result has a high enough score
            minimum_bm25_for_nonexistent = 0.05 if threshold_index == 0 else 0.01
            
            if verbose:
                print(f"Using ArangoSearch view: {view_name}")
                print(f"BM25 parameters: k1={k1}, b={b}")
                print(f"BM25 threshold: {bm25_threshold}")
                print(f"Embedding similarity threshold: {embedding_threshold}")
                print(f"Hybrid score threshold: {hybrid_threshold}")
                print(f"Minimum BM25 score for nonexistent terms: {minimum_bm25_for_nonexistent}")
                print(f"Limit: {limit}")
            
            # First run a simple BM25 query to check if there are any relevant documents
            # This acts as a first filter for nonexistent terms
            bm25_check_query = textwrap.dedent(f"""
            FOR doc IN @@view
                SEARCH ANALYZER(
                    doc.title IN TOKENS(@search_text, "text_en") OR
                    doc.description IN TOKENS(@search_text, "text_en") OR
                    doc.content IN TOKENS(@search_text, "text_en"),
                    "text_en"
                )
                LET score = BM25(doc, @k1, @b)
                FILTER score > @minimum_bm25
                RETURN score
            """)
            
            # Execute BM25 check query
            bm25_check_bind_vars = {
                "@view": view_name,
                "search_text": query_text,
                "k1": k1,
                "b": b,
                "minimum_bm25": minimum_bm25_for_nonexistent
            }
            
            if verbose:
                print("\nRunning BM25 existence check:")
                print(bm25_check_query)
                print(f"Minimum BM25 required: {minimum_bm25_for_nonexistent}")
            
            cursor = await asyncio.to_thread(db.aql.execute, bm25_check_query, bind_vars=bm25_check_bind_vars)
            bm25_scores = await asyncio.to_thread(list, cursor)
            
            # Skip BM25 check for lower thresholds to ensure we don't miss semantic matches
            if threshold_index == 0 and not bm25_scores:
                if verbose:
                    print(f"No documents have a BM25 score > {minimum_bm25_for_nonexistent} for query '{query_text}'")
                # Don't return empty yet - continue with the next threshold
                continue 
            
            if verbose and bm25_scores:
                print(f"Found {len(bm25_scores)} documents with BM25 score > {minimum_bm25_for_nonexistent}")
                print(f"BM25 scores: {bm25_scores}")
            
            # Use separate subqueries for embedding and BM25 results, following the example
            aql_query = textwrap.dedent(f"""
            LET embedding_results = (
                FOR doc IN {collection_name}
                    LET similarity = COSINE_SIMILARITY(doc.embedding, @query_vector)
                    FILTER similarity >= @embedding_similarity_threshold
                    SORT similarity DESC
                    LIMIT @limit
                    RETURN {{
                        doc: doc,
                        _key: doc._key,
                        similarity_score: similarity,
                        bm25_score: 0
                    }}
            )
            
            LET bm25_results = (
                FOR doc IN @@view
                    SEARCH ANALYZER(
                        doc.title IN TOKENS(@search_text, "text_en") OR
                        doc.description IN TOKENS(@search_text, "text_en") OR
                        doc.content IN TOKENS(@search_text, "text_en"),
                        "text_en"
                    )
                    LET bm25_score = BM25(doc, @k1, @b)
                    FILTER bm25_score > @bm25_threshold
                    SORT bm25_score DESC
                    LIMIT @limit
                    RETURN {{
                        doc: doc,
                        _key: doc._key,
                        similarity_score: 0,
                        bm25_score: bm25_score
                    }}
            )
            
            // Merge and deduplicate embedding and BM25 results
            LET merged_results = (
                FOR result IN UNION_DISTINCT(embedding_results, bm25_results)
                    COLLECT key = result._key INTO group
                    LET doc = FIRST(group[*].result.doc)
                    LET similarity_score = MAX(group[*].result.similarity_score)
                    LET bm25_score = MAX(group[*].result.bm25_score)
                    LET hybrid_score = (similarity_score * 0.5) + (bm25_score * 0.5)
                    // Add stricter filtering for hybrid scores
                    FILTER hybrid_score >= @hybrid_score_threshold
                    RETURN {{
                        "rule": doc,
                        "_key": key,
                        "similarity_score": similarity_score,
                        "bm25_score": bm25_score,
                        "hybrid_score": hybrid_score
                    }}
            )
            
            // Sort and limit merged results
            LET final_results = (
                FOR result IN merged_results
                    SORT result.hybrid_score DESC
                    LIMIT @limit
                    RETURN result
            )
            
            // Return final results
            RETURN final_results
            """)
            
            if verbose:
                print("\nAQL QUERY:")
                print("-"*80)
                print(aql_query)
                print("-"*80)
            
            # Execute query
            bind_vars = {
                "@view": view_name,
                "@collection": collection_name,
                "search_text": query_text,
                "query_vector": query_embedding["embedding"],
                "embedding_similarity_threshold": embedding_threshold,
                "bm25_threshold": bm25_threshold,
                "hybrid_score_threshold": hybrid_threshold,
                "k1": k1,
                "b": b,
                "limit": limit
            }
            
            if verbose:
                print(f"Bind variables: {bind_vars}")
            
            cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars=bind_vars)
            results_list = await asyncio.to_thread(list, cursor)
            
            # The query returns a list with a single item, which is the list of results
            if results_list:
                results = results_list[0]
                if verbose:
                    print(f"Found {len(results)} results with threshold attempt {threshold_index+1}")
        
        # Additional check for extremely low relevance results across all thresholds
        if results and all(result["hybrid_score"] < 0.05 for result in results):
            if verbose:
                print(f"All results have extremely low relevance scores (< 0.05) for query '{query_text}'. Results might not be relevant.")
        
        if verbose:
            try:
                from tabulate import tabulate
            except ImportError:
                print("WARNING: tabulate not installed. Install with: pip install tabulate")
                # Simple tabulate replacement if not available
                def tabulate(data, headers, tablefmt="grid"):
                    if not data:
                        return "No data"
                    result = []
                    # Add headers
                    result.append(" | ".join(headers))
                    result.append("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
                    # Add rows
                    for row in data:
                        result.append(" | ".join(str(cell) for cell in row))
                    return "\n".join(result)
            
            print("\nSEARCH RESULTS:")
            print("-"*80)
            
            if results:
                # Prepare data for tabulate
                table_data = []
                for i, result in enumerate(results):
                    doc = result["rule"]
                    hybrid_score = result["hybrid_score"]
                    bm25_score = result["bm25_score"]
                    vector_score = result["similarity_score"]
                    table_data.append([
                        i+1,
                        doc.get("rule_number", "N/A"),
                        doc.get("title", "Untitled"),
                        doc.get("description", "")[:50] + "..." if len(doc.get("description", "")) > 50 else doc.get("description", ""),
                        f"{hybrid_score:.4f}",
                        f"{bm25_score:.4f}",
                        f"{vector_score:.4f}"
                    ])
                
                # Display table
                headers = ["#", "Rule #", "Title", "Description", "Hybrid Score", "BM25 Score", "Vector Score"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                print("No results found")
        
        # Convert to the expected return format (rule, score)
        formatted_results = [(result["rule"], result["hybrid_score"]) for result in results]
        
        print(f"Found {len(formatted_results)} rules with hybrid search")
        return formatted_results
        
    except Exception as e:
        print(f"Error performing hybrid search: {e}")
        import traceback
        traceback.print_exc()
        return []

async def query_by_rule_number(db, rule_number):
    """Query the database for a rule by its rule number."""
    collection = db.collection('rules')
    aql_query = f"""
    FOR doc IN {collection_name}
        FILTER doc.rule_number == @rule_number
        RETURN doc
    """
    cursor = collection.find({'rule_number': rule_number})
    results = await asyncio.to_thread(list, cursor)
    return results[0] if results else None


async def query_by_title(db, title):
    """Query the database for a rule by its title."""
    collection = db.collection('rules')
    cursor = collection.find({'title': title})
    results = await asyncio.to_thread(list, cursor)
    return results[0] if results else None


async def query_by_description(db, description):
    """Query the database for a rule by its description."""
    collection = db.collection('rules')
    cursor = collection.find({'description': description})
    results = await asyncio.to_thread(list, cursor)
    return results[0] if results else None

def main():
    """Main function."""
    print("Cursor Rules Database - Simple Version")
    
    config = {
        "arango_config": {
            "hosts": ["http://localhost:8529"],
            "username": "root", 
            "password": "openSesame"
        }
    }
    
    # Set up database
    rules_dir = ".cursor/rules"
    db = setup_cursor_rules_db(config, rules_dir)
    if not db:
        print("Failed to set up database")
        return 1
    
    # Get all rules
    rules = get_all_rules(db)
    if not rules:
        print("No rules found")
        return 1
    
    # Print rule information for each rule
    for rule in rules:
        print(f"\nRule {rule['rule_number']}: {rule['title']}")
        print(f"Type: {rule['rule_type']}")
        print(f"Applies to: {rule['glob_pattern']}")
        print(f"Description: {rule['description']}")
        
        # Get examples for the rule
        examples = get_examples_for_rule(db, rule["_key"])
        if examples:
            for i, example in enumerate(examples, 1):
                print(f"\nExample {i}: {example['title']}")
                print(f"Description: {example['description']}")
                print("\nGood Example:")
                print(example['good_example'])
                print("\nBad Example:")
                print(example['bad_example'])
    
    print("\nDemo completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 