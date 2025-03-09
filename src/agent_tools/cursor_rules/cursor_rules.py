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

# For embeddings
try:
    from fetch_page.embedding.embedding_utils import create_embedding_sync, ensure_text_has_prefix
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Embedding utilities not available. Vector search will be disabled.")

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

def load_rules_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load rules from markdown files in a directory.
    
    Args:
        directory_path: Path to directory containing rules
        
    Returns:
        List of rule dictionaries
    """
    rules = []
    
    try:
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory not found: {directory_path}")
            return []
        
        # Find all markdown files with pattern like 001-rule-name.md
        rule_files = list(directory.glob("*.md*"))
        
        for file in rule_files:
            print(f"Reading rule file: {file.name}")
            
            try:
                rule_key = file.stem
                
                # Extract rule number from filename
                rule_number_match = re.match(r"^(\d+)-", rule_key)
                rule_number = rule_number_match.group(1) if rule_number_match else "999"
                
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract title from H1 heading
                title_match = re.search(r"^# (.+?)$", content, re.MULTILINE)
                title = title_match.group(1) if title_match else rule_key
                
                # Extract description from text after H1
                desc_match = re.search(r"^# .+\n+(.+?)(\n\n|\n#)", content, re.MULTILINE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""
                
                # Prepare text for embedding
                embedding_text = f"RULE: {title}\nDESCRIPTION: {description}\nCONTENT: {content}"
                
                # Generate embedding if available
                embedding_result = generate_embedding(embedding_text)
                
                # Create rule document
                rule = {
                    "_key": rule_key,
                    "key": rule_key,
                    "title": title,
                    "description": description,
                    "glob_pattern": "*.py",  # Default, could be extracted from content
                    "priority": int(rule_number) * 10,  # Higher rule number = higher priority
                    "content": content,
                    "embedding": embedding_result["embedding"],
                    "embedding_metadata": embedding_result["metadata"]
                }
                
                rules.append(rule)
            except Exception as e:
                print(f"Error processing rule file {file.name}: {e}")
                continue
        
        print(f"Loaded {len(rules)} rules from {directory_path}")
        
    except Exception as e:
        print(f"Error loading rules from directory: {e}")
    
    return rules

def setup_cursor_rules_db(config: Dict[str, Any], rules_dir: str = None, db_name: str = "cursor_rules_test"):
    """
    Set up the cursor rules database with sample data.
    
    Args:
        config: Dictionary with database configuration
        rules_dir: Optional path to rules directory
        db_name: Name of the database to create/connect to
        
    Returns:
        Database handle if successful, None otherwise
    """
    try:
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

def get_examples_for_rule(db, rule_key):
    """
    Get examples for a specific rule.
    
    Args:
        db: Database handle
        rule_key: Rule key
        
    Returns:
        List of examples
    """
    try:
        # Check if examples collection exists
        if "examples" not in db.collections():
            return []
        
        # Get examples for rule
        query = """
        FOR example IN examples
            FILTER example.rule_key == @rule_key
            SORT example.created_at DESC
            RETURN example
        """
        results = db.aql.execute(query, bind_vars={"rule_key": rule_key})
        return list(results)
    except Exception as e:
        print(f"Error getting examples for rule: {e}")
        return []

def create_arangosearch_view(db, collection_name="rules", view_name="rules_view"):
    """
    Create or update an ArangoSearch view for the rules collection with stemming.
    
    Args:
        db: Database handle
        collection_name: Name of the collection
        view_name: Name of the ArangoSearch view
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if view already exists
        existing_views = db.views()
        view_exists = any(view["name"] == view_name for view in existing_views)
        
        if view_exists:
            print(f"ArangoSearch view '{view_name}' already exists")
            return True
        
        # Create a new ArangoSearch view with stemming
        view_properties = {
            "type": "arangosearch",
            "links": {
                collection_name: {
                    "includeAllFields": False,
                    "fields": {
                        "title": {
                            "analyzers": ["text_en"]  # English stemming
                        },
                        "description": {
                            "analyzers": ["text_en"]  # English stemming
                        },
                        "content": {
                            "analyzers": ["text_en"]  # English stemming
                        }
                    }
                }
            }
        }
        
        db.create_arangosearch_view(view_name, properties=view_properties)
        print(f"Successfully created ArangoSearch view '{view_name}' with stemming")
        return True
        
    except Exception as e:
        print(f"Error creating ArangoSearch view: {e}")
        return False

def bm25_keyword_search(db, query_text, limit=5):
    """
    Perform BM25 keyword search using an ArangoSearch view with stemming.
    Falls back to direct collection search if view creation fails.
    
    Args:
        db: Database handle
        query_text: Search query text
        limit: Maximum number of results
        
    Returns:
        List of matching rules with BM25 scores
    """
    try:
        # Ensure ArangoSearch view exists
        view_name = "rules_view"
        view_created = create_arangosearch_view(db, "rules", view_name)
        
        # Clean and prepare query
        search_tokens = []
        for token in query_text.lower().split():
            # Remove special characters
            token = ''.join(c for c in token if c.isalnum())
            if len(token) > 2:  # Ignore very short tokens
                search_tokens.append(token)
        
        # Join with OR for matching any term
        search_pattern = " OR ".join(search_tokens)
        
        # BM25 parameters
        k = 1.2  # Term frequency saturation
        b = 0.75  # Length normalization
        
        # Choose query based on whether view was created successfully
        if view_created:
            # Execute BM25 search with stemming through the view
            print(f"Running BM25 keyword search with stemming via ArangoSearch view")
            aql_query = """
            FOR doc IN rules_view
                SEARCH ANALYZER(
                    doc.title LIKE @search_tokens OR
                    doc.description LIKE @search_tokens OR
                    doc.content LIKE @search_tokens,
                    "text_en"
                )
                LET score = BM25(doc, @k, @b)
                SORT score DESC
                LIMIT @limit
                RETURN {
                    "rule": doc,
                    "score": score,
                    "matched_terms": TOKENS(@query, "text_en")
                }
            """
        else:
            # Fallback to direct text search without stemming
            print("Falling back to direct text search (without stemming)")
            aql_query = """
            FOR doc IN rules
                LET titleMatch = CONTAINS(LOWER(doc.title), LOWER(@query))
                LET descMatch = CONTAINS(LOWER(doc.description), LOWER(@query))
                LET contentMatch = CONTAINS(LOWER(doc.content), LOWER(@query))
                
                LET score = (
                    (titleMatch ? 1.0 : 0) + 
                    (descMatch ? 0.7 : 0) + 
                    (contentMatch ? 0.5 : 0)
                ) / 2.2
                
                FILTER score > 0
                SORT score DESC
                LIMIT @limit
                RETURN {
                    "rule": doc,
                    "score": score,
                    "matched_terms": []
                }
            """
        
        # Execute query
        bind_vars = {
            "query": query_text,
            "search_tokens": search_pattern,
            "k": k,
            "b": b,
            "limit": limit
        }
        
        print(f"Running keyword search for: {query_text}")
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        
        print(f"Found {len(results)} rules matching keywords")
        return results
        
    except Exception as e:
        print(f"Error performing keyword search: {e}")
        return []

def semantic_search(db, query_text, limit=5):
    """
    Perform semantic search using vector embeddings.
    
    Args:
        db: Database handle
        query_text: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching rules with scores
    """
    if not EMBEDDING_AVAILABLE:
        print("Semantic search not available - embedding utilities not found")
        return []
    
    try:
        # Generate embedding for the query
        # Using search_query prefix for queries
        query_text = "search_query: " + query_text
        query_embedding = create_embedding_sync(query_text)
        
        # Perform vector similarity search using AQL
        query = """
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
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "query_embedding": query_embedding["embedding"],
                "limit": limit
            }
        )
        results = list(cursor)
        
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

def hybrid_search(db, query_text, limit=5):
    """
    Perform hybrid search combining BM25 text search and vector similarity.
    Uses COSINE_SIMILARITY for vector similarity and text matching for BM25.
    
    Args:
        db: Database handle
        query_text: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching rules with combined scores
    """
    if not EMBEDDING_AVAILABLE:
        print("Hybrid search not available - embedding utilities not found")
        return []
    
    try:
        print(f"Starting hybrid search for query: '{query_text}'")
        
        # Generate embedding for the query
        query_with_prefix = "search_query: " + query_text
        query_embedding = create_embedding_sync(query_with_prefix)
        
        # Ensure we have a vector index on the embedding field
        collection_name = "rules"
        try:
            rules_collection = db.collection(collection_name)
            
            # Check if the vector index exists
            has_vector_index = False
            for idx in rules_collection.indexes():
                if idx.get("type") == "vector" and "embedding" in idx.get("fields", []):
                    has_vector_index = True
                    print(f"Found existing vector index: {idx['id']}")
                    break
            
            if not has_vector_index:
                print("Creating vector index on embedding field...")
                
                # First validate all embeddings
                if validate_embeddings(db, collection_name):
                    # Collection size for appropriate nLists calculation
                    collection_count = rules_collection.count()
                    nlist = max(1, min(collection_count, 100))
                    
                    # Creating vector index with appropriate parameters
                    vector_index = {
                        "type": "vector",
                        "fields": ["embedding"],
                        "params": {
                            "metric": "cosine",
                            "dimension": 768,
                            "nLists": nlist
                        }
                    }
                    
                    rules_collection.add_index(vector_index)
                    print("Vector index created successfully")
                    has_vector_index = True
                else:
                    print("Invalid embeddings found, cannot create vector index")
            
        except Exception as e:
            print(f"Warning: Error checking/creating vector index: {e}")
            has_vector_index = False
        
        # Try vector search with COSINE_SIMILARITY
        try:
            print("Running hybrid search with COSINE_SIMILARITY...")
            
            # Set search parameters
            similarity_threshold = 0.3  # Lower threshold to find more potential matches
            
            # Optimized AQL query using COSINE_SIMILARITY
            aql_query = """
            // Vector search results
            LET embedding_results = (
                FOR doc IN rules
                LET similarity = COSINE_SIMILARITY(doc.embedding, @queryVector)
                FILTER similarity >= @similarity_threshold
                SORT similarity DESC
                LIMIT @limit
                RETURN {
                    rule: doc,
                    vector_score: similarity,
                    text_score: 0
                }
            )
            
            // Text search results
            LET text_results = (
                FOR doc IN rules
                LET titleMatch = CONTAINS(LOWER(doc.title), LOWER(@searchText))
                LET descMatch = CONTAINS(LOWER(doc.description), LOWER(@searchText))
                LET contentMatch = CONTAINS(LOWER(doc.content), LOWER(@searchText))
                
                LET textScore = (
                    (titleMatch ? 1.0 : 0) + 
                    (descMatch ? 0.7 : 0) + 
                    (contentMatch ? 0.5 : 0)
                ) / 2.2
                
                FILTER textScore > 0
                SORT textScore DESC
                LIMIT @limit
                RETURN {
                    rule: doc,
                    vector_score: 0,
                    text_score: textScore
                }
            )
            
            // Merge and calculate hybrid scores
            LET merged_results = (
                FOR result IN UNION_DISTINCT(embedding_results, text_results)
                COLLECT key = result.rule._key INTO group
                LET doc = FIRST(group[*].result.rule)
                LET vector_score = MAX(group[*].result.vector_score)
                LET text_score = MAX(group[*].result.text_score)
                LET hybrid_score = (vector_score * 0.6) + (text_score * 0.4)
                
                RETURN {
                    rule: doc,
                    hybrid_score: hybrid_score,
                    vector_score: vector_score,
                    text_score: text_score
                }
            )
            
            // Sort by hybrid score and return
            FOR result IN merged_results
            SORT result.hybrid_score DESC
            LIMIT @limit
            RETURN result
            """
            
            cursor = db.aql.execute(
                aql_query,
                bind_vars={
                    "queryVector": query_embedding["embedding"],
                    "searchText": query_text.lower(),
                    "similarity_threshold": similarity_threshold,
                    "limit": limit * 2  # Get more results and then filter down
                }
            )
            
            results = list(cursor)
            
            if results:
                print(f"Found {len(results)} relevant rules via hybrid search (using COSINE_SIMILARITY)")
                
                # Final sorting and limiting to ensure best results
                results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                results = results[:limit]
                
                return results
            else:
                print("No results found using vector search, trying fallback...")
        except Exception as e:
            print(f"Vector search error with COSINE_SIMILARITY: {e}")
            print("Falling back to manual calculation")
        
        # Fallback to manual calculation if vector search fails or returns no results
        print("Performing hybrid search with manual calculation")
        
        # First try to get rules with text matches
        text_query = """
        FOR rule IN rules
            LET titleMatch = CONTAINS(LOWER(rule.title), LOWER(@searchText))
            LET descMatch = CONTAINS(LOWER(rule.description), LOWER(@searchText))
            LET contentMatch = CONTAINS(LOWER(rule.content), LOWER(@searchText))
            
            LET textScore = (
                (titleMatch ? 1.0 : 0) + 
                (descMatch ? 0.7 : 0) + 
                (contentMatch ? 0.5 : 0)
            ) / 2.2
            
            FILTER textScore > 0 OR @returnAll == true
            SORT textScore DESC
            RETURN {
                rule: rule,
                text_score: textScore
            }
        """
        
        cursor = db.aql.execute(
            text_query,
            bind_vars={
                "searchText": query_text.lower(),
                "returnAll": len(query_text.strip()) < 3  # Return all for very short queries
            }
        )
        
        results = list(cursor)
        
        # If no text matches, get all rules
        if not results:
            print("No text matches found, checking all rules for vector similarity")
            all_rules_query = "FOR rule IN rules RETURN { rule: rule, text_score: 0 }"
            cursor = db.aql.execute(all_rules_query)
            results = list(cursor)
        
        # Calculate vector similarity for each result
        for result in results:
            rule = result["rule"]
            rule_embedding = rule.get("embedding", [])
            
            # Skip rules without embeddings
            if not rule_embedding:
                result["vector_score"] = 0
                result["hybrid_score"] = result["text_score"] * 0.4  # Text only
                continue
                
            # Calculate cosine similarity
            try:
                # Get vector scores
                vec_a = np.array(query_embedding["embedding"])
                vec_b = np.array(rule_embedding)
                
                # Normalize vectors
                vec_a = vec_a / np.linalg.norm(vec_a)
                vec_b = vec_b / np.linalg.norm(vec_b)
                
                # Calculate cosine similarity
                vector_similarity = np.dot(vec_a, vec_b)
                
                # Compute hybrid score (60% vector, 40% text)
                text_score = result["text_score"]
                hybrid_score = (vector_similarity * 0.6) + (text_score * 0.4)
                
                result["vector_score"] = vector_similarity
                result["hybrid_score"] = hybrid_score
                
            except Exception as sim_error:
                print(f"Warning: Error calculating similarity: {sim_error}")
                result["vector_score"] = 0
                result["hybrid_score"] = result["text_score"] * 0.4  # Text only
        
        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Limit results
        results = results[:limit]
        
        print(f"Found {len(results)} relevant rules via hybrid search (manual calculation)")
        return results
            
    except Exception as e:
        print(f"Error performing hybrid search: {e}")
        # Fall back to pure vector search on error
        return semantic_search(db, query_text, limit)

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