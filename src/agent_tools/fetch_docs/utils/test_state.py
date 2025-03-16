#!/usr/bin/env python3
"""
Test State Storage Utilities

This module provides functions to store and retrieve test states in the ArangoDB database.
These functions allow the AI agent to record and query the history of test runs,
creating a persistent record that can be referenced in future sessions.

Documentation References:
- python-arango: https://python-arango.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import asyncio
import logging
import os
import sys
import platform
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Import existing utility functions to leverage them
from ..core.cursor_rules import (
    hybrid_search,
    bm25_keyword_search,
    semantic_search,
    setup_cursor_rules_db
)

# Updated import path to match new file structure
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

logger = logging.getLogger(__name__)

# Collection names
TEST_STATES_COLLECTION = 'test_states'
TEST_RUNS_COLLECTION = 'test_runs'
TEST_HISTORY_COLLECTION = 'test_history'
TEST_FAILURES_COLLECTION = 'test_failures'

async def ensure_test_collections(db):
    """
    Ensure that all test-related collections exist in the database.
    Leverages existing ArangoDB patterns.
    
    Args:
        db: ArangoDB database connection
    
    Returns:
        Dict of collections
    """
    # Check if collections exist and create them if they don't
    collections = await asyncio.to_thread(db.collections)
    collection_names = [c['name'] for c in collections]
    
    collections_to_ensure = [
        TEST_STATES_COLLECTION,
        TEST_RUNS_COLLECTION,
        TEST_HISTORY_COLLECTION,
        TEST_FAILURES_COLLECTION
    ]
    
    result = {}
    for collection_name in collections_to_ensure:
        if collection_name not in collection_names:
            logger.info(f"Creating {collection_name} collection")
            await asyncio.to_thread(db.create_collection, collection_name)
        result[collection_name] = db.collection(collection_name)
    
    # Create ArangoSearch view for test states if it doesn't exist
    views = await asyncio.to_thread(db.views)
    view_names = [v['name'] for v in views]
    
    # Create view properties with proper analyzers for text search
    view_properties = {
        "links": {
            TEST_STATES_COLLECTION: {
                "fields": {
                    "tag_name": {
                        "analyzers": ["text_en"]
                    },
                    "notes": {
                        "analyzers": ["text_en"]
                    },
                    "failing_tests": {
                        "analyzers": ["text_en"]
                    },
                    "cli_implementation": {
                        "analyzers": ["text_en"]
                    },
                    "environment.platform": {
                        "analyzers": ["text_en"]
                    },
                    "environment.python_version": {
                        "analyzers": ["text_en"]
                    }
                }
            },
            TEST_FAILURES_COLLECTION: {
                "fields": {
                    "test_name": {
                        "analyzers": ["text_en"]
                    },
                    "error_message": {
                        "analyzers": ["text_en"]
                    },
                    "analysis": {
                        "analyzers": ["text_en"]
                    }
                }
            }
        }
    }
    
    # Check if view exists
    if 'test_states_view' not in view_names:
        logger.info("Creating test_states_view")
        await asyncio.to_thread(
            lambda: db.create_arangosearch_view('test_states_view', view_properties)
        )
    else:
        # Update view if it exists but ensure it has the right configuration
        logger.info("Updating test_states_view")
        await asyncio.to_thread(
            lambda: db.update_view('test_states_view', view_properties)
        )
    
    return result

async def ensure_test_states_collection(db):
    """
    Ensure that the test_states collection exists in the database.
    
    Args:
        db: ArangoDB database connection
    
    Returns:
        The test_states collection
    """
    # Check if collection exists and create it if it doesn't
    collections = await asyncio.to_thread(db.collections)
    collection_names = [c['name'] for c in collections]
    
    if TEST_STATES_COLLECTION not in collection_names:
        logger.info(f"Creating {TEST_STATES_COLLECTION} collection")
        await asyncio.to_thread(db.create_collection, TEST_STATES_COLLECTION)
    
    return db.collection(TEST_STATES_COLLECTION)

async def get_environment_info() -> Dict[str, Any]:
    """
    Get current environment information.
    Uses existing system libraries without adding new dependencies.
    
    Returns:
        Dict with environment information
    """
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith(('PYTHON', 'PATH', 'VIRTUAL_ENV', 'CONDA'))
        },
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd()
    }

async def store_test_state(db, tag_name: str, test_results: Dict[str, Any], notes: Optional[str] = None):
    """
    Store the current test state in the database.
    Now enhanced to store more structured information and relationships.
    
    Args:
        db: ArangoDB database connection
        tag_name: Name of the git tag associated with this test state
        test_results: Dictionary containing test results information
        notes: Optional notes about this test state
    
    Returns:
        The document ID of the stored test state
    """
    # Ensure collections exist
    collections = await ensure_test_collections(db)
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Get environment information
    environment = await get_environment_info()
    
    # Create document with test state information
    document = {
        "_key": run_id,
        "timestamp": timestamp,
        "tag_name": tag_name,
        "tests_total": test_results.get("total", 0),
        "tests_passed": test_results.get("passed", 0),
        "tests_failed": test_results.get("failed", 0),
        "failing_tests": test_results.get("failing_tests", []),
        "cli_implementation": test_results.get("cli_implementation", ""),
        "notes": notes or "",
        "environment": environment
    }
    
    # Insert document using asyncio.to_thread to avoid blocking
    result = await asyncio.to_thread(collections[TEST_STATES_COLLECTION].insert, document)
    logger.info(f"Stored test state with ID: {result['_id']}")
    
    # Store detailed information about failing tests
    if test_results.get("failing_tests"):
        for test_name in test_results["failing_tests"]:
            # Extract error details if available
            error_details = test_results.get("failure_details", {}).get(test_name, {})
            error_message = error_details.get("error_message", "Unknown error")
            
            # Generate analysis of the failure using existing information
            failure_analysis = f"Test '{test_name}' failed during run '{run_id}'. "
            if "traceback" in error_details:
                failure_analysis += f"The error occurred in {error_details.get('file', 'unknown file')}."
            
            failure_doc = {
                "run_id": run_id,
                "test_name": test_name,
                "timestamp": timestamp,
                "error_message": error_message,
                "traceback": error_details.get("traceback", ""),
                "analysis": failure_analysis,
                "environment": environment
            }
            
            await asyncio.to_thread(collections[TEST_FAILURES_COLLECTION].insert, failure_doc)
            
    # Store test history records for comparison
    for test_name in test_results.get("test_names", []):
        is_failing = test_name in test_results.get("failing_tests", [])
        
        history_doc = {
            "run_id": run_id,
            "test_name": test_name,
            "timestamp": timestamp,
            "tag_name": tag_name,
            "passed": not is_failing,
            "execution_time": test_results.get("execution_times", {}).get(test_name, 0)
        }
        
        await asyncio.to_thread(collections[TEST_HISTORY_COLLECTION].insert, history_doc)
    
    return result

async def get_test_state(db, tag_name: Optional[str] = None, limit: int = 1):
    """
    Retrieve test states from the database.
    Now enhanced to use hybrid search when appropriate.
    
    Args:
        db: ArangoDB database connection
        tag_name: Optional tag name to filter by
        limit: Maximum number of test states to retrieve (default: 1)
    
    Returns:
        List of test state documents
    """
    # Ensure test_states collection exists
    collection = await ensure_test_states_collection(db)
    
    # Build query based on parameters
    if tag_name:
        query = f"FOR doc IN test_states FILTER doc.tag_name == @tag_name SORT doc.timestamp DESC LIMIT {limit} RETURN doc"
        bind_vars = {"tag_name": tag_name}
    else:
        query = f"FOR doc IN test_states SORT doc.timestamp DESC LIMIT {limit} RETURN doc"
        bind_vars = {}
    
    # Execute query using asyncio.to_thread
    cursor = await asyncio.to_thread(db.aql.execute, query, bind_vars=bind_vars)
    results = await asyncio.to_thread(list, cursor)
    
    # For each test state, fetch related failure details
    for result in results:
        if "failing_tests" in result and result["failing_tests"]:
            run_id = result.get("_key")
            if run_id:
                # Fetch failure details
                failure_query = f"FOR doc IN {TEST_FAILURES_COLLECTION} FILTER doc.run_id == @run_id RETURN doc"
                failure_vars = {"run_id": run_id}
                failure_cursor = await asyncio.to_thread(db.aql.execute, failure_query, bind_vars=failure_vars)
                failure_details = await asyncio.to_thread(list, failure_cursor)
                result["failure_details"] = failure_details
    
    return results

async def search_test_states(db, query: str, limit: int = 5):
    """
    Search for test states using hybrid search.
    Leverages existing hybrid_search function.
    
    Args:
        db: ArangoDB database connection
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of matching test states
    """
    # Ensure view exists
    await ensure_test_collections(db)
    
    # Use existing hybrid_search function to search test states
    search_results = await hybrid_search(
        db,
        query,
        "test_states",
        limit,
        False
    )
    
    # Transform results to be consistent with get_test_state
    results = []
    for doc in search_results:
        # Fetch full test state document
        if "_id" in doc:
            doc_id = doc["_id"].split("/")[1]
            state_query = f"FOR doc IN {TEST_STATES_COLLECTION} FILTER doc._key == @doc_id RETURN doc"
            state_vars = {"doc_id": doc_id}
            state_cursor = await asyncio.to_thread(db.aql.execute, state_query, bind_vars=state_vars)
            states = await asyncio.to_thread(list, state_cursor)
            if states:
                results.append(states[0])
    
    return results

async def get_all_test_states(db):
    """
    Retrieve all test states from the database.
    
    Args:
        db: ArangoDB database connection
    
    Returns:
        List of all test state documents
    """
    # Ensure test_states collection exists
    collection = await ensure_test_states_collection(db)
    
    # Execute query to get all test states, sorted by timestamp
    query = "FOR doc IN test_states SORT doc.timestamp DESC RETURN doc"
    
    # Execute query using asyncio.to_thread
    cursor = await asyncio.to_thread(db.aql.execute, query)
    results = await asyncio.to_thread(list, cursor)
    
    return results

async def get_test_history(db, test_name: str, limit: int = 10):
    """
    Get history for a specific test.
    
    Args:
        db: ArangoDB database connection
        test_name: Name of the test to get history for
        limit: Maximum number of history records to return
        
    Returns:
        List of test history records
    """
    # Ensure collections exist
    await ensure_test_collections(db)
    
    # Execute query to get test history
    query = f"""
    FOR doc IN {TEST_HISTORY_COLLECTION}
    FILTER doc.test_name == @test_name
    SORT doc.timestamp DESC
    LIMIT {limit}
    RETURN doc
    """
    bind_vars = {"test_name": test_name}
    
    # Execute query using asyncio.to_thread
    cursor = await asyncio.to_thread(db.aql.execute, query, bind_vars=bind_vars)
    results = await asyncio.to_thread(list, cursor)
    
    return results

async def compare_test_runs(db, run_id1: str, run_id2: str):
    """
    Compare two test runs.
    
    Args:
        db: ArangoDB database connection
        run_id1: ID of first test run
        run_id2: ID of second test run
        
    Returns:
        Comparison results
    """
    # Ensure collections exist
    await ensure_test_collections(db)
    
    # Get test states for both runs
    query1 = f"FOR doc IN {TEST_STATES_COLLECTION} FILTER doc._key == @run_id RETURN doc"
    query2 = f"FOR doc IN {TEST_STATES_COLLECTION} FILTER doc._key == @run_id RETURN doc"
    
    # Execute queries using asyncio.to_thread
    cursor1 = await asyncio.to_thread(db.aql.execute, query1, bind_vars={"run_id": run_id1})
    cursor2 = await asyncio.to_thread(db.aql.execute, query2, bind_vars={"run_id": run_id2})
    
    # Get results using asyncio.to_thread
    run1 = await asyncio.to_thread(list, cursor1)
    run2 = await asyncio.to_thread(list, cursor2)
    
    if not run1 or not run2:
        return {"error": "One or both run IDs not found"}
        
    run1 = run1[0]
    run2 = run2[0]
    
    # Compare basic statistics
    comparison = {
        "run1": {
            "id": run_id1,
            "timestamp": run1.get("timestamp"),
            "tag_name": run1.get("tag_name"),
            "total": run1.get("tests_total", 0),
            "passed": run1.get("tests_passed", 0),
            "failed": run1.get("tests_failed", 0)
        },
        "run2": {
            "id": run_id2,
            "timestamp": run2.get("timestamp"),
            "tag_name": run2.get("tag_name"),
            "total": run2.get("tests_total", 0),
            "passed": run2.get("tests_passed", 0),
            "failed": run2.get("tests_failed", 0)
        },
        "diff": {
            "total": run2.get("tests_total", 0) - run1.get("tests_total", 0),
            "passed": run2.get("tests_passed", 0) - run1.get("tests_passed", 0),
            "failed": run2.get("tests_failed", 0) - run1.get("tests_failed", 0)
        }
    }
    
    # Compare failing tests
    failing1 = set(run1.get("failing_tests", []))
    failing2 = set(run2.get("failing_tests", []))
    
    comparison["failing_tests"] = {
        "fixed": list(failing1 - failing2),  # Tests that were failing in run1 but not in run2
        "new": list(failing2 - failing1),    # Tests that were not failing in run1 but are in run2
        "persistent": list(failing1 & failing2)  # Tests failing in both runs
    }
    
    # Compare environment
    env1 = run1.get("environment", {})
    env2 = run2.get("environment", {})
    
    # Identify environment differences
    env_diff = {}
    for key in set(env1.keys()) | set(env2.keys()):
        if key in env1 and key in env2 and env1[key] != env2[key]:
            env_diff[key] = {
                "run1": env1[key],
                "run2": env2[key]
            }
    
    comparison["environment_diff"] = env_diff
    
    return comparison

async def get_timeline_view(db, test_names: Optional[List[str]] = None, days: int = 30):
    """
    Generate a timeline view of test executions.
    
    Args:
        db: ArangoDB database connection
        test_names: Optional list of test names to filter by
        days: Number of days to include in the timeline (default: 30)
    
    Returns:
        Dict with timeline data
    """
    # TODO: Implement timeline view
    return {"status": "Not implemented yet"}

async def _search_test_failures(db, query, limit=5):
    """
    Search for test failures using hybrid search combining BM25 and vector similarity.
    
    This function uses the established hybrid search pattern from cursor_rules.py:
    1. Uses BM25 for keyword search with the text_en analyzer
    2. Uses vector similarity for semantic search
    3. Combines and ranks results with a weighted score
    
    Args:
        db: ArangoDB database connection
        query: Search query to find relevant test failures
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        List of test failure documents with relevance scores
    """
    import textwrap
    from agent_tools.cursor_rules.core.cursor_rules import generate_embedding
    
    # Ensure collections exist and view is updated with the right analyzers
    await ensure_test_collections(db)
    
    try:
        # Generate embedding for the query
        query_embedding = await asyncio.to_thread(generate_embedding, query)
        if not query_embedding or "embedding" not in query_embedding:
            # Fall back to BM25 only search if embedding generation fails
            print("Failed to generate embedding for query. Falling back to BM25 search only.")
            # Simplified BM25 search
            aql_query = f"""
            FOR doc IN test_states_view
                SEARCH ANALYZER(
                    doc.test_name IN TOKENS(@query, "text_en") OR
                    doc.error_message IN TOKENS(@query, "text_en") OR
                    doc.analysis IN TOKENS(@query, "text_en"),
                    "text_en"
                )
                FILTER IS_DOCUMENT(doc) AND STARTS_WITH(doc._id, 'test_failures/')
                LET score = BM25(doc)
                SORT score DESC
                LIMIT {limit}
                RETURN {{
                    "doc": doc,
                    "score": score
                }}
            """
            
            cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars={"query": query})
            results = await asyncio.to_thread(list, cursor)
            
            return [{"test_name": r["doc"].get("test_name"),
                    "timestamp": r["doc"].get("timestamp"),
                    "error_message": r["doc"].get("error_message"),
                    "analysis": r["doc"].get("analysis"),
                    "score": r["score"]} for r in results]
        
        # Parameters for search
        view_name = "test_states_view"
        embedding_similarity_threshold = 0.5
        bm25_threshold = 0.0
        k1 = 1.2  # Term frequency saturation
        b = 0.75  # Length normalization factor
        
        # Use the same hybrid search pattern as in cursor_rules.py with separate subqueries
        aql_query = textwrap.dedent(f"""
        LET embedding_results = (
            FOR doc IN test_failures
                FILTER doc.embedding != null
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
            FOR doc IN {view_name}
                SEARCH ANALYZER(
                    doc.test_name IN TOKENS(@search_text, "text_en") OR
                    doc.error_message IN TOKENS(@search_text, "text_en") OR
                    doc.analysis IN TOKENS(@search_text, "text_en"),
                    "text_en"
                )
                FILTER IS_DOCUMENT(doc) AND STARTS_WITH(doc._id, 'test_failures/')
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
                RETURN {{
                    "doc": doc,
                    "_key": key,
                    "similarity_score": similarity_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score
                }}
        )
        
        // Sort and limit merged results
        FOR result IN merged_results
            SORT result.hybrid_score DESC
            LIMIT @limit
            RETURN result
        """)
        
        # Execute query
        bind_vars = {
            "search_text": query,
            "query_vector": query_embedding["embedding"],
            "embedding_similarity_threshold": embedding_similarity_threshold,
            "bm25_threshold": bm25_threshold,
            "k1": k1,
            "b": b,
            "limit": limit
        }
        
        cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars=bind_vars)
        results = await asyncio.to_thread(list, cursor)
        
        # Return formatted results
        return [{"test_name": r["doc"].get("test_name"),
                 "timestamp": r["doc"].get("timestamp"),
                 "error_message": r["doc"].get("error_message"),
                 "analysis": r["doc"].get("analysis"),
                 "bm25_score": r["bm25_score"],
                 "vector_score": r["similarity_score"],
                 "score": r["hybrid_score"]} for r in results]
                 
    except Exception as e:
        print(f"Error performing hybrid search: {e}")
        import traceback
        traceback.print_exc()
        return [] 