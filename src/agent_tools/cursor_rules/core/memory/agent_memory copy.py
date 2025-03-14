#!/usr/bin/env python3
"""
Agent Memory System - Core Implementation

This module implements a memory system for AI agents that handles:
1. Memory persistence with importance-based decay
2. Recency boosting for recently accessed information
3. Knowledge correction through upserts
4. Confidence scoring for facts

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
- ArangoDB Vector Search: https://www.arangodb.com/docs/stable/aql/functions-vector.html
- Pydantic: https://docs.pydantic.dev/
"""

import datetime
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple

from arango import ArangoClient
from loguru import logger
from pydantic import BaseModel, Field
from agent_tools.cursor_rules.embedding.embedding_utils import create_embedding_sync
from agent_tools.cursor_rules.utils.vector_utils import format_embedding_for_debug, get_vector_stats, truncate_vector_for_display

# Configuration with added domain boost parameter.
DEFAULT_CONFIG = {
    "arango_host": "http://localhost:8529",
    "db_name": "agent_memory_db",
    "facts_collection": "agent_facts",
    "associations_collection": "agent_associations",
    "username": "root",
    "password": "openSesame",
    "default_ttl_days": 30,
    "importance_decay_factor": 0.5,
    "recency_boost_factor": 0.4,
    "confidence_threshold": 0.3,
    "default_preserved_domains": ["physics"],  # Default domains to preserve/boost in search
    "preserved_domain_boost": 0.5  # Increased boost for preserved domains
}


class MemoryFact(BaseModel):
    """Schema for a memory fact stored in the database."""
    fact_id: str = Field(..., description="Unique identifier for this fact")
    content: str = Field(..., description="The actual content of the fact")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score from 0-1")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score from 0-1")
    created_at: str = Field(..., description="ISO timestamp of creation")
    last_accessed: str = Field(..., description="ISO timestamp of last access")
    access_count: int = Field(0, ge=0, description="Number of times this fact was accessed")
    ttl_days: float = Field(..., ge=0, description="Time-to-live in days")
    domains: List[str] = Field([], description="Knowledge domains this fact belongs to")
    source: Optional[str] = Field(None, description="Source of this information")
    # Optional fields for knowledge correction
    previous_content: Optional[str] = Field(None, description="Previous version if updated")
    updated_at: Optional[str] = Field(None, description="When this fact was last updated")
    # For associative memory
    related_facts: List[str] = Field([], description="IDs of related facts")


class AgentMemorySystem:
    """
    Agent memory system implementation that manages fact storage, retrieval,
    and memory management features like decay and associations.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory system with configuration."""
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.client = None
        self.db = None
        self.facts = None
        self.associations = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize database connection and ensure collections exist.
        Returns True if successful, False otherwise.
        """
        try:
            self.client = ArangoClient(hosts=self.config["arango_host"])
            # Connect to the system database
            sys_db = self.client.db(
                "_system",
                username=self.config["username"],
                password=self.config["password"]
            )
            # Check if our DB exists; create if not
            if not sys_db.has_database(self.config["db_name"]):
                sys_db.create_database(self.config["db_name"])
            # Connect to our database
            self.db = self.client.db(
                self.config["db_name"],
                username=self.config["username"],
                password=self.config["password"]
            )
            # Set up collections and views
            self._setup_collections()
            self.initialized = True
            logger.info(f"Agent memory system initialized with database {self.config['db_name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize agent memory system: {e}")
            return False

    def _setup_collections(self) -> None:
        """Set up required collections and indexes."""
        self._setup_facts_collection()
        self._setup_associations_collection()

    def _setup_facts_collection(self) -> None:
        """Set up the facts collection with proper indexes."""
        facts_name = self.config["facts_collection"]
        if not self.db.has_collection(facts_name):
            # Create collection
            self.facts = self.db.create_collection(facts_name)
            # Create indexes using the recommended add_index method from v8.1.6
            self.facts.add_index({"type": "persistent", "fields": ["fact_id"], "unique": True})
            self.facts.add_index({"type": "persistent", "fields": ["importance"]})
            self.facts.add_index({"type": "persistent", "fields": ["confidence"]})
            self.facts.add_index({"type": "persistent", "fields": ["last_accessed"]})
            self.facts.add_index({"type": "persistent", "fields": ["access_count"]})
            self.facts.add_index({"type": "persistent", "fields": ["ttl_days"]})
            self.facts.add_index({"type": "persistent", "fields": ["domains"]})
            
            # Create ArangoSearch view for text search
            view_name = f"{facts_name}_view"
            self._create_search_view(facts_name, view_name)
        else:
            self.facts = self.db.collection(facts_name)

    def _create_search_view(self, collection_name: str, view_name: str) -> None:
        """
        Create an ArangoSearch view for the collection.
        
        This function attempts to create a search view for text-based searches.
        If creation fails, it logs the error but continues, as the system can
        still function with reduced search capabilities.
        
        Args:
            collection_name: Name of the collection to create a view for
            view_name: Name to give the new view
        
        Returns:
            None
        """
        try:
            views = self.db.views()
            view_exists = any(v["name"] == view_name for v in views)
            if not view_exists:
                try:
                    self.db.create_arangosearch_view(
                        view_name,
                        properties={
                            "links": {
                                collection_name: {
                                    "fields": {
                                        "content": {
                                            "analyzers": ["text_en"]
                                        }
                                    }
                                }
                            }
                        }
                    )
                    logger.info(f"Created search view {view_name} for collection {collection_name}")
                except Exception as e:
                    logger.error(f"Failed to create search view: {e}")
                    logger.warning("The system will continue with reduced search capabilities")
                    # Log a more detailed error message with potential troubleshooting steps
                    logger.debug(f"Search view creation error: {str(e)}")
                    logger.debug("This may be due to missing text analyzers. The system will continue with basic functionality.")
            else:
                logger.info(f"Search view {view_name} already exists")
        except Exception as e:
            logger.error(f"Error checking views: {e}")
            logger.warning("The system will continue with reduced search capabilities")

    def _setup_associations_collection(self) -> None:
        """Set up the associations edge collection with proper indexes."""
        assoc_name = self.config["associations_collection"]
        if not self.db.has_collection(assoc_name):
            self.associations = self.db.create_collection(assoc_name, edge=True)
            self.associations.add_index({"type": "persistent", "fields": ["weight"]})
            self.associations.add_index({"type": "persistent", "fields": ["type"]})
        else:
            self.associations = self.db.collection(assoc_name)

    def remember(self, content: str, importance: float = 0.5, 
        confidence: float = 0.7, domains: List[str] = None,
        source: str = None, fact_id: str = None, ttl_days: float = None,
        glossary_terms: List[str] = None, correction_history: List[Dict] = None,
        alternatives: List[Dict] = None, resolution_notes: str = None,
        resolved_from: str = None, merged_from: List[str] = None,
        merge_notes: str = None) -> Dict[str, Any]:
        """
        Store a new fact in memory or update an existing fact if it exists.
        Uses an upsert operation for knowledge correction.

        Args:
            content: The fact content to remember
            importance: How important this fact is (0-1)
            confidence: Confidence in this fact's accuracy (0-1)
            domains: Knowledge domains this fact belongs to
            source: Source of this information
            fact_id: Optional custom ID for this fact
            ttl_days: Optional TTL in days (if not provided, calculated from importance)
            glossary_terms: Optional list of terms to index for glossary search
            correction_history: List of previous versions of this fact
            alternatives: List of alternative facts with lower confidence
            resolution_notes: Notes about how contradictions were resolved
            resolved_from: ID of the fact this resolves
            merged_from: List of fact IDs that were merged to create this fact
            merge_notes: Notes about why facts were merged

        Returns:
            Dict with operation result
        
        Documentation references:
        - ArangoDB Python Driver: https://python-arango.readthedocs.io/
        - ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
        """
        if not self.initialized:
            self.initialize()

        current_time = datetime.datetime.now().isoformat()

        # Generate fact_id if not provided
        if not fact_id:
            hash_obj = hashlib.md5(content.encode())
            fact_id = f"fact_{hash_obj.hexdigest()[:10]}"

        # Calculate TTL based on importance if not provided
        if ttl_days is None:
            ttl_days = self.config["default_ttl_days"] * (1 + importance)
            
        # Generate embedding for semantic search
        embedding_data = self._get_embedding(content)
        embedding = embedding_data.get("embedding") if embedding_data else None
        
        # Extract glossary terms if not provided
        if glossary_terms is None:
            glossary_terms = []
            # Simple extraction of potential glossary terms (capitalized words)
            words = content.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    glossary_terms.append(word.strip('.,;:()[]{}""\''))

        # Create a fact structure with all fields
        fact = {
            "fact_id": fact_id,
            "content": content,
            "importance": importance,
            "confidence": confidence,
            "created_at": current_time,
            "last_accessed": current_time,
            "access_count": 1,
            "ttl_days": ttl_days,
            "domains": domains or [],
            "source": source,
            "related_facts": [],
            "embedding": embedding,
            "glossary_terms": glossary_terms,
            # Add new fields for knowledge correction
            "correction_history": correction_history or [],
            "alternatives": alternatives or [],
            "resolution_notes": resolution_notes,
            "resolved_from": resolved_from,
            "merged_from": merged_from or [],
            "merge_notes": merge_notes
        }

        logger.info(f"Attempting to store fact: {fact_id}")
        # Create a debug-friendly version of fact details with truncated embedding
        debug_fact = fact.copy()
        if embedding is not None:
            debug_fact["embedding"] = format_embedding_for_debug({"embedding": embedding})
        logger.debug(f"Fact details: {debug_fact}")

        # Note: collection names cannot be bound so we use an f-string to inject it
        aql_query = f"""
        UPSERT {{ fact_id: @fact_id }}
        INSERT @fact
        UPDATE {{
            content: @content,
            importance: @importance,
            confidence: @confidence,
            previous_content: OLD.content,
            access_count: OLD.access_count + 1,
            last_accessed: @current_time,
            updated_at: @current_time,
            domains: @domains,
            source: @source,
            ttl_days: @ttl_days,
            embedding: @embedding,
            glossary_terms: @glossary_terms,
            correction_history: @correction_history,
            alternatives: @alternatives,
            resolution_notes: @resolution_notes,
            resolved_from: @resolved_from,
            merged_from: @merged_from,
            merge_notes: @merge_notes
        }} IN {self.config['facts_collection']}
        RETURN {{ new: NEW, old: OLD, operation: OLD ? 'update' : 'insert' }}
        """

        bind_vars = {
            "fact_id": fact_id,
            "fact": fact,
            "content": content,
            "importance": importance,
            "confidence": confidence,
            "current_time": current_time,
            "domains": domains or [],
            "source": source,
            "ttl_days": ttl_days,
            "embedding": embedding,
            "glossary_terms": glossary_terms or [],
            "correction_history": correction_history or [],
            "alternatives": alternatives or [],
            "resolution_notes": resolution_notes,
            "resolved_from": resolved_from,
            "merged_from": merged_from or [],
            "merge_notes": merge_notes
        }

        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            result = next(cursor, None)
            # Create a debug-friendly version of the result for logging
            log_result = result.copy() if result else None
            if log_result and 'new' in log_result and 'embedding' in log_result['new']:
                log_result['new'] = log_result['new'].copy()
                log_result['new']['embedding'] = truncate_vector_for_display(log_result['new']['embedding'], max_items=3)
            logger.info(f"Successfully stored fact {fact_id}: {log_result}")
            return result
        except Exception as e:
            logger.error(f"Failed to store fact {fact_id}: {e}")
            raise

    def recall(
        self, 
        query: str, 
        threshold: float = 0.4, 
        boost_recency: bool = True, limit: int = 5, 
        domain_filter: Optional[List[str]] = None,
        semantic: bool = True,
           bm25: bool = True,
           glossary: bool = True) -> List[Dict[str, Any]]:
        """
        Recall facts from memory using configurable hybrid search components.
        This query unifies the semantic, BM25, and glossary components.
        Domain search is applied in Python after retrieving candidates, and 
        if no results are found, a keyword (domain) search fallback is used.
        """
        if not self.initialized:
            self.initialize()

        if not query or query.strip() == "":
            return []

        # Normalize domain_filter for consistency and logging
        if domain_filter:
            domain_filter = [d.lower() for d in domain_filter]
            logger.debug(f"Recall: Using domain filter (normalized): {domain_filter}")

        # Validate that at least one search component is enabled
        if not any([semantic, bm25, glossary]):
            logger.warning("No search components enabled, defaulting to BM25 search")
            bm25 = True  # Default to BM25 if nothing is enabled

        current_time = datetime.datetime.now().isoformat()
        recency_factor = self.config["recency_boost_factor"] if boost_recency else 0

        # Search parameters
        k1 = 1.2  # BM25 term frequency saturation
        b = 0.75  # BM25 length normalization factor
        bm25_threshold = 0.1
        embedding_threshold = 0.5
        glossary_threshold = 0.3

        # Generate embedding for semantic search if enabled
        query_embedding = None
        if semantic:
            try:
                query_embedding = self._get_embedding(query)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                query_embedding = None

        # Determine if we have a valid embedding for semantic search.
        valid_semantic = (
            semantic and query_embedding and 
            query_embedding.get("embedding") and 
            len(query_embedding["embedding"]) > 0
        )
        if semantic and not valid_semantic:
            logger.warning("No valid embedding for semantic search, disabling semantic component")
            semantic = False

        # Calculate component weights based on enabled components
        total_components = sum([semantic, bm25, glossary])
        base_weight = 0.8 / total_components if total_components > 0 else 0  # Reserve 0.2 for importance and recency
        semantic_weight = base_weight if semantic else 0
        bm25_weight = base_weight if bm25 else 0
        glossary_weight = base_weight if glossary else 0
        importance_weight = 0.1
        recency_weight = 0.1

        # Fetch more results to account for domain filtering
        fetch_limit = limit * 3 if domain_filter else limit
        
        # Build the hybrid search query with NO domain filtering in AQL
        aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                // Calculate scores from different search methods
                LET similarity_score = {f'''(
                    COSINE_SIMILARITY(fact.embedding, @query_embedding) >= @embedding_threshold
                    ? COSINE_SIMILARITY(fact.embedding, @query_embedding)
                    : 0
                )''' if valid_semantic else '0'}
                
                LET bm25_score = {f'''(
                    FOR doc IN {self.config['facts_collection']}_view
                        SEARCH ANALYZER(
                            BOOST(doc.content IN TOKENS(@query, "text_en"), 1.0),
                            "text_en"
                        ) AND doc._key == fact._key
                        RETURN BM25(doc, @k1, @b)
                )[0] || 0''' if bm25 else '0'}
                
                LET glossary_matches = {f'''(
                    FOR term IN fact.glossary_terms || []
                        FILTER LOWER(term) IN TOKENS(LOWER(@query), "text_en")
                        RETURN 1
                )''' if glossary else '[]'}
                
                LET glossary_score = {f'''(
                    LENGTH(glossary_matches) > 0 
                    ? MIN([LENGTH(glossary_matches) / LENGTH(TOKENS(@query, "text_en"))])
                    : 0
                )''' if glossary else '0'}
                
                // Calculate recency and hybrid scores
                LET days_since_access = DATE_DIFF(fact.last_accessed, @current_time, "d")
                LET days_factor = 1.0 / (1.0 + ABS(days_since_access))
                LET recency_boost = (days_factor * @recency_factor)
                
                LET hybrid_score = (
                    (similarity_score * @semantic_weight) + 
                    (bm25_score * @bm25_weight) +
                    (glossary_score * @glossary_weight) +
                    (fact.importance * @importance_weight) +
                    (recency_boost * @recency_weight)
                )
                
                FILTER hybrid_score >= @threshold
                
                SORT hybrid_score DESC
                LIMIT @limit
                
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance,
                    confidence: fact.confidence,
                    domains: fact.domains || [],
                    score: hybrid_score,
                    components: {{
                        semantic_score: similarity_score,
                        bm25_score: bm25_score,
                        glossary_score: glossary_score,
                        importance_boost: fact.importance,
                        recency_boost: recency_boost
                    }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count,
                    source: fact.source,
                    glossary_terms: fact.glossary_terms || []
                }}
        """

        # Set up bind variables.
        bind_vars = {
            "query": query,
            "current_time": current_time,
            "recency_factor": recency_factor,
            "limit": fetch_limit,
            "threshold": threshold,
            "bm25_threshold": bm25_threshold,
            "glossary_threshold": glossary_threshold,
            "k1": k1,
            "b": b,
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "glossary_weight": glossary_weight,
            "importance_weight": importance_weight,
            "recency_weight": recency_weight,
            "embedding_threshold": embedding_threshold
        }
        # Only add the query_embedding bind variable if we are doing semantic search.
        if valid_semantic:
            bind_vars["query_embedding"] = query_embedding["embedding"]
            logger.debug(f"Using valid embedding vector with {len(query_embedding['embedding'])} dimensions")

        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            candidates = list(cursor)
            
            logger.debug(f"Raw query returned {len(candidates)} candidates before domain filtering")
            
            # Apply domain filtering in Python if needed
            if domain_filter and len(domain_filter) > 0:
                logger.debug(f"Applying domain filtering in Python for domains: {domain_filter}")
                original_count = len(candidates)
                
                # Simple but effective Python-side domain filtering
                filtered_results = []
                for i, fact in enumerate(candidates):
                    fact_domains = [d.lower() for d in fact.get("domains", [])]
                    
                    # Check if any domain in the fact matches any domain in the filter
                    matches = any(d in fact_domains for d in domain_filter)
                    
                    if matches:
                        logger.debug(f"Fact {i+1}: '{fact['content'][:30]}...' - KEPT with domains: {fact.get('domains')}")
                        filtered_results.append(fact)
                    else:
                        logger.debug(f"Fact {i+1}: '{fact['content'][:30]}...' - FILTERED OUT with domains: {fact.get('domains')}")
                
                logger.debug(f"Domain filtering reduced results from {original_count} to {len(filtered_results)}")
                candidates = filtered_results
            
            # Respect the original limit parameter
            results = candidates[:limit]
            
            if results:
                fact_ids = [fact["fact_id"] for fact in results]
                self._update_access_counts(fact_ids)

            # If no results and a domain filter is provided, fall back to a keyword (domain) search.
            if not results and domain_filter:
                logger.debug(f"No results with hybrid search and domain filter {domain_filter}, trying keyword search fallback")
                results = self.keyword_search(
                    query=query,
                    threshold=0.001,  # Very low threshold for domain search
                    limit=limit,
                    domain_filter=domain_filter
                )
            
            logger.debug(f"Final recall results: {len(results)} items")
            return results

        except Exception as e:
            logger.error(f"Error during fact recall: {e}")
            # Log more details for diagnostics
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.debug(f"Query: {aql_query[:200]}...(truncated)")
            logger.debug(f"Bind vars: {bind_vars}")
            return []

    def keyword_search(self, query: str, threshold: float = 0.3, limit: int = 5, 
                       domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform a separate keyword search based solely on domain matching.
        Domain filtering is applied in Python after retrieving candidates.
        """
        if not self.initialized:
            self.initialize()

        # Normalize domain_filter for consistency
        if domain_filter:
            domain_filter = [d.lower() for d in domain_filter]
            logger.debug(f"Keyword search: Using domain filter (normalized): {domain_filter}")

        query_tokens = query.lower().split()
        
        # Fetch more candidates to allow for filtering
        fetch_limit = limit * 3 if domain_filter else limit
        
        bind_vars = {
            "query_tokens": query_tokens,
            "limit": fetch_limit,
            "domain_threshold": 0.1,  # low threshold for domain matching
            "domains_to_preserve": self.config.get("default_preserved_domains", []),
            "preserved_domain_boost": self.config.get("preserved_domain_boost", 0.5)
        }
        
        aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                LET domain_matches = (
                    FOR token IN @query_tokens
                        FOR domain IN fact.domains
                            FILTER CONTAINS(LOWER(domain), token) OR token == LOWER(domain)
                            RETURN 1
                )
                
                LET domain_score = LENGTH(domain_matches) > 0 
                    ? MIN([LENGTH(domain_matches) / LENGTH(@query_tokens), 0.95])
                    : 0
                    
                LET preserved_domains = INTERSECTION(fact.domains, @domains_to_preserve)
                LET domain_boost = LENGTH(preserved_domains) > 0 ? @preserved_domain_boost : 0
                LET final_domain_score = domain_score + domain_boost
                
                // Apply score threshold filtering in AQL
                FILTER final_domain_score >= @domain_threshold
                
                SORT final_domain_score DESC
                LIMIT @limit
                
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance,
                    confidence: fact.confidence,
                    domains: fact.domains,
                    score: final_domain_score,
                    components: {{
                        domain_score: final_domain_score
                    }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count,
                    source: fact.source,
                    glossary_terms: fact.glossary_terms
                }}
        """
        
        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            candidates = list(cursor)
            
            logger.debug(f"Keyword search raw query returned {len(candidates)} candidates")
            
            # Apply domain filtering in Python if needed
            if domain_filter and len(domain_filter) > 0:
                logger.debug(f"Keyword search: Applying domain filtering for domains: {domain_filter}")
                original_count = len(candidates)
                
                # Simple but effective Python-side domain filtering
                filtered_results = []
                for i, fact in enumerate(candidates):
                    fact_domains = [d.lower() for d in fact.get("domains", [])]
                    
                    # Check if any domain in the fact matches any domain in the filter
                    matches = any(d in fact_domains for d in domain_filter)
                    
                    if matches:
                        logger.debug(f"Keyword search - Fact {i+1}: '{fact['content'][:30]}...' - KEPT with domains: {fact.get('domains')}")
                        filtered_results.append(fact)
                    else:
                        logger.debug(f"Keyword search - Fact {i+1}: '{fact['content'][:30]}...' - FILTERED OUT with domains: {fact.get('domains')}")
                
                logger.debug(f"Keyword search domain filtering reduced results from {original_count} to {len(filtered_results)}")
                candidates = filtered_results
            
            # Respect the original limit parameter
            results = candidates[:limit]
            
            if results:
                fact_ids = [fact["fact_id"] for fact in results]
                self._update_access_counts(fact_ids)
                
            logger.debug(f"Final keyword_search results: {len(results)} items")
            return results
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            # Log more details for diagnostics
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _bm25_only_search(self, query: str, threshold: float, limit: int, 
                          domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fallback search using only BM25 when embedding is not available.
        
        Args:
            query: The search query
            threshold: Minimum BM25 score threshold
            limit: Maximum number of results
            domain_filter: Optional list of domains to filter by
            
        Returns:
            List of matching facts with scores
            
        Documentation references:
        - ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
        """
        current_time = datetime.datetime.now().isoformat()
        
        logger.debug(f"Running BM25-only search for '{query}' with domain filter: {domain_filter}")
        
        # Ensure we properly handle the domain filter
        safe_domain_filter = domain_filter if domain_filter is not None else []
        
        view_name = f"{self.config['facts_collection']}_view"
        
        # Check if view exists, if not, fall back to simple search
        views = self.db.views()
        view_exists = any(v["name"] == view_name for v in views)
        
        if view_exists:
            aql_query = f"""
            FOR fact IN {view_name}
                SEARCH ANALYZER(
                    fact.content IN TOKENS(@query, "text_en"),
                    "text_en"
                )
                LET bm25_score = BM25(fact, @k1, @b)
                FILTER bm25_score > @threshold
                
                // Handle domain filtering properly according to ArangoDB docs
                FILTER @domain_filter == null OR 
                       LENGTH(@domain_filter) == 0 OR
                       LENGTH(INTERSECTION(fact.domains || [], @domain_filter)) > 0
                
                SORT bm25_score DESC
                LIMIT @limit
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance || 0.5,
                    confidence: fact.confidence || 0.5,
                    domains: fact.domains || [],
                    score: bm25_score,
                    components: {{
                        semantic_score: 0,
                        bm25_score: bm25_score,
                        importance_boost: fact.importance || 0.5,
                        recency_boost: 0
                    }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count || 0,
                    source: fact.source
                }}
            """
        else:
            # Simple fallback if view doesn't exist
            aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                FILTER CONTAINS(LOWER(fact.content), LOWER(@query))
                
                // Handle domain filtering properly
                FILTER @domain_filter == null OR 
                       LENGTH(@domain_filter) == 0 OR
                       LENGTH(INTERSECTION(fact.domains || [], @domain_filter)) > 0
                
                SORT fact.importance DESC
                LIMIT @limit
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance || 0.5,
                    confidence: fact.confidence || 0.5,
                    domains: fact.domains || [],
                    score: 0.5, // Default score for simple text search
                    components: {{
                        semantic_score: 0,
                        bm25_score: 0.5,
                        importance_boost: fact.importance || 0.5,
                        recency_boost: 0
                    }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count || 0,
                    source: fact.source
                }}
            """

        try:
            bind_vars = {
                "query": query,
                "threshold": threshold,
                "limit": limit,
                "domain_filter": safe_domain_filter,
                "k1": 1.2,  # BM25 term frequency saturation
                "b": 0.75   # BM25 length normalization factor
            }
            
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
            
            if results:
                logger.debug(f"BM25-only search found {len(results)} results")
                fact_ids = [fact["fact_id"] for fact in results]
                self._update_access_counts(fact_ids)
            else:
                logger.debug("BM25-only search found no results")
                
            return results
        except Exception as e:
            logger.error(f"Error during BM25-only search: {e}")
            logger.debug(f"Query: {aql_query[:100]}..., Bind vars: {bind_vars}")
            return []

    def _update_access_counts(self, fact_ids: List[str]) -> None:
        """Update access counts and last_accessed timestamp for facts."""
        if not fact_ids:
            return

        current_time = datetime.datetime.now().isoformat()
        safe_query = f"""
        FOR fact IN {self.config['facts_collection']}
            FILTER fact.fact_id IN @fact_ids
            UPDATE fact WITH {{
                access_count: fact.access_count + 1,
                last_accessed: @current_time
            }} IN {self.config['facts_collection']}
        """
        bind_vars = {
            "fact_ids": fact_ids,
            "current_time": current_time
        }
        
        try:
            self.db.aql.execute(safe_query, bind_vars=bind_vars)
        except Exception as e:
            logger.error(f"Error updating access counts: {e}")

    def decay_memories(self, day_factor: float = 1.0) -> Dict[str, Any]:
        """
        Apply memory decay based on importance, recency, and access count.
        This reduces TTL for unimportant memories and eventually removes them.

        Args:
            day_factor: Multiplier for time progression (default: 1.0)

        Returns:
            Statistics about the decay operation
        """
        if not self.initialized:
            self.initialize()

        aql_query = f"""
        LET day_factor = @day_factor
        
        FOR fact IN {self.config['facts_collection']}
            LET importance_protection = fact.importance * @importance_factor
            LET access_protection = (fact.access_count / 10.0 < 1.0 ? fact.access_count / 10.0 : 1.0) * 0.5
            LET total_protection = importance_protection + access_protection
            
            LET ttl_reduction = day_factor * (1.0 - total_protection)
            LET new_ttl = (fact.ttl_days - ttl_reduction > 0 ? fact.ttl_days - ttl_reduction : 0)
            
            UPDATE fact WITH {{ ttl_days: new_ttl }} IN {self.config['facts_collection']}
            
            COLLECT decay_status = 
                new_ttl <= 0 ? "expired" : 
                new_ttl < fact.ttl_days * 0.5 ? "critical" :
                new_ttl < fact.ttl_days * 0.8 ? "decaying" : 
                "stable"
            RETURN {{
                status: decay_status,
                count: COUNT(1)
            }}
        """

        bind_vars = {
            "importance_factor": self.config["importance_decay_factor"],
            "day_factor": day_factor
        }
        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        stats = {status["status"]: status["count"] for status in results}
        
        # Actually remove expired memories
        removed_count = 0
        if stats.get("expired", 0) > 0:
            removed_count = self._remove_expired_memories()
            stats["removed"] = removed_count
            
        return stats

    def _remove_expired_memories(self) -> int:
        """Remove memories with TTL <= 0."""
        collection_name = self.config["facts_collection"]
        aql_query = f"""
        FOR fact IN {collection_name}
            FILTER fact.ttl_days <= 0
            REMOVE fact IN {collection_name}
            RETURN 1
        """
        try:
            cursor = self.db.aql.execute(aql_query)
            results = list(cursor)
            return len(results)
        except Exception as e:
            logger.error(f"Error removing expired memories: {e}")
            return 0

    def create_association(self, fact_id1: str, fact_id2: str, 
                           association_type: str = "related",
                           weight: float = 0.5) -> Dict[str, Any]:
        """
        Create an association between two facts.

        Args:
            fact_id1: ID of the first fact
            fact_id2: ID of the second fact
            association_type: Type of association (e.g., "related", "causes", etc.)
            weight: Strength of the association (0-1)

        Returns:
            Result of the association creation
        """
        if not self.initialized:
            self.initialize()

        try:
            cursor1 = self.facts.find({"fact_id": fact_id1}, limit=1)
            fact1 = next(cursor1, None)
            cursor2 = self.facts.find({"fact_id": fact_id2}, limit=1)
            fact2 = next(cursor2, None)

            if not fact1 or not fact2:
                missing = []
                if not fact1:
                    missing.append(fact_id1)
                if not fact2:
                    missing.append(fact_id2)
                return {"status": "error", "message": f"Facts not found: {missing}"}

            association = {
                "_from": f"{self.config['facts_collection']}/{fact1['_key']}",
                "_to": f"{self.config['facts_collection']}/{fact2['_key']}",
                "type": association_type,
                "weight": weight,
                "created_at": datetime.datetime.now().isoformat()
            }

            aql_query = f"""
            FOR assoc IN {self.config['associations_collection']}
                FILTER assoc._from == @from AND assoc._to == @to
                RETURN assoc
            """
            bind_vars = {
                "from": association["_from"],
                "to": association["_to"]
            }
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            existing = next(cursor, None)

            if existing:
                self.associations.update(existing["_key"], {
                    "weight": weight,
                    "type": association_type,
                    "updated_at": datetime.datetime.now().isoformat()
                })
                return {"status": "updated", "association": existing["_key"]}
            else:
                result = self.associations.insert(association)
                return {"status": "created", "association": result}
        except Exception as e:
            logger.error(f"Error creating association: {e}")
            return {"status": "error", "message": str(e)}

    def find_related_facts(self, fact_id: str, 
                           min_weight: float = 0.3,
                           types: Optional[List[str]] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find facts related to a given fact through associations.

        Args:
            fact_id: The fact ID to find relations for
            min_weight: Minimum association weight to consider
            types: Optional list of association types to filter by
            limit: Maximum number of results to return

        Returns:
            List of related facts with their association details
        """
        if not self.initialized:
            self.initialize()

        try:
            cursor = self.facts.find({"fact_id": fact_id}, limit=1)
            fact = next(cursor, None)
            if not fact:
                logger.warning(f"Fact {fact_id} not found for relation search")
                return []

            fact_id_with_collection = f"{self.config['facts_collection']}/{fact['_key']}"
            aql_query = f"""
            FOR vertex, edge IN 1..1 ANY @start_vertex {self.config['associations_collection']}
                FILTER edge.weight >= @min_weight
                FILTER @types == null OR edge.type IN @types
                LET related_fact = DOCUMENT(vertex._id)
                RETURN {{
                    fact_id: related_fact.fact_id,
                    content: related_fact.content,
                    importance: related_fact.importance,
                    confidence: related_fact.confidence,
                    domains: related_fact.domains,
                    association: {{
                        type: edge.type,
                        weight: edge.weight,
                        created_at: edge.created_at
                    }}
                }}
                LIMIT @limit
            """
            bind_vars = {
                "start_vertex": fact_id_with_collection,
                "min_weight": min_weight,
                "types": types,
                "limit": limit
            }
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error finding related facts: {e}")
            return []

    def hybrid_search(self, query: str, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Unified method for hybrid search to maintain compatibility with tests.
        This is a wrapper around the recall method (semantic, BM25, and glossary enabled)
        and falls back to a separate keyword (domain) search if no results are found.

        Args:
            query: The search query
            threshold: Minimum relevance score (0-1)
            limit: Maximum number of results

        Returns:
            List of matching facts
        """
        logger.debug(f"Running hybrid search with query: '{query}', threshold: {threshold}, limit: {limit}")
        
        # Perform the hybrid search (excluding domain matching)
        results = self.recall(
            query=query,
            threshold=threshold,
            limit=limit,
            semantic=True,
            bm25=True,
            glossary=True,
            domain_filter=None  # No explicit domain filtering in this stage
        )
        
        # If no results are found, try the separate keyword (domain) search
        if not results:
            logger.debug("No hybrid results found; falling back to keyword (domain) search.")
            results = self.keyword_search(
                query=query,
                threshold=0.001,  # Very low threshold for domain search
                limit=limit,
                domain_filter=self.config.get("default_preserved_domains")
            )
                
        return results

    def cleanup(self) -> None:
        """Clean up resources when done with the memory system."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.facts = None
            self.associations = None
            self.initialized = False

    def _get_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Generate an embedding for the given text using embedding utilities.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A dictionary containing the embedding vector and metadata, or an empty embedding if generation fails
        
        Documentation references:
        - Agent Tools Embedding Utils: Provides create_embedding_sync function
        """
        if not text or text.strip() == "":
            logger.warning("Cannot generate embedding for empty text")
            return {"embedding": [], "model": "fallback", "error": "Empty text"}
        
        try:
            logger.debug(f"Generating embedding for text: '{text[:50]}...' (length: {len(text)})")
            embedding_data = create_embedding_sync(text)
            
            # Verify that embedding_data has the expected structure
            if embedding_data and 'embedding' in embedding_data and embedding_data['embedding']:
                # Log vector stats for debugging without printing full vector
                stats = get_vector_stats(embedding_data['embedding'])
                logger.debug(f"Successfully generated embedding with stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, length={len(embedding_data['embedding'])}")
                return embedding_data
            else:
                logger.warning(f"Embedding data issue: {embedding_data if embedding_data else 'None returned'}")
                # Return an empty embedding rather than None to prevent errors in search functions
                return {"embedding": [], "model": "fallback", "error": "Missing or empty embedding in response"}
        except Exception as e:
            logger.error(f"Error generating embedding for '{text[:30]}...': {str(e)}")
            import traceback
            logger.debug(f"Embedding generation traceback: {traceback.format_exc()}")
            # Return empty embedding instead of None to prevent errors in search functions
            return {"embedding": [], "model": "fallback", "error": str(e)}

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a fact by its ID.
        
        Args:
            fact_id: The ID of the fact to retrieve
            
        Returns:
            The fact document if found, None otherwise
            
        Documentation references:
        - ArangoDB Python Driver: https://python-arango.readthedocs.io/
        - ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Use AQL to get the fact by ID to ensure consistent access pattern
            aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                FILTER fact.fact_id == @fact_id
                RETURN fact
            """
            
            bind_vars = {
                "fact_id": fact_id
            }
            
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            fact = next(cursor, None)
            
            if fact:
                # Update access count and timestamp
                self._update_access_counts([fact_id])
            
            return fact
            
        except Exception as e:
            logger.error(f"Error retrieving fact {fact_id}: {e}")
            return None

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pprint import pprint

    def debug_domain_search():
        """
        Debug utility for testing domain preservation search.
        
        This function:
        1. Initializes a memory system with test data
        2. Adds facts with various domains including physics
        3. Tests recall with different parameters
        4. Shows detailed scoring information
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Debug agent memory system')
        parser.add_argument('--query', type=str, default="physics concepts",
                          help='Search query to test')
        parser.add_argument('--domains', type=str, nargs='+', default=["physics"],
                          help='Domains to preserve/filter')
        parser.add_argument('--debug_type', type=str, default="domain",
                          choices=['domain', 'hybrid', 'recall', 'all'],
                          help='Type of debugging to perform')
        parser.add_argument('--threshold', type=float, default=0.01,
                          help='Score threshold')
        args = parser.parse_args()
        
        # Initialize memory system
        memory = AgentMemorySystem()
        memory.initialize()
        
        # Clear existing data for clean testing
        print("Clearing existing data...")
        memory.db.aql.execute(f"FOR doc IN {memory.config['facts_collection']} REMOVE doc IN {memory.config['facts_collection']}")
        
        # Add test facts with various domains
        print("Adding test facts...")
        facts = [
            {
                "content": "The concept of gravity is fundamental to physics",
                "importance": 0.9,
                "domains": ["physics", "science"],
                "ttl_days": 1000  # Permanent fact
            },
            {
                "content": "E=mc² is Einstein's famous equation",
                "importance": 0.8,
                "domains": ["physics", "relativity"],
                "ttl_days": 900
            },
            {
                "content": "Water boils at 100 degrees Celsius at sea level",
                "importance": 0.5,
                "domains": ["chemistry", "science"],
                "ttl_days": 400
            },
            {
                "content": "Mitochondria is the powerhouse of the cell",
                "importance": 0.6,
                "domains": ["biology", "science"],
                "ttl_days": 500
            },
            {
                "content": "Pythagoras theorem relates to triangles",
                "importance": 0.7,
                "domains": ["mathematics", "geometry"],
                "ttl_days": 700
            },
            {
                "content": "The capital of France is Paris",
                "importance": 0.4,
                "domains": ["geography", "general_knowledge"],
                "ttl_days": 300
            },
        ]
        
        # Remember all facts and show embeddings stats
        print("\nAdding facts with embeddings:")
        for fact in facts:
            # Capture the result without printing it directly
            try:
                result = memory.remember(**fact)
                if 'new' in result and 'embedding' in result['new']:
                    print(f"\nFact: '{fact['content'][:30]}...'")
                    # Create a debug-friendly version of the result
                    debug_result = result['new'].copy()
                    if 'embedding' in debug_result:
                        debug_result['embedding'] = truncate_vector_for_display(debug_result['embedding'], max_items=3)
                    print(f"  Debug fact: {debug_result}")
                    stats = get_vector_stats(result['new']['embedding'])
                    print(f"  Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, norm={stats['norm']:.3f}")
            except Exception as e:
                print(f"Error remembering fact: {e}")
        
        print(f"\n--- Debug: {args.debug_type} ---")
        
        # Debug based on selected type
        if args.debug_type in ['domain', 'all']:
            print("\n=== Testing domain matching (Keyword Search) ===")
            query_tokens = args.query.lower().split()
            print(f"Query tokens: {query_tokens}")
            print("\nManual domain matching:")
            for i, fact in enumerate(facts):
                domain_matches = []
                for token in query_tokens:
                    for domain in fact["domains"]:
                        if token in domain.lower() or token == domain.lower():
                            domain_matches.append((token, domain))
                domain_score = len(domain_matches) / len(query_tokens) if query_tokens else 0
                preserved_domains = set(fact["domains"]) & set(args.domains or [])
                domain_boost = 0.5 if preserved_domains else 0
                final_score = domain_score + domain_boost
                print(f"Fact {i+1}: '{fact['content'][:30]}...'")
                print(f"  Domains: {fact['domains']}")
                print(f"  Domain matches: {domain_matches}")
                print(f"  Domain score: {domain_score:.3f}")
                print(f"  Preserved domains: {preserved_domains}")
                print(f"  Domain boost: {domain_boost}")
                print(f"  Final domain score: {final_score:.3f}")
                print()
        
        if args.debug_type in ['recall', 'all']:
            print("\n=== Testing recall method (Hybrid without domain matching) ===")
            # Generate query embedding and show stats
            query_embedding = memory._get_embedding(args.query)
            if query_embedding:
                # Don't print the full embedding object
                stats = get_vector_stats(query_embedding["embedding"])
                print(f"\nQuery embedding:")
                print(f"  Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, norm={stats['norm']:.3f}")
                print(f"  Vector: {truncate_vector_for_display(query_embedding['embedding'], max_items=3)}")
            
            # Capture recall results without printing them directly
            try:
                recall_results = memory.recall(
                    query=args.query,
                    threshold=args.threshold,
                    domain_filter=args.domains if args.domains else None
                )
                
                print(f"\nRecall results (threshold={args.threshold}):")
                for i, result in enumerate(recall_results):
                    # Create a debug-friendly version of the result
                    debug_result = result.copy()
                    print(f"\nResult {i+1}: {debug_result['content']}")
                    print(f"  Score: {debug_result['score']:.3f}")
                    print(f"  Domains: {debug_result['domains']}")
                    print(f"  Component scores:")
                    for component, score in debug_result['components'].items():
                        print(f"    {component}: {score:.3f}")
            except Exception as e:
                print(f"Error during recall: {e}")
        
        if args.debug_type in ['hybrid', 'all']:
            print("\n=== Testing hybrid search (with fallback to keyword search) ===")
            # Capture hybrid results without printing them directly
            try:
                hybrid_results = memory.hybrid_search(
                    query=args.query,
                    threshold=args.threshold
                )
                
                print(f"Hybrid search results (threshold={args.threshold}):")
                for i, result in enumerate(hybrid_results):
                    # Create a debug-friendly version of the result
                    debug_result = result.copy()
                    print(f"\nResult {i+1}: {debug_result['content']}")
                    print(f"  Score: {debug_result['score']:.3f}")
                    print(f"  Domains: {debug_result['domains']}")
                    print(f"  Component scores:")
                    for component, score in debug_result['components'].items():
                        print(f"    {component}: {score:.3f}")
            except Exception as e:
                print(f"Error during hybrid search: {e}")
        
        if args.debug_type in ['all']:
            print("\n=== AQL Query Preview ===")
            print("This utility doesn't show the actual AQL query to avoid complexity")
            print("Check the implementation in the recall() and keyword_search() methods for the query structure")
                
        print("\nDone!")

    # Run the debug function
    debug_domain_search()
