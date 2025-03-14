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
from typing import Dict, List, Any, Optional

from arango import ArangoClient
from loguru import logger
from pydantic import BaseModel, Field

from agent_tools.cursor_rules.embedding.embedding_utils import create_embedding_sync
from agent_tools.cursor_rules.utils.vector_utils import (
    format_embedding_for_debug,
    get_vector_stats,
    truncate_vector_for_display,
)

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
    "preserved_domain_boost": 0.5,             # Increased boost for preserved domains
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
    previous_content: Optional[str] = Field(None, description="Previous version if updated")
    updated_at: Optional[str] = Field(None, description="When this fact was last updated")
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
            sys_db = self.client.db(
                "_system",
                username=self.config["username"],
                password=self.config["password"]
            )
            if not sys_db.has_database(self.config["db_name"]):
                sys_db.create_database(self.config["db_name"])
            self.db = self.client.db(
                self.config["db_name"],
                username=self.config["username"],
                password=self.config["password"]
            )
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
            self.facts = self.db.create_collection(facts_name)
            # Using dictionary syntax (as recommended for python-arango v8.1.6)
            self.facts.add_index({"type": "persistent", "fields": ["fact_id"], "unique": True})
            self.facts.add_index({"type": "persistent", "fields": ["importance"]})
            self.facts.add_index({"type": "persistent", "fields": ["confidence"]})
            self.facts.add_index({"type": "persistent", "fields": ["last_accessed"]})
            self.facts.add_index({"type": "persistent", "fields": ["access_count"]})
            self.facts.add_index({"type": "persistent", "fields": ["ttl_days"]})
            self.facts.add_index({"type": "persistent", "fields": ["domains"]})
            # Create an ArangoSearch view for text search
            view_name = f"{facts_name}_view"
            self._create_search_view(facts_name, view_name)
        else:
            self.facts = self.db.collection(facts_name)

    def _create_search_view(self, collection_name: str, view_name: str) -> None:
        """
        Create an ArangoSearch view for the collection.
        If creation fails, we log and continue with reduced functionality.
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
                    logger.debug(f"Search view creation error: {str(e)}")
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

    def remember(
        self,
        content: str,
        importance: float = 0.5,
        confidence: float = 0.7,
        domains: Optional[List[str]] = None,
        source: Optional[str] = None,
        fact_id: Optional[str] = None,
        ttl_days: Optional[float] = None,
        glossary_terms: Optional[List[str]] = None,
        correction_history: Optional[List[Dict]] = None,
        alternatives: Optional[List[Dict]] = None,
        resolution_notes: Optional[str] = None,
        resolved_from: Optional[str] = None,
        merged_from: Optional[List[str]] = None,
        merge_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a new fact (or update an existing one) using an upsert operation.
        """
        if not self.initialized:
            self.initialize()

        current_time = datetime.datetime.now().isoformat()

        if not fact_id:
            hash_obj = hashlib.md5(content.encode())
            fact_id = f"fact_{hash_obj.hexdigest()[:10]}"

        if ttl_days is None:
            ttl_days = self.config["default_ttl_days"] * (1 + importance)

        embedding_data = self._get_embedding(content)
        embedding = embedding_data.get("embedding") if embedding_data else None

        if glossary_terms is None:
            glossary_terms = []
            for word in content.split():
                if word and word[0].isupper() and len(word) > 2:
                    glossary_terms.append(word.strip('.,;:()[]{}""\''))

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
            "correction_history": correction_history or [],
            "alternatives": alternatives or [],
            "resolution_notes": resolution_notes,
            "resolved_from": resolved_from,
            "merged_from": merged_from or [],
            "merge_notes": merge_notes,
        }

        logger.info(f"Attempting to store fact: {fact_id}")
        debug_fact = fact.copy()
        if embedding is not None:
            debug_fact["embedding"] = format_embedding_for_debug({"embedding": embedding})
        logger.debug(f"Fact details: {debug_fact}")

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
            "merge_notes": merge_notes,
        }

        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            result = next(cursor, None)
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
        boost_recency: bool = True,
        limit: int = 5,
        domain_filter: Optional[List[str]] = None,
        semantic: bool = True,
        bm25: bool = True,
        glossary: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recall facts using BM25, semantic, and glossary scoring. This method
        retrieves a broader candidate set (without applying domain filtering in AQL)
        and then filters (and optionally re-sorts) the results in Python.
        """
        if not self.initialized:
            self.initialize()
        if not query or query.strip() == "":
            return []

        # Ensure at least one search component is enabled
        if not any([semantic, bm25, glossary]):
            logger.warning("No search components enabled, defaulting to BM25 search")
            bm25 = True

        current_time = datetime.datetime.now().isoformat()
        recency_factor = self.config["recency_boost_factor"] if boost_recency else 0

        # Search parameters
        k1 = 1.2
        b = 0.75
        bm25_threshold = 0.1
        embedding_threshold = 0.5
        glossary_threshold = 0.3

        query_embedding = None
        valid_semantic = False
        if semantic:
            try:
                query_embedding = self._get_embedding(query)
                valid_semantic = bool(query_embedding and query_embedding.get("embedding") and len(query_embedding["embedding"]) > 0)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                query_embedding = None

        total_components = sum([1 if semantic and valid_semantic else 0, 1 if bm25 else 0, 1 if glossary else 0])
        base_weight = 0.8 / total_components if total_components > 0 else 0
        semantic_weight = base_weight if (semantic and valid_semantic) else 0
        bm25_weight = base_weight if bm25 else 0
        glossary_weight = base_weight if glossary else 0
        importance_weight = 0.1
        recency_weight = 0.1

        # Fetch more candidates (double the limit) to allow for domain filtering later.
        fetch_limit = limit * 2 if domain_filter and len(domain_filter) > 0 else limit

        # Build a simpler AQL query that computes BM25, semantic, and glossary scores
        aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                LET similarity_score = {"(" +
                    "COSINE_SIMILARITY(fact.embedding, @query_embedding) " +
                    ">= @embedding_threshold ? COSINE_SIMILARITY(fact.embedding, @query_embedding) : 0)" if valid_semantic else "0"}
                
                LET bm25_score = {"(" +
                    "FOR doc IN " + f"{self.config['facts_collection']}_view " +
                    "SEARCH ANALYZER(BOOST(doc.content IN TOKENS(@query, 'text_en'), 1.0), 'text_en') " +
                    "FILTER doc._key == fact._key " +
                    "RETURN BM25(doc, @k1, @b)" +
                    ")[0] || 0" if bm25 else "0"}
                
                LET glossary_matches = {"(" +
                    "FOR term IN fact.glossary_terms || [] " +
                    "FILTER LOWER(term) IN TOKENS(LOWER(@query), 'text_en') " +
                    "RETURN 1" +
                    ")" if glossary else "[]"}
                
                LET glossary_score = {"(" +
                    "LENGTH(glossary_matches) > 0 ? MIN([LENGTH(glossary_matches) / LENGTH(TOKENS(@query, 'text_en'))]) : 0" if glossary else "0"}
                
                LET days_since_access = DATE_DIFF(fact.last_accessed, @current_time, 'd')
                LET days_factor = 1.0 / (1.0 + ABS(days_since_access))
                LET recency_boost = days_factor * @recency_factor
                
                LET hybrid_score = (similarity_score * @semantic_weight) + (bm25_score * @bm25_weight) +
                                   (glossary_score * @glossary_weight) + (fact.importance * @importance_weight) +
                                   (recency_boost * @recency_weight)
                
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

        bind_vars = {
            "query": query,
            "current_time": current_time,
            "recency_factor": recency_factor,
            "limit": fetch_limit,
            "threshold": threshold,
            "k1": k1,
            "b": b,
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "glossary_weight": glossary_weight,
            "importance_weight": importance_weight,
            "recency_weight": recency_weight,
            "embedding_threshold": embedding_threshold,
        }
        if valid_semantic:
            bind_vars["query_embedding"] = query_embedding["embedding"]

        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            candidates = list(cursor)
        except Exception as e:
            logger.error(f"Error during fact recall: {e}")
            logger.debug(f"Query: {aql_query[:100]}..., Bind vars: {bind_vars}")
            return []

        # Now, if a domain filter is provided, apply it in Python.
        if domain_filter:
            logger.debug(f"Applying domain filtering in Python for domains: {domain_filter}")
            filtered = [
                fact for fact in candidates
                if any(domain in fact.get("domains", []) for domain in domain_filter)
            ]
            logger.debug(f"Domain filtering reduced candidate count from {len(candidates)} to {len(filtered)}")
            candidates = filtered

        # Limit the results to the requested number and update access counts.
        results = candidates[:limit]
        if results:
            fact_ids = [fact["fact_id"] for fact in results]
            self._update_access_counts(fact_ids)

        # Fallback: if still no results and a domain filter was given, try keyword_search.
        if not results and domain_filter:
            logger.debug(f"No candidates after hybrid search and domain filtering; falling back to keyword search for domains {domain_filter}")
            results = self.keyword_search(query=query, threshold=0.001, limit=limit, domain_filter=domain_filter)

        return results

    def keyword_search(self, query: str, threshold: float = 0.3, limit: int = 5,
                       domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform a separate keyword search based solely on domain matching.
        """
        if not self.initialized:
            self.initialize()

        query_tokens = query.lower().split()
        # Fetch more candidates to allow for filtering.
        fetch_limit = limit * 2 if domain_filter and len(domain_filter) > 0 else limit

        bind_vars = {
            "query_tokens": query_tokens,
            "limit": fetch_limit,
            "domain_threshold": 0.1,
            "domains_to_preserve": domain_filter if domain_filter is not None else self.config.get("default_preserved_domains", []),
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
                LET domain_score = (LENGTH(domain_matches) > 0 ? MIN([LENGTH(domain_matches) / LENGTH(@query_tokens), 0.95]) : 0)
                LET preserved_domains = INTERSECTION(fact.domains, @domains_to_preserve)
                LET domain_boost = (LENGTH(preserved_domains) > 0 ? @preserved_domain_boost : 0)
                LET final_domain_score = domain_score + domain_boost
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
                    components: {{ domain_score: final_domain_score }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count,
                    source: fact.source,
                    glossary_terms: fact.glossary_terms
                }}
        """

        try:
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
            # Additionally filter in Python if needed.
            if domain_filter:
                results = [fact for fact in results if any(d in fact.get("domains", []) for d in domain_filter)]
            if results:
                fact_ids = [fact["fact_id"] for fact in results]
                self._update_access_counts(fact_ids)
            return results
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            return []

    def _bm25_only_search(self, query: str, threshold: float, limit: int,
                          domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fallback BM25-only search; similar pattern to keyword_search."""
        current_time = datetime.datetime.now().isoformat()
        logger.debug(f"Running BM25-only search for '{query}' with domain filter: {domain_filter}")
        safe_domain_filter = domain_filter if domain_filter is not None else []
        view_name = f"{self.config['facts_collection']}_view"
        views = self.db.views()
        view_exists = any(v["name"] == view_name for v in views)
        if view_exists:
            aql_query = f"""
            FOR fact IN {view_name}
                SEARCH ANALYZER(fact.content IN TOKENS(@query, "text_en"), "text_en")
                LET bm25_score = BM25(fact, @k1, @b)
                FILTER bm25_score > @threshold
                FILTER @domain_filter == null OR LENGTH(@domain_filter) == 0 OR LENGTH(INTERSECTION(fact.domains || [], @domain_filter)) > 0
                SORT bm25_score DESC
                LIMIT @limit
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance || 0.5,
                    confidence: fact.confidence || 0.5,
                    domains: fact.domains || [],
                    score: bm25_score,
                    components: {{ bm25_score: bm25_score }},
                    last_accessed: fact.last_accessed,
                    access_count: fact.access_count || 0,
                    source: fact.source
                }}
            """
        else:
            aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                FILTER CONTAINS(LOWER(fact.content), LOWER(@query))
                FILTER @domain_filter == null OR LENGTH(@domain_filter) == 0 OR LENGTH(INTERSECTION(fact.domains || [], @domain_filter)) > 0
                SORT fact.importance DESC
                LIMIT @limit
                RETURN {{
                    fact_id: fact.fact_id,
                    content: fact.content,
                    importance: fact.importance || 0.5,
                    confidence: fact.confidence || 0.5,
                    domains: fact.domains || [],
                    score: 0.5,
                    components: {{ bm25_score: 0.5 }},
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
                "k1": 1.2,
                "b": 0.75
            }
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
            if results:
                fact_ids = [fact["fact_id"] for fact in results]
                self._update_access_counts(fact_ids)
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
            UPDATE fact WITH {{ access_count: fact.access_count + 1, last_accessed: @current_time }} IN {self.config['facts_collection']}
        """
        bind_vars = {"fact_ids": fact_ids, "current_time": current_time}
        try:
            self.db.aql.execute(safe_query, bind_vars=bind_vars)
        except Exception as e:
            logger.error(f"Error updating access counts: {e}")

    def decay_memories(self, day_factor: float = 1.0) -> Dict[str, Any]:
        """
        Apply memory decay based on importance, recency, and access count.
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
                new_ttl < fact.ttl_days * 0.8 ? "decaying" : "stable"
            RETURN {{ status: decay_status, count: COUNT(1) }}
        """
        bind_vars = {"importance_factor": self.config["importance_decay_factor"], "day_factor": day_factor}
        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(cursor)
        stats = {status["status"]: status["count"] for status in results}
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

    def create_association(self, fact_id1: str, fact_id2: str, association_type: str = "related", weight: float = 0.5) -> Dict[str, Any]:
        """
        Create an association between two facts.
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
                if not fact1: missing.append(fact_id1)
                if not fact2: missing.append(fact_id2)
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
            bind_vars = {"from": association["_from"], "to": association["_to"]}
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

    def find_related_facts(self, fact_id: str, min_weight: float = 0.3, types: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find facts related to a given fact through associations.
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
            bind_vars = {"start_vertex": fact_id_with_collection, "min_weight": min_weight, "types": types, "limit": limit}
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error finding related facts: {e}")
            return []

    def hybrid_search(self, query: str, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Unified hybrid search that wraps recall. It first retrieves a candidate
        set using BM25/semantic scoring and then applies additional domain filtering
        and merging in Python.
        """
        logger.debug(f"Running hybrid search with query: '{query}', threshold: {threshold}, limit: {limit}")
        # Use recall() without any domain filtering in the AQL query
        candidates = self.recall(query=query, threshold=threshold, limit=limit, semantic=True, bm25=True, glossary=True, domain_filter=None)
        # Now, if a domain filter is desired, apply it in Python
        # (For this function, we assume if a domain filter is desired, it was passed in configuration)
        domain_filter = self.config.get("default_preserved_domains")
        if domain_filter:
            logger.debug(f"Applying Python-based domain filtering for domains: {domain_filter}")
            candidates = [fact for fact in candidates if any(d in fact.get("domains", []) for d in domain_filter)]
            candidates = candidates[:limit]
        # Fallback to keyword_search if no candidates remain
        if not candidates and domain_filter:
            logger.debug("No hybrid candidates after filtering; falling back to keyword search.")
            candidates = self.keyword_search(query=query, threshold=0.001, limit=limit, domain_filter=domain_filter)
        return candidates

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
        """
        if not text or text.strip() == "":
            logger.warning("Cannot generate embedding for empty text")
            return {"embedding": [], "model": "fallback", "error": "Empty text"}
        try:
            logger.debug(f"Generating embedding for text: '{text[:50]}...' (length: {len(text)})")
            embedding_data = create_embedding_sync(text)
            if embedding_data and 'embedding' in embedding_data and embedding_data['embedding']:
                stats = get_vector_stats(embedding_data['embedding'])
                logger.debug(f"Successfully generated embedding with stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, length={len(embedding_data['embedding'])}")
                return embedding_data
            else:
                logger.warning(f"Embedding data issue: {embedding_data if embedding_data else 'None returned'}")
                return {"embedding": [], "model": "fallback", "error": "Missing or empty embedding in response"}
        except Exception as e:
            logger.error(f"Error generating embedding for '{text[:30]}...': {str(e)}")
            import traceback
            logger.debug(f"Embedding generation traceback: {traceback.format_exc()}")
            return {"embedding": [], "model": "fallback", "error": str(e)}

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a fact by its ID.
        """
        if not self.initialized:
            self.initialize()
        try:
            aql_query = f"""
            FOR fact IN {self.config['facts_collection']}
                FILTER fact.fact_id == @fact_id
                RETURN fact
            """
            bind_vars = {"fact_id": fact_id}
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            fact = next(cursor, None)
            if fact:
                self._update_access_counts([fact_id])
            return fact
        except Exception as e:
            logger.error(f"Error retrieving fact {fact_id}: {e}")
            return None


if __name__ == "__main__":
    import argparse
    import sys
    from pprint import pprint

    def debug_domain_search():
        """
        Debug utility for testing domain preservation search.
        """
        parser = argparse.ArgumentParser(description="Debug agent memory system")
        parser.add_argument("--query", type=str, default="physics concepts", help="Search query to test")
        parser.add_argument("--domains", type=str, nargs="+", default=["physics"], help="Domains to preserve/filter")
        parser.add_argument("--debug_type", type=str, default="domain", choices=["domain", "hybrid", "recall", "all"], help="Type of debugging to perform")
        parser.add_argument("--threshold", type=float, default=0.01, help="Score threshold")
        args = parser.parse_args()

        memory = AgentMemorySystem()
        memory.initialize()

        print("Clearing existing data...")
        memory.db.aql.execute(f"FOR doc IN {memory.config['facts_collection']} REMOVE doc IN {memory.config['facts_collection']}")

        print("Adding test facts...")
        facts = [
            {
                "content": "The concept of gravity is fundamental to physics",
                "importance": 0.9,
                "domains": ["physics", "science"],
                "ttl_days": 1000,
            },
            {
                "content": "E=mc² is Einstein's famous equation",
                "importance": 0.8,
                "domains": ["physics", "relativity"],
                "ttl_days": 900,
            },
            {
                "content": "Water boils at 100 degrees Celsius at sea level",
                "importance": 0.5,
                "domains": ["chemistry", "science"],
                "ttl_days": 400,
            },
            {
                "content": "Mitochondria is the powerhouse of the cell",
                "importance": 0.6,
                "domains": ["biology", "science"],
                "ttl_days": 500,
            },
            {
                "content": "Pythagoras theorem relates to triangles",
                "importance": 0.7,
                "domains": ["mathematics", "geometry"],
                "ttl_days": 700,
            },
            {
                "content": "Neural networks are AI models",
                "importance": 0.8,
                "domains": ["ai", "technology"],
                "ttl_days": 54,
            },
            {
                "content": "Python is a programming language",
                "importance": 0.8,
                "domains": ["programming", "technology"],
                "ttl_days": 54,
            },
        ]

        fact_ids = []
        for fact in facts:
            try:
                result = memory.remember(**fact)
                if result and "new" in result:
                    fact_ids.append(result["new"]["fact_id"])
                    print(f"Stored fact: {result['new']['fact_id']} - {fact['content'][:30]}...")
            except Exception as e:
                print(f"Error remembering fact: {e}")

        print(f"\nStored {len(fact_ids)} facts.")
        # Test recall with domain filtering (should only return facts with the 'ai' domain)
        results = memory.recall(
            query="technology",
            domain_filter=["ai"],
            threshold=0.05,
            semantic=True,
            bm25=True,
            glossary=True
        )
        if not results:
            print("No results found with hybrid search, trying BM25 only fallback...")
            results = memory.recall(
                query="technology",
                domain_filter=["ai"],
                threshold=0.01,
                semantic=False,
                bm25=True,
                glossary=True
            )

        print(f"\nSearch results count: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['content']}")
            print(f"  Domains: {result['domains']}")
            print(f"  Score: {result['score']}")
            if "components" in result:
                print(f"  Score components: {result['components']}")

    debug_domain_search()
