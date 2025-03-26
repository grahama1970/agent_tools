#!/usr/bin/env python3
"""
AI Memory System

A comprehensive memory system for AI assistant operations that integrates embeddings
for semantic search, context maintenance, and learning from past interactions.

This system extends both the base TestStateManager and EnhancedMemory with proper
embedding capabilities for more accurate semantic search, better relationship discovery,
and improved error pattern recognition.

Key Features:
1. Vector-based semantic search with proper embeddings
2. Automatic similarity detection between documents
3. Contextual memory retrieval based on the current task
4. Bidirectional relationship discovery between related knowledge
5. Error pattern recognition with semantic matching
6. Batch processing for efficient memory operations
7. Knowledge staleness tracking and prioritization
8. Multi-modal memory support for different types of information
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple, cast
from collections import defaultdict
import numpy as np  # Make sure numpy is imported for vector operations
from sklearn.metrics.pairwise import cosine_similarity  # Import for similarity calculation

# Import base functionality
from .test_state_manager import (
    get_state_manager, 
    what_am_i_doing, 
    remember_context, 
    add_docs, 
    get_docs
)

# Import enhanced memory system for backward compatibility
from .enhanced_memory import (
    detect_context_staleness,
    load_project_docs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("memory_system.log")
    ]
)
logger = logging.getLogger("ai_memory")

# Import embedding utilities (with fallback to TF-IDF)
try:
    # Try to import embedding utilities for semantic search
    from agent_tools.fetch_docs.embedding.embedding_utils import (
        create_embedding_sync,
        create_embedding_with_sentence_transformer
    )
    proper_embeddings_available = True
    logger.info("Using proper embedding utilities for semantic search")
except ImportError:
    try:
        # Fallback to scikit-learn for TF-IDF vectorization
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        proper_embeddings_available = False
        tfidf_available = True
        logger.info("Using TF-IDF for semantic search (embeddings not available)")
    except ImportError:
        tfidf_available = False
        logger.warning("Semantic search unavailable. Install scikit-learn or setup embeddings.")

# -------------------------------------------------
# Vector Store for Document Embeddings
# -------------------------------------------------

class VectorStore:
    """Vector database for storing and searching embeddings."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.doc_ids = []
        self.embeddings = []
        self.doc_metadata = {}
        self.embedding_model = None
        self.embedding_method = None
        self.embedding_dim = None
        self.initialized = False
        
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a document to the vector store with its embedding.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to create embedding from
            metadata: Optional document metadata
        
        Returns:
            True if successful, False otherwise
        """
        embedding_result = None
        
        # Get embedding based on available methods
        try:
            if proper_embeddings_available:
                # Use proper embedding model
                embedding_result = create_embedding_sync(content)
                self.embedding_method = "transformer"
            elif tfidf_available:
                # Use simple TF-IDF fallback
                if not self.embedding_model:
                    # Initialize TF-IDF vectorizer
                    self.embedding_model = TfidfVectorizer(stop_words='english')
                    # Create an initial model with this document
                    sparse_vector = self.embedding_model.fit_transform([content])
                    dense_vector = sparse_vector.toarray()[0].tolist()
                    embedding_result = {
                        'embedding': dense_vector,
                        'metadata': {
                            'embedding_model': 'tfidf-vectorizer',
                            'embedding_timestamp': datetime.datetime.now().isoformat(),
                            'embedding_method': 'tfidf',
                            'embedding_dim': len(dense_vector)
                        }
                    }
                else:
                    # Use existing TF-IDF model
                    sparse_vector = self.embedding_model.transform([content])
                    dense_vector = sparse_vector.toarray()[0].tolist()
                    embedding_result = {
                        'embedding': dense_vector,
                        'metadata': {
                            'embedding_model': 'tfidf-vectorizer',
                            'embedding_timestamp': datetime.datetime.now().isoformat(),
                            'embedding_method': 'tfidf',
                            'embedding_dim': len(dense_vector)
                        }
                    }
                
            else:
                logger.error("No embedding method available")
                return False
                
            # Store document and embedding
            self.doc_ids.append(doc_id)
            self.embeddings.append(embedding_result['embedding'])
            self.doc_metadata[doc_id] = metadata or {}
            
            # Update store metadata
            self.embedding_method = embedding_result['metadata']['embedding_method']
            self.embedding_dim = embedding_result['metadata']['embedding_dim']
            self.embedding_model = embedding_result['metadata']['embedding_model']
            self.initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")
            return False
            
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add multiple documents to the vector store efficiently.
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and optional 'metadata'
            
        Returns:
            Number of documents successfully added
        """
        success_count = 0
        
        for doc in documents:
            doc_id = doc.get('id') or doc.get('topic')
            content = doc.get('content', '')
            
            if not doc_id or not content:
                continue
                
            # Add metadata if available
            metadata = {k: v for k, v in doc.items() if k not in ['id', 'content', 'topic']}
            if 'topic' in doc and 'id' not in metadata:
                metadata['topic'] = doc['topic']
                
            if self.add_document(doc_id, content, metadata):
                success_count += 1
                
        return success_count
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            
        Returns:
            List of document matches with similarity scores
        """
        if not self.initialized:
            return []
            
        try:
            # Get query embedding
            query_embedding = None
            
            if proper_embeddings_available:
                # Use proper embedding model
                embedding_result = create_embedding_sync(query)
                query_embedding = embedding_result['embedding']
            elif tfidf_available and self.embedding_model:
                # Use TF-IDF vectorizer
                sparse_vector = self.embedding_model.transform([query])
                query_embedding = sparse_vector.toarray()[0].tolist()
            else:
                logger.error("No embedding method available for search")
                return []
                
            # Calculate similarities
            if proper_embeddings_available or tfidf_available:
                # Convert to numpy arrays for efficient computation
                query_array = np.array(query_embedding).reshape(1, -1)
                doc_array = np.array(self.embeddings)
                
                # Compute cosine similarities
                similarities = cosine_similarity(query_array, doc_array)[0]
                
                # Get top results
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                # Prepare results with minimum similarity threshold
                results = []
                for i in top_indices:
                    if similarities[i] > 0.1:  # Minimum similarity threshold
                        doc_id = self.doc_ids[i]
                        metadata = self.doc_metadata.get(doc_id, {})
                        results.append({
                            'id': doc_id,
                            'similarity': float(similarities[i]),
                            'metadata': metadata
                        })
                        
                return results
                
            return []
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def find_related_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents related to a specific document by ID.
        
        Args:
            doc_id: Document ID to find relations for
            top_k: Maximum results to return
            
        Returns:
            List of related documents with similarity scores
        """
        if not self.initialized or doc_id not in self.doc_ids:
            return []
            
        try:
            # Get document index
            doc_index = self.doc_ids.index(doc_id)
            
            # Get document embedding
            doc_embedding = self.embeddings[doc_index]
            
            # Calculate similarities to all other documents
            doc_array = np.array(self.embeddings)
            query_array = np.array(doc_embedding).reshape(1, -1)
            
            similarities = cosine_similarity(query_array, doc_array)[0]
            
            # Sort and filter (skip the document itself which would be highest similarity)
            # Create (index, similarity) pairs and sort by similarity
            indexed_similarities = [(i, similarities[i]) for i in range(len(similarities)) if i != doc_index]
            indexed_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k results
            results = []
            for i, similarity in indexed_similarities[:top_k]:
                if similarity > 0.2:  # Minimum similarity threshold for relations
                    related_id = self.doc_ids[i]
                    metadata = self.doc_metadata.get(related_id, {})
                    results.append({
                        'id': related_id,
                        'similarity': float(similarity),
                        'metadata': metadata
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error finding related documents: {e}")
            return []

# Global vector store instance
vector_store = VectorStore()


# -------------------------------------------------
# AI Memory System Core Functions  
# -------------------------------------------------

def initialize_memory_system(db_path: Optional[str] = None, load_docs: bool = True):
    """
    Initialize the AI memory system and load existing data.
    
    Args:
        db_path: Optional path to the state database
        load_docs: Whether to load project documentation
        
    Returns:
        Initialization result message
    """
    manager = get_state_manager(db_path)
    
    # Initialize vector store if embedding capabilities available
    if load_docs:
        result = set_up_automatic_loading(db_path)
        return result
    else:
        return "Memory system initialized without loading docs"
        

def set_up_automatic_loading(db_path: Optional[str] = None):
    """
    Set up automatic documentation loading and vector indexing.
    
    Args:
        db_path: Optional path to the state database
        
    Returns:
        Loading result message
    """
    manager = get_state_manager(db_path)
    
    # Check if we already loaded docs recently (within last 24 hours)
    last_load = manager.get_metadata('last_auto_load_time')
    if last_load:
        try:
            last_time = datetime.datetime.fromisoformat(last_load)
            now = datetime.datetime.now()
            # If loaded less than 24 hours ago, don't reload
            if (now - last_time).total_seconds() < 86400:  # 24 hours
                logger.info(f"Documentation automatically loaded {(now - last_time).total_seconds() / 3600:.2f} hours ago. Skipping.")
                
                # If vector store not initialized, initialize it with existing documents
                if not vector_store.initialized:
                    all_docs = find_all_docs()
                    if all_docs:
                        # Convert to format for vector store
                        docs_for_vector = [
                            {
                                'id': doc.get('topic', ''),
                                'content': f"{doc.get('summary', '')} {doc.get('content', '')}",
                                'metadata': {
                                    'topic': doc.get('topic', ''),
                                    'summary': doc.get('summary', ''),
                                    'importance': doc.get('importance', 5),
                                    'tags': doc.get('tags', [])
                                }
                            }
                            for doc in all_docs
                        ]
                        
                        added = vector_store.add_documents(docs_for_vector)
                        logger.info(f"Initialized vector store with {added} existing documents")
                        
                return "Automatic loading skipped (recent load detected)"
        except (ValueError, TypeError):
            pass  # If timestamp format is wrong, proceed with loading
    
    # Load project documentation
    load_count = load_project_docs(True)
    
    # Update last load time
    manager.set_metadata('last_auto_load_time', datetime.datetime.now().isoformat())
    
    # Build vector store with all documents
    all_docs = find_all_docs()
    if all_docs:
        # Convert to format for vector store
        docs_for_vector = [
            {
                'id': doc.get('topic', ''),
                'content': f"{doc.get('summary', '')} {doc.get('content', '')}",
                'metadata': {
                    'topic': doc.get('topic', ''),
                    'summary': doc.get('summary', ''),
                    'importance': doc.get('importance', 5),
                    'tags': doc.get('tags', [])
                }
            }
            for doc in all_docs
        ]
        
        added = vector_store.add_documents(docs_for_vector)
        logger.info(f"Built vector store with {added} documents")
    
    return f"Automatic loading complete: {load_count} documents loaded"


def remember(task: str, goal: str, progress: str, next_steps: str, notes: Optional[str] = None, 
             priority: int = 5, tags: Optional[List[str]] = None) -> str:
    """
    Remember current context with enhanced metadata and vector indexing.
    
    Args:
        task: What I'm currently doing
        goal: Why I'm doing it
        progress: Current step in the process
        next_steps: What to do next
        notes: Additional notes
        priority: Priority level (1-10)
        tags: List of tags for categorization
        
    Returns:
        Confirmation message
    """
    manager = get_state_manager()
    context_key = f"memory-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create context data
    kwargs = {
        'task': task,
        'goal': goal,
        'progress': progress,
        'next_steps': next_steps
    }
    
    # Add optional fields
    if notes:
        kwargs['notes'] = notes
        
    # Store additional metadata
    manager.set_metadata(f"context_{context_key}_priority", priority)
    
    if tags:
        manager.set_metadata(f"context_{context_key}_tags", tags)
    
    # Update context
    manager.update_context(context_key, **kwargs)
    
    # Store in history
    context_history = manager.get_metadata('context_history', [])
    context_history.append({
        'key': context_key,
        'task': task,
        'timestamp': datetime.datetime.now().isoformat(),
        'priority': priority,
        'tags': tags or []
    })
    
    # Keep only the most recent 100 entries to avoid excessive growth
    if len(context_history) > 100:
        context_history = context_history[-100:]
        
    manager.set_metadata('context_history', context_history)
    
    # If vector store is initialized, add this context to it as well
    if vector_store.initialized:
        context_content = f"{task} {goal} {progress} {next_steps}"
        if notes:
            context_content += f" {notes}"
            
        vector_store.add_document(
            context_key,
            context_content,
            {
                'type': 'context',
                'task': task,
                'priority': priority,
                'tags': tags or [],
                'timestamp': datetime.datetime.now().isoformat()
            }
        )
    
    return f"Context saved as {context_key} with priority {priority}"


def recall(search_term: Optional[str] = None, last: bool = False, 
           tag: Optional[str] = None, days: Optional[int] = None,
           priority_min: Optional[int] = None, semantic: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Recall context information with enhanced search capabilities and embedding-based retrieval.
    
    Args:
        search_term: Search term to find in contexts
        last: Whether to get only the most recent context
        tag: Filter by tag
        days: Only include contexts from the last X days
        priority_min: Minimum priority level (1-10)
        semantic: Whether to use semantic search with embeddings
        
    Returns:
        Context information or list of contexts
    """
    manager = get_state_manager()
    
    # Get most recent context if requested
    if last:
        context = manager.get_context()
        if not context:
            return "No context found."
            
        # Get metadata
        context_key = context.get('context_key')
        priority = manager.get_metadata(f"context_{context_key}_priority", 5)
        tags = manager.get_metadata(f"context_{context_key}_tags", [])
        
        result = {
            'task': context.get('task'),
            'goal': context.get('goal'),
            'progress': context.get('progress'),
            'next_steps': context.get('next_steps'),
            'priority': priority,
            'tags': tags,
            'key': context_key
        }
        
        if context.get('notes'):
            result['notes'] = context.get('notes')
            
        return result
    
    # Use semantic search if requested and available
    if search_term and semantic and vector_store.initialized:
        # Search vector store for contexts
        vector_results = vector_store.search(search_term, top_k=10)
        
        # Filter for context type
        context_results = [r for r in vector_results if r.get('metadata', {}).get('type') == 'context']
        
        # Apply additional filters
        filtered_results = []
        for result in context_results:
            metadata = result.get('metadata', {})
            context_key = result.get('id')
            
            # Skip if doesn't match filters
            if tag and tag not in metadata.get('tags', []):
                continue
                
            if priority_min is not None and metadata.get('priority', 0) < priority_min:
                continue
                
            if days is not None:
                try:
                    timestamp = metadata.get('timestamp')
                    if timestamp:
                        result_time = datetime.datetime.fromisoformat(timestamp)
                        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
                        if result_time < cutoff:
                            continue
                except (ValueError, TypeError):
                    pass
            
            # Get full context from database
            context = manager.get_context(context_key)
            if context:
                filtered_results.append({
                    'key': context_key,
                    'task': context.get('task'),
                    'goal': context.get('goal'),
                    'progress': context.get('progress'),
                    'next_steps': context.get('next_steps'),
                    'priority': metadata.get('priority', 5),
                    'tags': metadata.get('tags', []),
                    'similarity': result.get('similarity', 0)
                })
                
        if filtered_results:
            # Sort by similarity and recency
            filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            return filtered_results
    
    # Get context history
    history = manager.get_metadata('context_history', [])
    
    # Apply filters
    results = []
    
    # Filter by days
    if days is not None:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        history = [h for h in history if h.get('timestamp', '') >= cutoff_str]
    
    # Filter by priority
    if priority_min is not None:
        history = [h for h in history if h.get('priority', 0) >= priority_min]
        
    # Filter by tag
    if tag:
        history = [h for h in history if tag in h.get('tags', [])]
    
    # Collect matching contexts
    matching_keys = set()
    
    # Direct database search for text content
    if search_term:
        # Get all matching contexts from database
        manager.cursor.execute("""
            SELECT context_key, task, goal, progress, next_steps, notes, updated_at
            FROM context
            WHERE 
                task LIKE ? OR
                goal LIKE ? OR
                progress LIKE ? OR
                next_steps LIKE ? OR
                notes LIKE ?
            ORDER BY updated_at DESC
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", 
              f"%{search_term}%", f"%{search_term}%"))
        
        for row in manager.cursor.fetchall():
            context_key = row['context_key']
            matching_keys.add(context_key)
            
            # Find metadata in history
            context_meta = next((h for h in history if h.get('key') == context_key), {})
            priority = context_meta.get('priority', 5)
            tags = context_meta.get('tags', [])
            
            results.append({
                'key': context_key,
                'task': row['task'],
                'goal': row['goal'],
                'progress': row['progress'],
                'next_steps': row['next_steps'],
                'updated_at': row['updated_at'],
                'priority': priority,
                'tags': tags
            })
    else:
        # No search term, return filtered history
        for entry in history:
            context_key = entry.get('key')
            if not context_key:
                continue
                
            # Get the full context
            context = manager.get_context(context_key)
            if not context:
                continue
                
            results.append({
                'key': context_key,
                'task': context.get('task'),
                'goal': context.get('goal'),
                'progress': context.get('progress'),
                'next_steps': context.get('next_steps'),
                'updated_at': context.get('updated_at'),
                'priority': entry.get('priority', 5),
                'tags': entry.get('tags', [])
            })
    
    # Sort by priority and recency
    results.sort(key=lambda x: (-(x.get('priority', 0)), -(x.get('updated_at', ''))), reverse=True)
    
    if not results:
        if search_term:
            return f"No contexts found matching '{search_term}'."
        else:
            return "No matching contexts found."
            
    return results


def save_docs(topic: str, content: str, summary: Optional[str] = None, 
             importance: int = 7, source: Optional[str] = None,
             tags: Optional[List[str]] = None, related: Optional[List[str]] = None) -> str:
    """
    Save documentation with enhanced vector storage for semantic search.
    
    Args:
        topic: Documentation topic
        content: Main documentation content
        summary: Short summary of documentation
        importance: Importance level (1-10)
        source: Source of documentation
        tags: List of tags for categorization
        related: List of related documentation topics
        
    Returns:
        Confirmation message
    """
    manager = get_state_manager()
    
    # Track history of documentation changes
    doc_history = manager.get_metadata(f"doc_history_{topic}", [])
    
    # Get current doc if exists
    existing_doc = manager.get_documentation(topic=topic)
    if existing_doc:
        # Add to history
        history_entry = {
            'version': len(doc_history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'content': existing_doc.get('content', ''),
            'summary': existing_doc.get('summary', '')
        }
        doc_history.append(history_entry)
    
    # Add documentation
    manager.add_documentation(
        topic,
        content,
        summary=summary or "",
        source=source or "",
        examples="",
        related_topics=",".join(related) if related else "",
        importance=importance
    )
    
    # Save history
    manager.set_metadata(f"doc_history_{topic}", doc_history)
    
    # Save tags
    if tags:
        manager.set_metadata(f"doc_tags_{topic}", tags)
        
        # Update global tag index
        all_tags = manager.get_metadata("all_doc_tags", {})
        for tag in tags:
            if tag not in all_tags:
                all_tags[tag] = []
            if topic not in all_tags[tag]:
                all_tags[tag].append(topic)
        manager.set_metadata("all_doc_tags", all_tags)
    
    # Update related documents (bidirectional relationships)
    if related:
        for rel_topic in related:
            # Get existing related topics for the related document
            rel_doc = manager.get_documentation(topic=rel_topic)
            if rel_doc:
                rel_related = rel_doc.get('related_topics', '').split(',')
                rel_related = [r.strip() for r in rel_related if r.strip()]
                if topic not in rel_related:
                    rel_related.append(topic)
                    manager.add_documentation(
                        rel_topic,
                        rel_doc.get('content', ''),
                        summary=rel_doc.get('summary', ''),
                        source=rel_doc.get('source', ''),
                        examples=rel_doc.get('examples', ''),
                        related_topics=",".join(rel_related),
                        importance=rel_doc.get('importance', 5)
                    )
    
    # Add to vector store if initialized
    if vector_store.initialized:
        # Combine summary and content for better embedding
        full_content = content
        if summary:
            full_content = f"{summary}\n\n{content}"
            
        vector_store.add_document(
            topic,
            full_content,
            {
                'type': 'documentation',
                'topic': topic, 
                'summary': summary or "",
                'importance': importance,
                'tags': tags or [],
                'timestamp': datetime.datetime.now().isoformat()
            }
        )
        
        # If proper embeddings are available, automatically find related documents
        if proper_embeddings_available and not related:
            # Find semantically similar documents
            similar_docs = vector_store.find_related_documents(topic, top_k=5)
            
            # Add relationships for highly similar documents
            for doc in similar_docs:
                if doc.get('similarity', 0) > 0.7:  # High similarity threshold
                    rel_topic = doc.get('id')
                    if rel_topic:
                        # Update this document's related topics
                        rel_doc = manager.get_documentation(topic=rel_topic)
                        if rel_doc:
                            # Add to related topics
                            current_related = manager.get_documentation(topic=topic).get('related_topics', '').split(',')
                            current_related = [r.strip() for r in current_related if r.strip()]
                            if rel_topic not in current_related:
                                current_related.append(rel_topic)
                                
                                manager.add_documentation(
                                    topic,
                                    content,
                                    summary=summary or "",
                                    source=source or "",
                                    examples="",
                                    related_topics=",".join(current_related),
                                    importance=importance
                                )
                                
                                # Add to related document's related topics
                                rel_related = rel_doc.get('related_topics', '').split(',')
                                rel_related = [r.strip() for r in rel_related if r.strip()]
                                if topic not in rel_related:
                                    rel_related.append(topic)
                                    
                                    manager.add_documentation(
                                        rel_topic,
                                        rel_doc.get('content', ''),
                                        summary=rel_doc.get('summary', ''),
                                        source=rel_doc.get('source', ''),
                                        examples=rel_doc.get('examples', ''),
                                        related_topics=",".join(rel_related),
                                        importance=rel_doc.get('importance', 5)
                                    )
        
    return f"Documentation saved: {topic} (importance: {importance})"


def find_docs(topic: Optional[str] = None, search: Optional[str] = None,
              tag: Optional[str] = None, importance_min: int = 0,
              semantic: bool = True, max_results: int = 10) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Find documentation with advanced semantic search capabilities.
    
    Args:
        topic: Specific topic to find
        search: Search term in documentation
        tag: Filter by tag
        importance_min: Minimum importance level (1-10)
        semantic: Whether to use semantic search (defaults to True with fallback)
        max_results: Maximum results to return
        
    Returns:
        Documentation or list of documentation
    """
    manager = get_state_manager()
    
    # Get specific topic
    if topic:
        doc = manager.get_documentation(topic=topic)
        if not doc:
            return f"No documentation found for topic: {topic}"
            
        # Add version history
        doc['versions'] = manager.get_metadata(f"doc_history_{topic}", [])
        # Add tags
        doc['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
        
        # If vector store is available, find related documents by vector similarity
        if vector_store.initialized and proper_embeddings_available:
            similar_docs = vector_store.find_related_documents(topic, top_k=5)
            if similar_docs:
                # Get full docs for similar documents
                related_by_vector = []
                for similar in similar_docs:
                    sim_topic = similar.get('id')
                    sim_doc = manager.get_documentation(topic=sim_topic)
                    if sim_doc:
                        related_by_vector.append({
                            'topic': sim_topic,
                            'summary': sim_doc.get('summary', ''),
                            'similarity': similar.get('similarity', 0)
                        })
                        
                doc['similar_docs'] = related_by_vector
        
        return doc
    
    # Semantic search
    if search and semantic and vector_store.initialized:
        vector_results = vector_store.search(search, top_k=max_results*2)  # Get more for filtering
        
        # Filter for documentation type
        doc_results = [r for r in vector_results if r.get('metadata', {}).get('type') == 'documentation']
        
        if doc_results:
            # Apply tag filter
            if tag:
                tag_topics = manager.get_metadata("all_doc_tags", {}).get(tag, [])
                doc_results = [r for r in doc_results if r.get('id') in tag_topics]
                
            # Apply importance filter
            if importance_min > 0:
                doc_results = [r for r in doc_results if r.get('metadata', {}).get('importance', 0) >= importance_min]
                
            # Get full documents for each result
            results = []
            for result in doc_results[:max_results]:
                topic = result.get('id')
                doc = manager.get_documentation(topic=topic)
                if doc:
                    doc['similarity'] = result.get('similarity', 0)
                    doc['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
                    results.append(doc)
                    
            if results:
                # Sort by similarity
                results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                return results
    
    # Regular search
    if search:
        results = manager.get_documentation(search_term=search)
        
        # Apply tag filter
        if tag and results:
            tag_topics = manager.get_metadata("all_doc_tags", {}).get(tag, [])
            results = [r for r in results if r.get('topic') in tag_topics]
            
        # Apply importance filter
        if importance_min > 0 and results:
            results = [r for r in results if r.get('importance', 0) >= importance_min]
            
        # Add tags to results
        for result in results:
            topic = result.get('topic')
            if topic:
                result['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
        
        # Limit results
        results = results[:max_results]
            
        if not results:
            return f"No documentation found matching '{search}'"
            
        return results
    
    # Filter by tag only
    if tag:
        tag_topics = manager.get_metadata("all_doc_tags", {}).get(tag, [])
        if not tag_topics:
            return f"No documentation found with tag: {tag}"
            
        results = []
        for topic in tag_topics:
            doc = manager.get_documentation(topic=topic)
            if doc and doc.get('importance', 0) >= importance_min:
                doc['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
                results.append(doc)
                
        results.sort(key=lambda x: (-x.get('importance', 0), x.get('topic', '')))
        return results[:max_results]
    
    # Get all topics
    topics = manager.get_documentation()
    
    # Apply importance filter
    if importance_min > 0 and topics:
        topics = [t for t in topics if t.get('importance', 0) >= importance_min]
        
    # Add tags to results
    for topic in topics:
        topic_name = topic.get('topic')
        if topic_name:
            topic['tags'] = manager.get_metadata(f"doc_tags_{topic_name}", [])
    
    # Sort by importance
    topics.sort(key=lambda x: (-x.get('importance', 0), x.get('topic', '')))
    
    if not topics:
        return "No documentation available."
        
    return topics[:max_results]


def find_all_docs() -> List[Dict[str, Any]]:
    """Get all documentation entries with full content."""
    manager = get_state_manager()
    
    # Get list of all topics
    topic_list = manager.get_documentation()
    if not topic_list:
        return []
        
    # Get full content for each topic
    results = []
    for topic_entry in topic_list:
        topic = topic_entry.get('topic')
        if topic:
            full_doc = manager.get_documentation(topic=topic)
            if full_doc:
                # Add tags
                full_doc['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
                results.append(full_doc)
                
    return results


def log_error(error_type: str, details: str, recovery_action: Optional[str] = None, 
             severity: int = 5, tags: Optional[List[str]] = None) -> str:
    """
    Log an error pattern for future prevention with embedding-based similarity.
    
    Args:
        error_type: Type of error
        details: Error details
        recovery_action: How to recover from this error
        severity: Error severity level (1-10)
        tags: List of tags for categorization
        
    Returns:
        Confirmation message
    """
    manager = get_state_manager()
    
    # Store in metadata
    errors = manager.get_metadata('error_patterns', [])
    
    # Create error entry
    error_entry = {
        'error_type': error_type,
        'details': details,
        'recovery_action': recovery_action,
        'severity': severity,
        'tags': tags or [],
        'timestamp': datetime.datetime.now().isoformat(),
        'context_key': manager.current_checkpoint
    }
    
    # Link to current context if available
    current_context = manager.get_context()
    if current_context:
        error_entry['task'] = current_context.get('task')
    
    errors.append(error_entry)
    
    # Store updated errors
    manager.set_metadata('error_patterns', errors)
    
    # Update error type index for faster lookups
    error_type_index = manager.get_metadata('error_type_index', {})
    if error_type not in error_type_index:
        error_type_index[error_type] = []
    
    # Store index of this error in the errors list
    error_type_index[error_type].append(len(errors) - 1)
    manager.set_metadata('error_type_index', error_type_index)
    
    # For high severity errors, create documentation entry
    if severity >= 8:
        save_docs(
            f"error-pattern-{error_type}",
            f"# Error Pattern: {error_type}\n\n"
            f"**Severity:** {severity}/10\n\n"
            f"## Details\n{details}\n\n"
            f"## Recovery Action\n{recovery_action or 'No recovery action specified'}\n\n"
            f"## Tags\n{', '.join(tags) if tags else 'No tags'}",
            summary=f"High severity error pattern: {error_type}",
            importance=severity,
            tags=['error-pattern'] + (tags or [])
        )
    
    # Add to vector store if available
    if vector_store.initialized:
        # Use combined content for best matching
        content = f"Error: {error_type} Details: {details}"
        if recovery_action:
            content += f" Recovery: {recovery_action}"
            
        vector_store.add_document(
            f"error-{error_type}-{len(errors)}",
            content,
            {
                'type': 'error',
                'error_type': error_type,
                'details': details,
                'recovery_action': recovery_action,
                'severity': severity,
                'tags': tags or [],
                'timestamp': datetime.datetime.now().isoformat()
            }
        )
    
    return f"Error logged: {error_type} (severity: {severity})"


def suggest_recovery(error_type: Optional[str] = None, details: Optional[str] = None,
                    use_semantic: bool = True) -> Union[str, List[Dict[str, Any]]]:
    """
    Suggest recovery based on past similar errors with embedding-based matching.
    
    Args:
        error_type: Type of error to look up
        details: Error details for semantic matching
        use_semantic: Whether to use semantic matching for error details
        
    Returns:
        Recovery suggestion or list of error types
    """
    manager = get_state_manager()
    
    # Get all errors
    errors = manager.get_metadata('error_patterns', [])
    
    # Case 1: Specific error type provided
    if error_type:
        # Use index for faster lookup
        error_type_index = manager.get_metadata('error_type_index', {})
        indices = error_type_index.get(error_type, [])
        
        if indices:
            matching = [errors[i] for i in indices if i < len(errors)]
        else:
            matching = [e for e in errors if e.get('error_type') == error_type]
            
        if not matching:
            return f"No recovery suggestions found for error type: {error_type}"
        
        # Sort by recency and severity
        matching.sort(key=lambda e: (
            e.get('timestamp', ''), 
            e.get('severity', 0)
        ), reverse=True)
        
        # Return most relevant match
        if details and use_semantic and vector_store.initialized:
            # Use vector store for semantic matching
            error_query = f"Error: {error_type} Details: {details}"
            vector_results = vector_store.search(error_query, top_k=3)
            
            # Filter for error type
            error_results = [r for r in vector_results if r.get('metadata', {}).get('type') == 'error']
            
            if error_results:
                # Get best match
                best_match = error_results[0]
                return best_match.get('metadata', {}).get('recovery_action', 'No recovery action specified')
                
        # Return most recent matching error
        return matching[0].get('recovery_action', 'No recovery action specified')
    
    # Case 2: Error details provided for semantic matching
    elif details and use_semantic and vector_store.initialized:
        # Use vector store for semantic matching
        vector_results = vector_store.search(details, top_k=5)
        
        # Filter for error type
        error_results = [r for r in vector_results if r.get('metadata', {}).get('type') == 'error']
        
        if error_results:
            # Prepare results
            results = []
            for result in error_results:
                metadata = result.get('metadata', {})
                results.append({
                    'error_type': metadata.get('error_type', 'unknown'),
                    'similarity': result.get('similarity', 0),
                    'recovery_action': metadata.get('recovery_action', 'No recovery action specified'),
                    'severity': metadata.get('severity', 0)
                })
                
            return results
    
    # Case 3: Return all error types with counts
    error_counts = {}
    for e in errors:
        error_type = e.get('error_type', 'unknown')
        if error_type not in error_counts:
            error_counts[error_type] = {'count': 0, 'max_severity': 0}
        
        error_counts[error_type]['count'] += 1
        error_counts[error_type]['max_severity'] = max(
            error_counts[error_type]['max_severity'],
            e.get('severity', 0)
        )
    
    return [
        {
            'error_type': error_type,
            'count': stats['count'],
            'max_severity': stats['max_severity']
        }
        for error_type, stats in error_counts.items()
    ]


def batch_operation(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute multiple memory operations in a single batch.
    
    Args:
        operations: List of operation dictionaries with 'op' key and parameters
        
    Returns:
        List of operation results
    """
    results = []
    
    for op in operations:
        op_type = op.get('op')
        
        try:
            if op_type == 'remember':
                result = remember(
                    op.get('task', ''),
                    op.get('goal', ''),
                    op.get('progress', ''),
                    op.get('next_steps', ''),
                    op.get('notes'),
                    op.get('priority', 5),
                    op.get('tags')
                )
            elif op_type == 'recall':
                result = recall(
                    op.get('search'),
                    op.get('last', False),
                    op.get('tag'),
                    op.get('days'),
                    op.get('priority_min'),
                    op.get('semantic', False)
                )
            elif op_type == 'save_docs':
                result = save_docs(
                    op.get('topic', ''),
                    op.get('content', ''),
                    op.get('summary'),
                    op.get('importance', 7),
                    op.get('source'),
                    op.get('tags'),
                    op.get('related')
                )
            elif op_type == 'find_docs':
                result = find_docs(
                    op.get('topic'),
                    op.get('search'),
                    op.get('tag'),
                    op.get('importance_min', 0),
                    op.get('semantic', True),
                    op.get('max_results', 10)
                )
            elif op_type == 'log_error':
                result = log_error(
                    op.get('error_type', ''),
                    op.get('details', ''),
                    op.get('recovery_action'),
                    op.get('severity', 5),
                    op.get('tags')
                )
            elif op_type == 'suggest_recovery':
                result = suggest_recovery(
                    op.get('error_type'),
                    op.get('details'),
                    op.get('use_semantic', True)
                )
            else:
                result = f"Unknown operation type: {op_type}"
                
        except Exception as e:
            result = {"error": str(e), "operation": op_type}
            
        results.append(result)
        
    return results


def get_system_status():
    """
    Get comprehensive status of the memory system.
    
    Returns:
        Dictionary with system status information
    """
    manager = get_state_manager()
    
    try:
        # Check if database is working
        manager.cursor.execute("SELECT COUNT(*) FROM state")
        state_count = manager.cursor.fetchone()[0]
        
        manager.cursor.execute("SELECT COUNT(*) FROM documentation")
        doc_count = manager.cursor.fetchone()[0]
        
        manager.cursor.execute("SELECT COUNT(*) FROM context")
        context_count = manager.cursor.fetchone()[0]
        
        # Check vector store status
        vector_status = {
            "available": vector_store.initialized,
            "embedding_method": vector_store.embedding_method if vector_store.initialized else None,
            "embedding_model": vector_store.embedding_model if vector_store.initialized else None,
            "proper_embeddings": proper_embeddings_available,
            "document_count": len(vector_store.doc_ids) if vector_store.initialized else 0
        }
        
        # Check for staleness
        staleness = detect_context_staleness()
        
        # Get current checkpoint
        current_checkpoint = manager.current_checkpoint
        
        # Get active work session
        current_session_id = manager.get_metadata('current_session')
        current_session = None
        
        if current_session_id:
            sessions = manager.get_metadata('work_sessions', [])
            for session in sessions:
                if session.get('id') == current_session_id:
                    current_session = session
                    break
        
        # Get recent verifications
        recent_verifications = manager.get_verification_history()[-5:] if manager.get_verification_history() else []
        
        return {
            "status": "operational",
            "database": {
                "state_entries": state_count,
                "documentation_entries": doc_count,
                "context_entries": context_count
            },
            "vector_store": vector_status,
            "current_checkpoint": current_checkpoint,
            "context_staleness": staleness,
            "current_session": current_session,
            "recent_verifications": recent_verifications,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


# -------------------------------------------------
# CLI Interface
# -------------------------------------------------

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Memory System for AI operations with vector search"
    )
    
    parser.add_argument(
        "--db-path",
        help="Path to the state database (defaults to in-memory)",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize memory system")
    init_parser.add_argument("--skip-docs", action="store_true", help="Skip loading documentation")
    
    # Remember command
    remember_parser = subparsers.add_parser("remember", help="Remember current context")
    remember_parser.add_argument("task", help="What I'm currently doing")
    remember_parser.add_argument("goal", help="Why I'm doing it")
    remember_parser.add_argument("progress", help="Current step in the process")
    remember_parser.add_argument("next_steps", help="What to do next")
    remember_parser.add_argument("--notes", "-n", help="Additional notes")
    remember_parser.add_argument(
        "--priority",
        "-p", 
        type=int,
        choices=range(1, 11),
        default=5,
        help="Priority level (1-10)"
    )
    remember_parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        help="Tags for categorization"
    )
    
    # Recall command
    recall_parser = subparsers.add_parser("recall", help="Recall context information")
    recall_parser.add_argument("--search", "-s", help="Search term in contexts")
    recall_parser.add_argument("--last", "-l", action="store_true", help="Get most recent context")
    recall_parser.add_argument("--tag", "-t", help="Filter by tag")
    recall_parser.add_argument("--days", "-d", type=int, help="Only include contexts from the last X days")
    recall_parser.add_argument(
        "--priority-min",
        "-p",
        type=int,
        choices=range(1, 11),
        help="Minimum priority level (1-10)"
    )
    recall_parser.add_argument("--semantic", "-v", action="store_true", help="Use semantic search")
    
    # Document commands
    save_docs_parser = subparsers.add_parser("save-docs", help="Save documentation")
    save_docs_parser.add_argument("topic", help="Documentation topic")
    save_docs_parser.add_argument("content", help="Documentation content")
    save_docs_parser.add_argument("--summary", "-s", help="Documentation summary")
    save_docs_parser.add_argument(
        "--importance",
        "-i",
        type=int,
        choices=range(1, 11),
        default=7,
        help="Importance level (1-10)"
    )
    save_docs_parser.add_argument("--source", help="Source of documentation")
    save_docs_parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        help="Tags for categorization"
    )
    save_docs_parser.add_argument(
        "--related",
        "-r",
        nargs="+",
        help="Related documentation topics"
    )
    
    find_docs_parser = subparsers.add_parser("find-docs", help="Find documentation")
    find_docs_parser.add_argument("--topic", "-t", help="Specific topic to find")
    find_docs_parser.add_argument("--search", "-s", help="Search term in documentation")
    find_docs_parser.add_argument("--tag", help="Filter by tag")
    find_docs_parser.add_argument(
        "--importance-min",
        "-i",
        type=int,
        choices=range(1, 11),
        default=0,
        help="Minimum importance level (1-10)"
    )
    find_docs_parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search")
    find_docs_parser.add_argument(
        "--max-results",
        "-m",
        type=int,
        default=10,
        help="Maximum results to return"
    )
    
    # Error handling commands
    error_parser = subparsers.add_parser("log-error", help="Log an error pattern")
    error_parser.add_argument("error_type", help="Type of error")
    error_parser.add_argument("details", help="Error details")
    error_parser.add_argument("--recovery", "-r", help="Recovery action")
    error_parser.add_argument(
        "--severity",
        "-s",
        type=int,
        choices=range(1, 11),
        default=5,
        help="Error severity level (1-10)"
    )
    error_parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        help="Tags for categorization"
    )
    
    recovery_parser = subparsers.add_parser("suggest-recovery", help="Suggest error recovery")
    recovery_parser.add_argument("--error-type", "-e", help="Type of error")
    recovery_parser.add_argument("--details", "-d", help="Error details for semantic matching")
    recovery_parser.add_argument("--no-semantic", action="store_true", help="Disable semantic matching")
    
    # Batch operation command
    batch_parser = subparsers.add_parser("batch", help="Execute multiple operations in a batch")
    batch_parser.add_argument("operations_file", help="JSON file with operations to execute")
    
    # System status command
    status_parser = subparsers.add_parser("status", help="Get memory system status")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return
    
    # Initialize state manager
    manager = get_state_manager(args.db_path)
    
    # Dispatch to command handlers
    result = None
    
    if args.command == "init":
        result = initialize_memory_system(args.db_path, not args.skip_docs)
        
    elif args.command == "remember":
        result = remember(
            args.task, 
            args.goal, 
            args.progress, 
            args.next_steps, 
            args.notes, 
            args.priority, 
            args.tags
        )
        
    elif args.command == "recall":
        result = recall(
            args.search, 
            args.last, 
            args.tag, 
            args.days, 
            args.priority_min,
            args.semantic
        )
        
    elif args.command == "save-docs":
        result = save_docs(
            args.topic, 
            args.content, 
            args.summary, 
            args.importance, 
            args.source, 
            args.tags, 
            args.related
        )
        
    elif args.command == "find-docs":
        result = find_docs(
            args.topic, 
            args.search, 
            args.tag, 
            args.importance_min, 
            not args.no_semantic, 
            args.max_results
        )
        
    elif args.command == "log-error":
        result = log_error(
            args.error_type, 
            args.details, 
            args.recovery, 
            args.severity, 
            args.tags
        )
        
    elif args.command == "suggest-recovery":
        result = suggest_recovery(
            args.error_type, 
            args.details, 
            not args.no_semantic
        )
        
    elif args.command == "batch":
        try:
            with open(args.operations_file, 'r') as f:
                operations = json.load(f)
            result = batch_operation(operations)
        except Exception as e:
            result = {"error": str(e)}
            
    elif args.command == "status":
        result = get_system_status()
    
    # Print result
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()