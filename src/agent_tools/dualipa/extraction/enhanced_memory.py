#!/usr/bin/env python3
"""
Enhanced Memory Management System

An improved memory system for AI assistant operations, focusing on reliability, 
context preservation, and learning from past failures. This system extends the 
base TestStateManager functionality with enhanced features for semantic search,
automatic documentation loading, relationship tracking, and error pattern learning.

Key Improvements:
1. Semantic search with embeddings for better document retrieval
2. Automatic documentation loading from project directories
3. Versioning and history tracking for knowledge evolution
4. Relationship mapping between related contexts and documents
5. Proactive context management with staleness detection
6. Error pattern learning and suggestion mechanisms
7. Hierarchical documentation organization
8. Command aliases and batch operations for efficiency
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
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from collections import defaultdict

# Try to import embedding utilities for semantic search
try:
    # For semantic search support
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    semantic_search_available = True
except ImportError:
    semantic_search_available = False
    print("Semantic search capabilities not available. Install sklearn and numpy for enhanced search.")

# Import base TestStateManager functionality
from .test_state_manager import (
    get_state_manager, 
    what_am_i_doing, 
    remember_context, 
    add_docs, 
    get_docs
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
logger = logging.getLogger("memory_system")

# -------------------------------------------------
# Core Memory Enhancement Functions
# -------------------------------------------------

class SemanticIndex:
    """Semantic indexing for documentation with TF-IDF vectors."""
    
    def __init__(self):
        """Initialize the semantic index."""
        self.available = semantic_search_available
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        self.doc_ids = []
        self.initialized = False
        
    def build_index(self, docs: List[Dict[str, Any]]):
        """Build index from documentation entries."""
        if not self.available or not docs:
            return False
            
        try:
            # Extract content and IDs
            contents = []
            self.doc_ids = []
            
            for doc in docs:
                content = doc.get('content', '')
                if not content:
                    continue
                    
                # Include summary in the content for better matching
                summary = doc.get('summary', '')
                if summary:
                    content = f"{summary}\n\n{content}"
                    
                contents.append(content)
                self.doc_ids.append(doc.get('topic'))
                
            # If we have documents, build the vector index
            if contents:
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.doc_vectors = self.vectorizer.fit_transform(contents)
                self.documents = docs
                self.initialized = True
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error building semantic index: {e}")
            return False
            
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        if not self.initialized or not self.available:
            return []
            
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity with all documents
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get indices of top matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Only include results with at least some similarity
            results = []
            for i in top_indices:
                if similarities[i] > 0.1:  # Minimum similarity threshold
                    doc = self.documents[i]
                    # Add similarity score
                    doc_with_score = doc.copy()
                    doc_with_score['similarity_score'] = float(similarities[i])
                    results.append(doc_with_score)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []


# Global semantic index instance
semantic_index = SemanticIndex()


def set_up_automatic_loading(db_path: Optional[str] = None):
    """Set up automatic documentation loading at startup."""
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
                return "Automatic loading skipped (recent load detected)"
        except (ValueError, TypeError):
            pass  # If timestamp format is wrong, proceed with loading
    
    # Load project documentation
    load_count = load_project_docs(True)
    
    # Update last load time
    manager.set_metadata('last_auto_load_time', datetime.datetime.now().isoformat())
    
    # Build semantic index
    all_docs = find_all_docs()
    if semantic_index.available:
        if semantic_index.build_index(all_docs):
            logger.info(f"Built semantic index with {len(all_docs)} documents")
        else:
            logger.warning("Failed to build semantic index")
    
    return f"Automatic loading complete: {load_count}"


def remember(task: str, goal: str, progress: str, next_steps: str, notes: Optional[str] = None, 
             priority: int = 5, tags: Optional[List[str]] = None) -> str:
    """
    Remember current context with enhanced metadata.
    
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
    
    return f"Context saved as {context_key} with priority {priority}"


def recall(search_term: Optional[str] = None, last: bool = False, 
           tag: Optional[str] = None, days: Optional[int] = None,
           priority_min: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Recall context information with enhanced search capabilities.
    
    Args:
        search_term: Search term to find in contexts
        last: Whether to get only the most recent context
        tag: Filter by tag
        days: Only include contexts from the last X days
        priority_min: Minimum priority level (1-10)
        
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
    Save documentation with enhanced metadata.
    
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
    
    # Rebuild semantic index
    if semantic_index.available:
        all_docs = find_all_docs()
        semantic_index.build_index(all_docs)
        
    return f"Documentation saved: {topic} (importance: {importance})"


def find_docs(topic: Optional[str] = None, search: Optional[str] = None,
              tag: Optional[str] = None, importance_min: int = 0,
              semantic: bool = False, max_results: int = 10) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Find documentation with enhanced search capabilities.
    
    Args:
        topic: Specific topic to find
        search: Search term in documentation
        tag: Filter by tag
        importance_min: Minimum importance level (1-10)
        semantic: Whether to use semantic search (if available)
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
        
        return doc
    
    # Semantic search
    if search and semantic and semantic_index.available and semantic_index.initialized:
        results = semantic_index.search(search, top_k=max_results)
        if results:
            # Enhance results with tags
            for result in results:
                topic = result.get('topic')
                if topic:
                    result['tags'] = manager.get_metadata(f"doc_tags_{topic}", [])
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


def load_project_docs(force: bool = False) -> str:
    """
    Load documentation from the project docs directory with enhanced structure.
    
    Args:
        force: Force reload existing docs
        
    Returns:
        Loading result message
    """
    manager = get_state_manager()
    docs_path = Path("/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/docs")
    
    if not docs_path.exists():
        return f"Documentation path does not exist: {docs_path}"
    
    loaded_count = 0
    skipped_count = 0
    
    # Track all document paths for hierarchy building
    doc_paths = list(docs_path.glob("**/*.md"))
    doc_paths.sort()
    
    # First pass: load all documents
    for doc_file in doc_paths:
        # Determine topic and path-based tags
        rel_path = doc_file.relative_to(docs_path)
        
        # Clean up topic name - use path structure for uniqueness
        path_parts = list(rel_path.parts)
        file_stem = path_parts[-1].replace('.md', '')
        
        # Special case for README files - use parent directory name
        if file_stem.lower() == 'readme' and len(path_parts) > 1:
            topic = f"{path_parts[-2]}/{file_stem}"
        else:
            topic = str(rel_path).replace('.md', '')
        
        # Skip if already loaded and not forcing reload
        if not force:
            existing_doc = manager.get_documentation(topic=topic)
            if existing_doc:
                skipped_count += 1
                continue
                
        with open(doc_file, 'r') as f:
            content = f.read()
            
            # Extract title from first header or use filename
            title = file_stem
            for line in content.split('\n'):
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
            
            # Extract summary (first paragraph after title)
            summary = ""
            in_paragraph = False
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    if in_paragraph:  # End of paragraph
                        break
                    continue
                if line.startswith('#'):  # Skip headers
                    continue
                if not in_paragraph:
                    in_paragraph = True
                if in_paragraph:
                    if not summary:
                        summary = line
                    else:
                        summary += " " + line
                    
            # Trim summary if too long
            if summary and len(summary) > 200:
                summary = summary[:197] + "..."
            
            # Generate tags from path
            tags = list(rel_path.parts)[:-1]  # All directory names
            if not tags and len(path_parts) > 1:
                tags = [path_parts[-2]]  # At least use parent directory
                
            # Calculate importance based on path depth and content
            # More specific (deeper) documents get higher importance
            importance = min(4 + len(tags) + (1 if 'important' in content.lower() else 0), 10)
            
            # Add documentation
            manager.add_documentation(
                topic,
                content,
                summary=summary,
                source=str(doc_file),
                examples="",
                related_topics="",
                importance=importance
            )
            
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
                
            loaded_count += 1
    
    # Second pass: establish relationships
    hierarchy = {}
    
    # Build directory hierarchy
    for doc_file in doc_paths:
        rel_path = doc_file.relative_to(docs_path)
        path_parts = list(rel_path.parts)
        
        # Skip for simplicity if more than 3 levels deep
        if len(path_parts) > 3:
            continue
            
        # Extract topic name
        file_stem = path_parts[-1].replace('.md', '')
        if file_stem.lower() == 'readme' and len(path_parts) > 1:
            topic = f"{path_parts[-2]}/{file_stem}"
        else:
            topic = str(rel_path).replace('.md', '')
            
        # Find potential parent documents
        parent_topics = []
        
        # Case 1: Files in same directory - README is parent
        if len(path_parts) > 1 and file_stem.lower() != 'readme':
            parent_topic = f"{path_parts[-2]}/README"
            parent_file = docs_path / path_parts[-2] / "README.md"
            if parent_file.exists():
                parent_topics.append(parent_topic)
                
        # Case 2: Files in subdirectories - parent directory README is parent
        if len(path_parts) > 1:
            for i in range(1, len(path_parts)):
                parent_dir = Path(*path_parts[:i])
                parent_file = docs_path / parent_dir / "README.md"
                if parent_file.exists():
                    parent_topic = f"{parent_dir}/README"
                    parent_topics.append(str(parent_topic))
        
        # Update relationships for this document
        doc = manager.get_documentation(topic=topic)
        if doc and parent_topics:
            # Set first parent as main parent
            related_topics = [p for p in parent_topics]
            
            # Update documentation with relationships
            manager.add_documentation(
                topic,
                doc.get('content', ''),
                summary=doc.get('summary', ''),
                source=doc.get('source', ''),
                examples=doc.get('examples', ''),
                related_topics=",".join(related_topics),
                importance=doc.get('importance', 5)
            )
            
            # Update parent documents
            for parent_topic in parent_topics:
                parent_doc = manager.get_documentation(topic=parent_topic)
                if parent_doc:
                    parent_related = parent_doc.get('related_topics', '').split(',')
                    parent_related = [r.strip() for r in parent_related if r.strip()]
                    if topic not in parent_related:
                        parent_related.append(topic)
                        
                    manager.add_documentation(
                        parent_topic,
                        parent_doc.get('content', ''),
                        summary=parent_doc.get('summary', ''),
                        source=parent_doc.get('source', ''),
                        examples=parent_doc.get('examples', ''),
                        related_topics=",".join(parent_related),
                        importance=parent_doc.get('importance', 5)
                    )
    
    # Build semantic index
    if semantic_index.available:
        all_docs = find_all_docs()
        if semantic_index.build_index(all_docs):
            logger.info(f"Built semantic index with {len(all_docs)} documents")
        else:
            logger.warning("Failed to build semantic index")
    
    return f"Loaded {loaded_count} documentation files from {docs_path} (skipped {skipped_count})"


def get_verification_summary(days: Optional[int] = None):
    """
    Get a summary of verification results with time filtering.
    
    Args:
        days: Only include verifications from the last X days
        
    Returns:
        Verification summary statistics
    """
    manager = get_state_manager()
    
    verifications = manager.get_verification_history()
    
    # Filter by time if requested
    if days is not None:
        filtered = []
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        
        for v in verifications:
            try:
                v_time = datetime.datetime.fromisoformat(v['timestamp'].replace(' ', 'T'))
                if v_time >= cutoff:
                    filtered.append(v)
            except (ValueError, TypeError):
                # Keep verifications with invalid timestamps
                filtered.append(v)
                
        verifications = filtered
    
    # Calculate statistics
    total = len(verifications)
    passed = sum(1 for v in verifications if v.get('passed', False))
    failed = total - passed
    
    # Group by checkpoint
    checkpoint_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
    for v in verifications:
        checkpoint = v.get('checkpoint', 'unknown')
        checkpoint_stats[checkpoint]['total'] += 1
        if v.get('passed', False):
            checkpoint_stats[checkpoint]['passed'] += 1
        else:
            checkpoint_stats[checkpoint]['failed'] += 1
    
    # Calculate pass rates
    for cp in checkpoint_stats.values():
        cp['pass_rate'] = (cp['passed'] / cp['total'] * 100) if cp['total'] > 0 else 0
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': (passed / total * 100) if total > 0 else 0,
        'by_checkpoint': dict(checkpoint_stats),
        'recent': verifications[-5:] if verifications else []
    }


def log_error(error_type: str, details: str, recovery_action: Optional[str] = None, 
             severity: int = 5, tags: Optional[List[str]] = None) -> str:
    """
    Log an error pattern for future prevention with enhanced metadata.
    
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
    
    return f"Error logged: {error_type} (severity: {severity})"


def suggest_recovery(error_type: Optional[str] = None, details: Optional[str] = None,
                    use_semantic: bool = False) -> Union[str, List[Dict[str, Any]]]:
    """
    Suggest recovery based on past similar errors with enhanced matching.
    
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
        if details and use_semantic and semantic_index.available:
            # Check if we can do semantic matching on error details
            try:
                # Simple semantic similarity using TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer().fit([e.get('details', '') for e in matching] + [details])
                vectors = vectorizer.transform([e.get('details', '') for e in matching])
                query_vector = vectorizer.transform([details])
                
                similarities = cosine_similarity(query_vector, vectors).flatten()
                best_idx = similarities.argmax()
                
                return matching[best_idx].get('recovery_action', 'No recovery action specified')
            except Exception as e:
                logger.warning(f"Error during semantic matching: {e}")
                
        # Return most recent matching error
        return matching[0].get('recovery_action', 'No recovery action specified')
    
    # Case 2: Error details provided for semantic matching
    elif details and use_semantic and semantic_index.available:
        try:
            # Use TF-IDF for semantic matching
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer().fit([e.get('details', '') for e in errors] + [details])
            vectors = vectorizer.transform([e.get('details', '') for e in errors])
            query_vector = vectorizer.transform([details])
            
            similarities = cosine_similarity(query_vector, vectors).flatten()
            
            # Get top 3 matches
            top_indices = similarities.argsort()[-3:][::-1]
            
            if not any(similarities[i] > 0.3 for i in top_indices):
                return "No similar error patterns found."
                
            results = []
            for i in top_indices:
                if similarities[i] > 0.3:  # Minimum similarity threshold
                    error = errors[i]
                    results.append({
                        'error_type': error.get('error_type', 'unknown'),
                        'similarity': float(similarities[i]),
                        'recovery_action': error.get('recovery_action', 'No recovery action specified'),
                        'severity': error.get('severity', 0)
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Error during semantic matching: {e}")
            # Fall back to returning all error types
    
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


def detect_context_staleness():
    """
    Detect if the current context is stale and needs refreshing.
    
    Returns:
        Dictionary with staleness information
    """
    manager = get_state_manager()
    context = manager.get_context()
    
    if not context:
        return {"stale": True, "reason": "No active context", "age_minutes": float('inf')}
    
    try:
        # Calculate age of context
        updated_at = context.get('updated_at')
        if not updated_at:
            return {"stale": True, "reason": "Context has no timestamp", "age_minutes": float('inf')}
            
        # Handle different timestamp formats
        if 'T' in updated_at:
            update_time = datetime.datetime.fromisoformat(updated_at)
        else:
            update_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S.%f")
            
        now = datetime.datetime.now()
        age_seconds = (now - update_time).total_seconds()
        age_minutes = age_seconds / 60
        
        # Check against staleness thresholds
        context_key = context.get('context_key', '')
        priority = manager.get_metadata(f"context_{context_key}_priority", 5)
        
        # Higher priority contexts have longer staleness thresholds
        max_age_minutes = 15 + (priority * 15)  # 30min for priority 1, 3 hours for priority 10
        
        stale = age_minutes > max_age_minutes
        
        return {
            "stale": stale,
            "reason": f"Context is {age_minutes:.1f} minutes old (threshold: {max_age_minutes})" if stale else "Context is current",
            "age_minutes": age_minutes,
            "threshold_minutes": max_age_minutes,
            "priority": priority
        }
        
    except Exception as e:
        logger.error(f"Error detecting context staleness: {e}")
        return {"stale": True, "reason": f"Error detecting staleness: {str(e)}", "age_minutes": float('inf')}


def track_work_session(session_type: str, start: bool = True, 
                      task: Optional[str] = None, result: Optional[str] = None) -> str:
    """
    Track work session start/end for productivity analysis.
    
    Args:
        session_type: Type of work session (extraction, analysis, etc.)
        start: Whether this is the start or end of a session
        task: Description of the task (required for session start)
        result: Result of the session (required for session end)
        
    Returns:
        Confirmation message
    """
    manager = get_state_manager()
    
    # Get existing sessions
    sessions = manager.get_metadata('work_sessions', [])
    
    timestamp = datetime.datetime.now().isoformat()
    
    if start:
        if not task:
            return "Error: Task description required for session start"
            
        # Create new session
        session_id = f"session-{timestamp}-{session_type}"
        new_session = {
            'id': session_id,
            'type': session_type,
            'start_time': timestamp,
            'task': task,
            'status': 'in_progress'
        }
        
        sessions.append(new_session)
        manager.set_metadata('work_sessions', sessions)
        
        # Set current session
        manager.set_metadata('current_session', session_id)
        
        return f"Started {session_type} session: {session_id}"
    else:
        # Find in-progress session
        current_id = manager.get_metadata('current_session')
        if not current_id:
            return "No active session found"
            
        for session in sessions:
            if session.get('id') == current_id:
                # Update session
                session['end_time'] = timestamp
                session['status'] = 'completed'
                session['result'] = result or 'Completed'
                session['duration_minutes'] = calculate_duration_minutes(
                    session.get('start_time'), timestamp
                )
                
                # Update sessions
                manager.set_metadata('work_sessions', sessions)
                
                # Clear current session
                manager.set_metadata('current_session', None)
                
                return f"Completed {session.get('type')} session: {session.get('id')}"
                
        return "Error: Active session not found in session history"


def calculate_duration_minutes(start_time: str, end_time: str) -> float:
    """Calculate duration in minutes between two ISO timestamps."""
    try:
        start = datetime.datetime.fromisoformat(start_time)
        end = datetime.datetime.fromisoformat(end_time)
        return (end - start).total_seconds() / 60.0
    except Exception:
        return 0.0


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
                    op.get('priority_min')
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
                    op.get('semantic', False),
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
                    op.get('use_semantic', False)
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
        
        # Check if semantic index is available
        semantic_status = {
            "available": semantic_index.available,
            "initialized": semantic_index.initialized if semantic_index.available else False,
            "document_count": len(semantic_index.doc_ids) if semantic_index.initialized else 0
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
            "semantic_search": semantic_status,
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
        description="Enhanced Memory System for AI operations"
    )
    
    parser.add_argument(
        "--db-path",
        help="Path to the state database (defaults to in-memory)",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Auto-load command - automatically load docs on startup
    auto_load_parser = subparsers.add_parser("auto-load", help="Automatically load documentation")
    
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
    
    # Remind command (alias for recall --last)
    remind_parser = subparsers.add_parser("remind", help="Show a reminder of what I'm currently doing")
    
    # Think command
    think_parser = subparsers.add_parser("think", help="Record a thought for future reference")
    think_parser.add_argument("thought", help="Thought content")
    
    # Note command
    note_parser = subparsers.add_parser("note", help="Take a quick note about something")
    note_parser.add_argument("topic", help="Note topic")
    note_parser.add_argument("content", help="Note content")
    
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
    find_docs_parser.add_argument("--semantic", action="store_true", help="Use semantic search")
    find_docs_parser.add_argument(
        "--max-results",
        "-m",
        type=int,
        default=10,
        help="Maximum results to return"
    )
    
    load_docs_parser = subparsers.add_parser("load-docs", help="Load project documentation")
    load_docs_parser.add_argument("--force", "-f", action="store_true", help="Force reload existing docs")
    
    # Verification commands
    verify_parser = subparsers.add_parser("verify-summary", help="Get verification summary")
    verify_parser.add_argument("--days", "-d", type=int, help="Only include verifications from the last X days")
    
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
    recovery_parser.add_argument("--semantic", "-s", action="store_true", help="Use semantic matching")
    
    # Context management commands
    staleness_parser = subparsers.add_parser("check-staleness", help="Check if current context is stale")
    
    # Work session tracking commands
    session_parser = subparsers.add_parser("session", help="Track work session")
    session_parser.add_argument("action", choices=["start", "end"], help="Start or end a session")
    session_parser.add_argument("type", help="Type of work session")
    session_parser.add_argument("--task", "-t", help="Task description (required for start)")
    session_parser.add_argument("--result", "-r", help="Session result (for end)")
    
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
    
    if args.command == "auto-load":
        result = set_up_automatic_loading(args.db_path)
        
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
            args.priority_min
        )
        
    elif args.command == "remind":
        result = recall(last=True)
        
    elif args.command == "think":
        result = think(args.thought)
        
    elif args.command == "note":
        result = note(args.topic, args.content)
        
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
            args.semantic, 
            args.max_results
        )
        
    elif args.command == "load-docs":
        result = load_project_docs(args.force)
        
    elif args.command == "verify-summary":
        result = get_verification_summary(args.days)
        
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
            args.semantic
        )
        
    elif args.command == "check-staleness":
        result = detect_context_staleness()
        
    elif args.command == "session":
        result = track_work_session(
            args.type,
            args.action == "start",
            args.task,
            args.result
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


# Alias functions for better AI context management

def think(thought):
    """Record a thought for future reference."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return save_docs(
        f"thought-{timestamp}", 
        thought,
        summary=thought[:100] + ('...' if len(thought) > 100 else ''),
        tags=["thought"],
        importance=6
    )


def remind_me():
    """Get a reminder of what I'm currently doing."""
    context = recall(last=True)
    
    # Check for staleness
    staleness = detect_context_staleness()
    
    if isinstance(context, dict) and context.get('task'):
        result = {
            "task": context.get('task'),
            "goal": context.get('goal'),
            "progress": context.get('progress'),
            "next_steps": context.get('next_steps'),
            "staleness": staleness
        }
        return result
    else:
        return {"error": "No active context found", "staleness": staleness}


def note(topic, content):
    """Take a quick note about something."""
    return save_docs(
        topic, 
        content,
        summary=content[:100] + ('...' if len(content) > 100 else ''),
        tags=["note"],
        importance=5
    )


def recall_thought(search=None):
    """Recall a previous thought."""
    if search:
        return find_docs(search=search, tag="thought")
    else:
        return find_docs(tag="thought")


if __name__ == "__main__":
    main()