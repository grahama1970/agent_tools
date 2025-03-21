"""Deduplication utilities for QA generation.

This module provides utilities for deduplicating QA pairs using both exact text
matching and semantic similarity. The module handles two types of deduplication:
1. Exact deduplication: Removes pairs with identical question-answer text (normalized)
2. Semantic deduplication: Removes pairs with similar meaning using embeddings

Official documentation:
- sentence-transformers: https://sbert.net/docs/
- numpy: https://numpy.org/doc/
- logging: https://docs.python.org/3/library/logging.html

Expected input/output:
- Input: List of QAPair objects that may contain duplicates
- Output: Filtered list of QAPair objects with duplicates removed
"""

import logging
import numpy as np
from typing import List, Tuple, Set
from agent_tools.dualipa.qa.models.qa_models import QAPair
from agent_tools.dualipa.qa.models.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, but have a fallback for MVP version
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available, using basic deduplication")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def exact_deduplicate(pairs: List[QAPair]) -> List[QAPair]:
    """Remove exact duplicate QA pairs.
    
    This function identifies and removes duplicate QA pairs based on identical
    question-answer content after normalizing case and whitespace. The function
    preserves the order of the first occurrence of each unique pair.
    
    Args:
        pairs: List of QA pairs that may contain duplicates
        
    Returns:
        Deduplicated list of QA pairs with exact duplicates removed
        
    Example:
        >>> pairs = [QAPair(question="What is X?", answer="Y"),
        ...          QAPair(question="what is x?", answer="y")]
        >>> result = exact_deduplicate(pairs)
        >>> len(result)
        1
    """
    # Use a set to track seen question-answer pairs
    seen = set()
    unique_pairs = []
    
    for pair in pairs:
        # Create a key for the question-answer pair
        qa_key = (pair.question.lower().strip(), pair.answer.lower().strip())
        
        if qa_key not in seen:
            seen.add(qa_key)
            unique_pairs.append(pair)
            
    return unique_pairs


def semantic_deduplicate(pairs: List[QAPair], threshold: float = SIMILARITY_THRESHOLD) -> List[QAPair]:
    """Remove semantically similar QA pairs using cosine similarity.
    
    This function identifies and removes semantically similar QA pairs based
    on their vector embeddings. It uses the sentence-transformers library to
    generate embeddings and compute cosine similarity between pairs. Pairs with
    similarity above the threshold are considered duplicates.
    
    Args:
        pairs: List of QA pairs (typically already processed by exact_deduplicate)
        threshold: Similarity threshold for deduplication (0.0-1.0, higher is more strict)
        
    Returns:
        Deduplicated list of QA pairs with semantically similar duplicates removed
        
    Notes:
        - Requires sentence-transformers library to be installed
        - Falls back to input list if sentence-transformers is not available
        - Uses the 'all-MiniLM-L6-v2' model for embeddings
        - Applies exact deduplication first
    """
    if not pairs:
        return []
        
    # First apply exact deduplication
    unique_pairs = exact_deduplicate(pairs)
    
    # If no sentence-transformers or only one pair, return
    if not SENTENCE_TRANSFORMERS_AVAILABLE or len(unique_pairs) <= 1:
        return unique_pairs
        
    try:
        # Load model (this is a placeholder - in a real implementation,
        # we would load the model once and reuse it)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for QA pairs
        texts = [f"{p.question} {p.answer}" for p in unique_pairs]
        embeddings = model.encode(texts)
        
        # Compute similarity matrix
        sim_matrix = np.inner(embeddings, embeddings)
        
        # Find unique pairs
        unique_indices = set()
        for i in range(len(unique_pairs)):
            if i not in unique_indices:
                for j in range(i+1, len(unique_pairs)):
                    if sim_matrix[i, j] > threshold:
                        unique_indices.add(j)
                        
        # Get unique pairs
        return [unique_pairs[i] for i in range(len(unique_pairs)) if i not in unique_indices]
    except Exception as e:
        logger.error(f"Error in semantic deduplication: {e}")
        return unique_pairs  # Fall back to exact deduplication


def deduplicate_qa_pairs(pairs: List[QAPair], threshold: float = SIMILARITY_THRESHOLD) -> List[QAPair]:
    """Deduplicate QA pairs using exact and semantic deduplication.
    
    This is the main entry point for deduplication, combining both exact and
    semantic deduplication in a single function. It first removes exact duplicates
    and then applies semantic deduplication if the sentence-transformers library
    is available.
    
    Args:
        pairs: List of QA pairs that may contain duplicates
        threshold: Similarity threshold for semantic deduplication (0.0-1.0)
        
    Returns:
        Deduplicated list of QA pairs with both exact and semantic duplicates removed
        
    Example:
        >>> pairs = [
        ...     QAPair(question="What is deduplication?", answer="Removing duplicates"),
        ...     QAPair(question="How do you deduplicate?", answer="By removing copies")
        ... ]
        >>> result = deduplicate_qa_pairs(pairs, threshold=0.8)
    """
    # First apply exact deduplication
    exact_unique = exact_deduplicate(pairs)
    
    # Then apply semantic deduplication if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return semantic_deduplicate(exact_unique, threshold)
    else:
        return exact_unique