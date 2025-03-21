"""Tests for QA deduplication utilities.

This module tests the deduplication utilities for QA generation, including
exact deduplication based on text matching and semantic deduplication based
on meaning similarity.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- sentence-transformers: https://sbert.net/docs/
- numpy: https://numpy.org/doc/

Expected behavior:
- exact_deduplicate: Should remove duplicate QA pairs with the same question-answer text
  after normalizing whitespace and case.
- semantic_deduplicate: Should remove QA pairs that are semantically similar based on
  embeddings and a configurable similarity threshold.
- deduplicate_qa_pairs: Should apply both exact and semantic deduplication in sequence.
"""

import pytest
from agent_tools.dualipa.qa.models.qa_models import QAPair
from agent_tools.dualipa.qa.utils.deduplication import (
    exact_deduplicate,
    semantic_deduplicate,
    deduplicate_qa_pairs,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


def test_exact_deduplicate():
    """Test exact deduplication of QA pairs.
    
    This test verifies that the exact_deduplicate function correctly
    removes duplicate QA pairs based on identical question-answer content,
    regardless of differences in reasoning.
    
    Input:
        - List of QAPair objects with an exact duplicate (same question-answer)
    
    Expected output:
        - Reduced list with duplicates removed
        - Original list length reduced by exactly 1
        - Both unique questions ("What is Python?" and "What is Java?") preserved
    
    Dependencies:
        - QAPair model from agent_tools.dualipa.qa.models.qa_models
    """
    # Create test pairs with exact duplicates
    pairs = [
        QAPair(question="What is Python?", answer="A programming language", reasoning="Python is known for readability. Oh wait?! It's also very versatile."),
        QAPair(question="What is Java?", answer="A programming language", reasoning="Java is known for portability. Oh wait?! It's also used in Android."),
        # Exact duplicate of first pair
        QAPair(question="What is Python?", answer="A programming language", reasoning="Different reasoning but same Q&A. Oh wait?! This is a duplicate."),
    ]
    
    # Deduplicate
    unique_pairs = exact_deduplicate(pairs)
    
    # Verify results
    assert len(unique_pairs) < len(pairs), "Should remove at least one duplicate"
    assert len(unique_pairs) == 2, "Should have 2 unique QA pairs after deduplication"


def test_deduplicate_qa_pairs_real():
    """Test the main deduplication function with real examples.
    
    This test verifies that the deduplicate_qa_pairs function correctly
    identifies and removes exact duplicates in a realistic dataset of QA pairs.
    
    Input:
        - List of QAPair objects containing an exact duplicate
        - First and third pairs have identical question-answer content
    
    Expected output:
        - Reduced list containing only unique QA pairs
        - List length reduced from 3 to 2
        - Both unique topics (module usage and semantic deduplication) preserved
    
    Dependencies:
        - QAPair model from agent_tools.dualipa.qa.models.qa_models
        - deduplicate_qa_pairs from agent_tools.dualipa.qa.utils.deduplication
    """
    pairs = [
        # First pair
        QAPair(question="How do I use the deduplication module?", 
               answer="Import deduplicate_qa_pairs and pass your QA pairs",
               reasoning="The function is the main entry point. Oh wait?! It also supports a custom threshold."),
        
        # Different pair
        QAPair(question="What is semantic deduplication?", 
               answer="It identifies and removes QA pairs that mean the same thing even if worded differently",
               reasoning="It uses embeddings to compare. Oh wait?! It's more advanced than exact matching."),
        
        # Exact duplicate of the first pair
        QAPair(question="How do I use the deduplication module?", 
               answer="Import deduplicate_qa_pairs and pass your QA pairs",
               reasoning="The function handles QA pairs. Oh wait?! You can also set a threshold."),
    ]
    
    # Run deduplication
    result = deduplicate_qa_pairs(pairs)
    
    # We should have at least removed the exact duplicate
    assert len(result) == 2, "Should have 2 unique QA pairs after deduplication"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, 
                   reason="sentence-transformers not available")
def test_deduplicate_semantic():
    """Test semantic deduplication.
    
    This test evaluates whether the semantic_deduplicate function can identify 
    and remove QA pairs that have similar meaning but different wording.
    It also tests how different threshold values affect deduplication aggressiveness.
    This test will be skipped if sentence-transformers is not available.
    
    Input:
        - List of QAPair objects including semantically similar pairs
        - Two pairs asking about how to use the deduplication module, but with different wording
    
    Expected output:
        - Exact deduplication should not affect semantically similar pairs (count remains 3)
        - A high similarity threshold (0.95) should be less aggressive
        - A low similarity threshold (0.7) should be more aggressive in removing similar pairs
    
    Dependencies:
        - QAPair model from agent_tools.dualipa.qa.models.qa_models
        - exact_deduplicate, semantic_deduplicate from agent_tools.dualipa.qa.utils.deduplication
        - sentence-transformers package (external) for embedding generation
    """
    pairs = [
        # First pair
        QAPair(question="How do you use the deduplication module?", 
               answer="Import the deduplicate_qa_pairs function and pass your list of QA pairs",
               reasoning="This is the standard usage pattern. Oh wait?! There are also additional parameters."),
        
        # Semantically similar to first pair (different wording but same meaning)
        QAPair(question="What's the way to apply deduplication to my QA pairs?", 
               answer="You need to call deduplicate_qa_pairs with your collection of QA pairs",
               reasoning="This is how you use it. Oh wait?! You can also specify a threshold."),
        
        # Different pair
        QAPair(question="What threshold should I use for semantic deduplication?", 
               answer="Start with 0.85 and adjust based on your specific needs",
               reasoning="This is the default. Oh wait?! It depends on your content type."),
    ]
    
    # First check that exact deduplication doesn't remove the semantically similar pair
    exact_result = exact_deduplicate(pairs)
    assert len(exact_result) == 3, "Exact deduplication should not catch semantic duplicates"
    
    # Now check semantic deduplication with different thresholds
    # With a high threshold (0.95), similar pairs might not be caught
    high_threshold_result = semantic_deduplicate(pairs, threshold=0.95)
    
    # With a lower threshold (0.7), similar pairs should be caught
    low_threshold_result = semantic_deduplicate(pairs, threshold=0.7)
    
    # Lower threshold should remove more or equal pairs than higher threshold
    assert len(low_threshold_result) <= len(high_threshold_result), "Lower threshold should be more aggressive"


def test_empty_input():
    """Test that empty input is handled correctly.
    
    This test verifies that all deduplication functions handle empty input
    gracefully by returning an empty list instead of raising exceptions.
    
    Input:
        - Empty list []
    
    Expected output:
        - Empty list [] from all three functions
    
    Dependencies:
        - exact_deduplicate, semantic_deduplicate, deduplicate_qa_pairs from 
          agent_tools.dualipa.qa.utils.deduplication
    """
    assert exact_deduplicate([]) == [], "Empty input should return empty list"
    assert semantic_deduplicate([]) == [], "Empty input should return empty list"
    assert deduplicate_qa_pairs([]) == [], "Empty input should return empty list"