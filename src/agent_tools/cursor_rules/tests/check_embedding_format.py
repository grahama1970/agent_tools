#!/usr/bin/env python3
"""
Check Embedding Format

This script checks how glossary terms and definitions are formatted for embedding.
"""

from agent_tools.cursor_rules.core.glossary import format_for_embedding
from agent_tools.cursor_rules.embedding.embedding_utils import ensure_text_has_prefix
from agent_tools.cursor_rules.core.cursor_rules import generate_embedding

def main():
    """Check the embedding format for glossary terms."""
    # Sample terms and definitions
    samples = [
        {
            "term": "Neural Network",
            "definition": "A computing system inspired by biological neurons that can learn from data"
        },
        {
            "term": "Semantic Search",
            "definition": "A search technique that considers the meaning and context of search terms"
        },
        {
            "term": "Vector Database",
            "definition": "A database designed to store and query high-dimensional vectors"
        }
    ]
    
    print("\n=== EMBEDDING FORMAT CHECK ===\n")
    
    # Check format_for_embedding
    print("1. format_for_embedding output:")
    for sample in samples:
        formatted = format_for_embedding(sample["term"], sample["definition"])
        print(f"   - {formatted}")
    
    print("\n2. After ensure_text_has_prefix:")
    for sample in samples:
        formatted = format_for_embedding(sample["term"], sample["definition"])
        prefixed = ensure_text_has_prefix(formatted)
        print(f"   - {prefixed}")
    
    print("\n3. Complete embedding process:")
    sample = samples[0]  # Just use the first sample
    formatted = format_for_embedding(sample["term"], sample["definition"])
    prefixed = ensure_text_has_prefix(formatted)
    
    print(f"   Original term: {sample['term']}")
    print(f"   Original definition: {sample['definition']}")
    print(f"   After format_for_embedding: {formatted}")
    print(f"   After ensure_text_has_prefix: {prefixed}")
    
    # Try to generate an actual embedding
    try:
        embedding_result = generate_embedding(formatted)
        embedding_length = len(embedding_result.get("embedding", []))
        print(f"   Embedding generated: {'Yes' if embedding_length > 0 else 'No'}")
        print(f"   Embedding length: {embedding_length}")
        print(f"   Embedding metadata: {embedding_result.get('metadata', {})}")
    except Exception as e:
        print(f"   Error generating embedding: {e}")
    
    print("\n=== EMBEDDING FORMAT ANALYSIS ===\n")
    print("Current format: [TERM: term] definition")
    print("Prefixed format: search_document: [TERM: term] definition")
    print("\nThis format combines the term and definition, which allows the embedding to capture")
    print("the relationship between them. When a user's question is semantically similar to this")
    print("combined representation, the system can identify relevant glossary entries.")

if __name__ == "__main__":
    main() 