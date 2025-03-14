#!/usr/bin/env python3
"""
Debug script for the Agent Memory System and Knowledge Correction

This script provides a way to test the memory system and knowledge correction
functionality in isolation to verify that they are working correctly.

Documentation references:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- Agent Memory System: See agent_memory.py
- Knowledge Correction: See knowledge_correction.py
"""

import sys
import logging
from loguru import logger
import argparse

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem
from agent_tools.cursor_rules.core.memory.knowledge_correction import (
    update_knowledge, 
    find_similar_facts,
    get_fact_history,
    analyze_contradictions,
    resolve_contradictions,
    merge_facts
)

def configure_logging(verbose=False):
    """Configure logging for the debug script."""
    logger.remove()  # Remove default handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    
    # If verbose, also show debug logs from dependencies
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

def test_embedding_generation(memory_system):
    """Test that embedding generation works correctly."""
    print("\n=== Testing Embedding Generation ===")
    test_text = "This is a test sentence for embedding generation."
    
    embedding_data = memory_system._get_embedding(test_text)
    
    if embedding_data and 'embedding' in embedding_data:
        print("✅ Successfully generated embedding")
        print(f"  Type: {type(embedding_data['embedding'])}")
        print(f"  Length: {len(embedding_data['embedding'])}")
        from agent_tools.cursor_rules.utils.vector_utils import get_vector_stats
        stats = get_vector_stats(embedding_data['embedding'])
        print(f"  Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, norm={stats['norm']:.3f}")
        return True
    else:
        print("❌ Failed to generate embedding")
        print(f"  Returned data: {embedding_data}")
        return False

def test_memory_storage(memory_system):
    """Test that basic memory storage works."""
    print("\n=== Testing Memory Storage ===")
    result = memory_system.remember(
        content="The Earth orbits the Sun in approximately 365 days",
        confidence=0.9,
        domains=["astronomy", "science"],
        importance=0.8
    )
    
    if result and 'new' in result and 'fact_id' in result['new']:
        print(f"✅ Successfully stored fact with ID: {result['new']['fact_id']}")
        return result['new']['fact_id']
    else:
        print("❌ Failed to store fact")
        print(f"  Result: {result}")
        return None

def test_similar_facts(memory_system, fact_id):
    """Test finding similar facts."""
    print("\n=== Testing Similar Facts Search ===")
    
    # First, get the original fact
    fact = memory_system.get_fact(fact_id)
    if not fact:
        print(f"❌ Could not retrieve fact with ID: {fact_id}")
        return False
        
    print(f"Original fact: '{fact['content']}'")
    
    # Now search for similar facts with slightly different wording
    similar_facts = find_similar_facts(
        memory_system,
        "The Earth takes about 365 days to orbit around the Sun",
        similarity_threshold=0.7,
        limit=5
    )
    
    if similar_facts:
        print(f"✅ Found {len(similar_facts)} similar facts")
        for i, fact in enumerate(similar_facts):
            print(f"  [{i+1}] '{fact['content']}' (score: {fact.get('similarity_score', 0):.3f})")
        return True
    else:
        print("❌ No similar facts found")
        # Try with lower threshold
        print("Trying with lower threshold...")
        similar_facts = find_similar_facts(
            memory_system,
            "The Earth takes about 365 days to orbit around the Sun",
            similarity_threshold=0.5,
            limit=5
        )
        if similar_facts:
            print(f"✅ Found {len(similar_facts)} similar facts with lower threshold")
            for i, fact in enumerate(similar_facts):
                print(f"  [{i+1}] '{fact['content']}' (score: {fact.get('similarity_score', 0):.3f})")
            return True
        return False

def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description='Debug agent memory system')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--host', default='http://localhost:8529', help='ArangoDB host')
    parser.add_argument('--username', default='root', help='ArangoDB username')
    parser.add_argument('--password', default='openSesame', help='ArangoDB password')
    parser.add_argument('--db_name', default='agent_memory_db', help='Database name')
    args = parser.parse_args()
    
    configure_logging(args.verbose)
    
    print("=== Agent Memory System Debug ===")
    print(f"Connecting to ArangoDB at {args.host}")
    
    # Initialize memory system
    memory = AgentMemorySystem({
        "arango_host": args.host,
        "username": args.username,
        "password": args.password,
        "db_name": args.db_name
    })
    
    print("Initializing memory system...")
    success = memory.initialize()
    
    if not success:
        print("❌ Failed to initialize memory system")
        return
    
    print("✅ Memory system initialized")
    
    # Run tests
    embedding_ok = test_embedding_generation(memory)
    if not embedding_ok:
        print("\n⚠️ Embedding generation failed - this will affect other tests")
    
    fact_id = test_memory_storage(memory)
    if fact_id:
        similar_ok = test_similar_facts(memory, fact_id)
        if not similar_ok:
            print("\n⚠️ Similar facts search failed")
    else:
        print("\n⚠️ Memory storage failed - cannot continue with similar facts test")
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    main()
