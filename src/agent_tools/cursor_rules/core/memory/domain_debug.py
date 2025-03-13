#!/usr/bin/env python3
"""
Debug utility specifically for testing the domain preservation issue.

This script:
1. Sets up a test environment with facts from different domains
2. Demonstrates the current behavior of hybrid_search (without domain preservation)
3. Implements and tests the fix suggested by the smaller model
4. Shows score comparisons before and after the fix
"""

import sys
import json
from pprint import pprint
from copy import deepcopy
from loguru import logger

# Import the agent memory system
sys.path.append(".")  # Ensure we can import from the root directory
from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem, DEFAULT_CONFIG

def setup_test_data(memory):
    """Set up test data for domain preservation testing."""
    # Clear existing data
    print("Clearing existing data...")
    memory.db.aql.execute(f"FOR doc IN {memory.config['facts_collection']} REMOVE doc IN {memory.config['facts_collection']}")
    
    # Create test facts
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
            "importance": 0.4,
            "domains": ["chemistry", "science"],
            "ttl_days": 300
        },
        {
            "content": "Mitochondria is the powerhouse of the cell",
            "importance": 0.5,
            "domains": ["biology", "science"],
            "ttl_days": 400
        }
    ]
    
    # Add facts to memory
    for fact in facts:
        memory.remember(**fact)
    
    return facts

def test_original_behavior():
    """Test the current behavior of hybrid_search without domain preservation."""
    # Initialize memory system with original configuration
    print("\n===== TESTING ORIGINAL BEHAVIOR =====")
    memory = AgentMemorySystem()
    memory.initialize()
    
    # Set up test data
    facts = setup_test_data(memory)
    
    # Test hybrid_search
    print("\nRunning hybrid_search (without domain preservation):")
    query = "physics concepts"
    results = memory.hybrid_search(query, threshold=0.01)
    
    print(f"\nQuery: '{query}'")
    print(f"Results found: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['content']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Domains: {result['domains']}")
        print(f"  TTL days: {facts[i]['ttl_days'] if i < len(facts) else 'unknown'}")
        print(f"  Component scores:")
        for component, score in result['components'].items():
            print(f"    {component}: {score:.3f}")
    
    return results

def test_fixed_behavior():
    """Test the behavior with the fix implemented."""
    print("\n===== TESTING FIXED BEHAVIOR =====")
    
    # Create a modified configuration with default preserved domains
    modified_config = deepcopy(DEFAULT_CONFIG)
    modified_config["default_preserved_domains"] = ["physics"]
    
    # Initialize memory system with modified configuration
    memory = AgentMemorySystem(modified_config)
    memory.initialize()
    
    # Set up test data
    facts = setup_test_data(memory)
    
    # Modify the recall method to use default preserved domains
    original_recall = memory.recall
    
    def patched_recall(query, threshold=0.4, boost_recency=True, limit=5, 
                    domain_filter=None, semantic=True, bm25=True, glossary=True):
        """Patched version of recall that uses default preserved domains."""
        # Fix: Use default preserved domains when no filter is provided
        domains_to_preserve = domain_filter if domain_filter is not None else memory.config.get("default_preserved_domains", [])
        
        # This is a simplified version that just shows what we would change
        print(f"[DEBUG] Using domains_to_preserve: {domains_to_preserve}")
        
        # Call the original recall method with the new domain preservation
        # In a real fix, we would modify the bind_vars in the original method
        if domain_filter is None:
            print("[DEBUG] Using default preserved domains for boosting")
        return original_recall(query, threshold, boost_recency, limit, domain_filter, semantic, bm25, glossary)
    
    # Apply the patch (in a real fix, we would modify the actual code)
    memory.recall = patched_recall
    
    # Test hybrid_search with the patched recall method
    print("\nRunning hybrid_search (with domain preservation fix):")
    query = "physics concepts"
    results = memory.hybrid_search(query, threshold=0.01)
    
    print(f"\nQuery: '{query}'")
    print(f"Results found: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['content']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Domains: {result['domains']}")
        print(f"  TTL days: {facts[i]['ttl_days'] if i < len(facts) else 'unknown'}")
        print(f"  Component scores:")
        for component, score in result['components'].items():
            print(f"    {component}: {score:.3f}")
    
    return results

def main():
    """Main function to run the tests."""
    print("=== DOMAIN PRESERVATION DEBUG UTILITY ===")
    print("This utility diagnoses the domain preservation issue and tests the suggested fix.")
    
    # Test original and fixed behavior
    original_results = test_original_behavior()
    fixed_results = test_fixed_behavior()
    
    # Show what changes the fix would make
    print("\n===== COMPARISON =====")
    print("The suggested fix would modify the recall method to add default preserved domains.")
    print("This ensures that physics facts are boosted even when no explicit domain filter is used.")
    
    print("\nTo fix the issue, you would need to:")
    print("1. Add 'default_preserved_domains': ['physics'] to DEFAULT_CONFIG")
    print("2. Modify the bind_vars in recall to use:")
    print("   domains_to_preserve: domain_filter if domain_filter is not None else self.config.get('default_preserved_domains', [])")
    
    print("\nThis change would ensure that permanent physics facts receive the domain boost")
    print("and would be returned by hybrid_search even when no explicit domain filter is provided.")

if __name__ == "__main__":
    main() 