#!/usr/bin/env python3
"""
Demonstration script for term embedding in queries.

This script demonstrates how terms are embedded in user queries using the
ModernBERT convention for glossary search integration.
"""

from agent_tools.cursor_rules.core.glossary import embed_term_in_query

def demonstrate_term_embedding():
    """Demonstrate different examples of term embedding in queries."""
    examples = [
        # Basic examples without definitions
        {
            "query": "Why does Microsoft Outlook suck so much?",
            "term": "Microsoft Outlook",
            "definition": None,
            "include_definition": False,
            "expected": "Why does [TERM: Microsoft Outlook] suck so much?"
        },
        {
            "query": "I need help with microsoft outlook settings",
            "term": "Microsoft Outlook",
            "definition": None,
            "include_definition": False,
            "expected": "I need help with [TERM: Microsoft Outlook] settings"
        },
        
        # Examples with definitions
        {
            "query": "Why does Microsoft Outlook suck so much?",
            "term": "Microsoft Outlook",
            "definition": "An email and calendar application developed by Microsoft",
            "include_definition": True,
            "expected": "Why does [TERM: Microsoft Outlook | An email and calendar application developed by Microsoft] suck so much?"
        },
        {
            "query": "How do I configure my email client?",
            "term": "Microsoft Outlook",
            "definition": "An email and calendar application developed by Microsoft",
            "include_definition": True,
            "expected": "How do I configure my email client? [TERM: Microsoft Outlook | An email and calendar application developed by Microsoft]"
        },
        
        # Multiple occurrences with definition
        {
            "query": "Can I connect Outlook to Gmail? Outlook seems complicated.",
            "term": "Outlook",
            "definition": "Microsoft's email and calendar application",
            "include_definition": True,
            "expected": "Can I connect [TERM: Outlook | Microsoft's email and calendar application] to Gmail? [TERM: Outlook | Microsoft's email and calendar application] seems complicated."
        }
    ]
    
    print("\nDEMONSTRATION OF TERM EMBEDDING IN QUERIES")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        query = example["query"]
        term = example["term"]
        definition = example["definition"]
        include_definition = example["include_definition"]
        expected = example["expected"]
        
        result = embed_term_in_query(query, term, definition=definition, include_definition=include_definition)
        
        print(f"\nExample {i}:")
        print(f"Original query: \"{query}\"")
        print(f"Term to embed:  \"{term}\"")
        if definition:
            print(f"Definition:     \"{definition}\"")
        print(f"Include def:    {include_definition}")
        print(f"Result:         \"{result}\"")
        print(f"Expected:       \"{expected}\"")
        print(f"Match:          {'✅' if result == expected else '❌'}")

if __name__ == "__main__":
    demonstrate_term_embedding() 