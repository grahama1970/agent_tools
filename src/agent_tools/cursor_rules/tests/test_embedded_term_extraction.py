#!/usr/bin/env python3
"""
Test Embedded Term Extraction

This test validates that terms embedded in user questions in the format
[TERM: term] are properly extracted and matched with definitions.
"""

import asyncio
import re
from typing import List, Dict, Any
from tabulate import tabulate

# Simple regex pattern for extracting embedded terms
TERM_PATTERN = r'\[TERM:\s*([^\]|]+)(?:\s*\|\s*([^\]]+))?\]'

def extract_embedded_terms(text: str) -> List[Dict[str, str]]:
    """
    Extract terms embedded in the text in the format [TERM: term] or [TERM: term | definition].
    
    Args:
        text: Input text containing embedded terms
        
    Returns:
        List of dictionaries with extracted terms and optional definitions
    """
    matches = re.finditer(TERM_PATTERN, text, re.IGNORECASE)
    extracted = []
    
    for match in matches:
        term = match.group(1).strip()
        definition = match.group(2).strip() if match.group(2) else None
        
        if term:
            extracted.append({
                "term": term,
                "definition": definition,
                "original_match": match.group(0)
            })
    
    return extracted

def find_matching_glossary_entries(terms: List[Dict[str, str]], glossary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find matching glossary entries for the extracted terms.
    
    Args:
        terms: List of extracted terms
        glossary: List of glossary entries
        
    Returns:
        List of matching glossary entries
    """
    results = []
    
    for term_info in terms:
        term = term_info["term"].lower()
        
        # Find exact and fuzzy matches
        for entry in glossary:
            glossary_term = entry["term"].lower()
            
            # Check for exact match or if glossary term contains the search term
            if term == glossary_term or term in glossary_term or glossary_term in term:
                results.append({
                    "extracted_term": term_info["term"],
                    "matched_term": entry["term"],
                    "definition": entry["definition"],
                    "match_type": "exact" if term == glossary_term else "partial",
                    "category": entry.get("category", "")
                })
    
    return results

# Sample glossary for testing
SAMPLE_GLOSSARY = [
    {
        "term": "Neural Network",
        "definition": "A computing system inspired by biological neurons that can learn from data",
        "category": "machine learning"
    },
    {
        "term": "Vector Database",
        "definition": "A database designed to store and query high-dimensional vectors",
        "category": "database"
    },
    {
        "term": "Semantic Search",
        "definition": "A search technique that considers meaning and context rather than just keywords",
        "category": "search"
    },
    {
        "term": "Embedding",
        "definition": "A technique that maps discrete objects like words to vectors of real numbers",
        "category": "machine learning"
    }
]

def run_term_extraction_tests():
    """Run tests for embedded term extraction."""
    test_cases = [
        {
            "name": "Simple embedded term",
            "input": "How does [TERM: Neural Network] work for image recognition?",
            "expected_terms": ["Neural Network"]
        },
        {
            "name": "Multiple embedded terms",
            "input": "When using [TERM: Semantic Search] with a [TERM: Vector Database], what performance can I expect?",
            "expected_terms": ["Semantic Search", "Vector Database"]
        },
        {
            "name": "Embedded term with definition",
            "input": "Let me explain [TERM: Embedding | A technique to convert words to vectors] in simple terms.",
            "expected_terms": ["Embedding"]
        },
        {
            "name": "No embedded terms",
            "input": "What is the best way to implement a neural network?",
            "expected_terms": []
        },
        {
            "name": "Case insensitive matching",
            "input": "How does [term: Neural Network] compare to traditional algorithms?",
            "expected_terms": ["Neural Network"]
        }
    ]
    
    print("\n=== EMBEDDED TERM EXTRACTION TESTS ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        # Extract terms
        extracted = extract_embedded_terms(test_case["input"])
        extracted_terms = [item["term"] for item in extracted]
        
        # Check if the extracted terms match the expected terms
        expected_terms = test_case["expected_terms"]
        missing_terms = [term for term in expected_terms if term not in extracted_terms]
        extra_terms = [term for term in extracted_terms if term not in expected_terms]
        
        if not missing_terms and not extra_terms:
            print("✅ SUCCESS: All expected terms were correctly extracted")
        else:
            if missing_terms:
                print(f"❌ ERROR: Missing expected terms: {missing_terms}")
            if extra_terms:
                print(f"❌ ERROR: Extracted unexpected terms: {extra_terms}")
        
        # Print the extracted terms
        if extracted:
            print("\nExtracted terms:")
            for item in extracted:
                if item["definition"]:
                    print(f"  - {item['term']} (with definition: {item['definition']})")
                else:
                    print(f"  - {item['term']}")
        else:
            print("  No terms extracted")
        
        # If terms were extracted, find matching glossary entries
        if extracted:
            matches = find_matching_glossary_entries(extracted, SAMPLE_GLOSSARY)
            
            if matches:
                print("\nMatching glossary entries:")
                table_data = []
                for match in matches:
                    table_data.append([
                        match["extracted_term"],
                        match["matched_term"],
                        match["definition"][:50] + "..." if len(match["definition"]) > 50 else match["definition"],
                        match["match_type"],
                        match["category"]
                    ])
                
                print(tabulate(
                    table_data,
                    headers=["Extracted Term", "Matched Term", "Definition", "Match Type", "Category"],
                    tablefmt="grid"
                ))
            else:
                print("  No matching glossary entries found")
        
        print("-" * 80)
    
    print("\n=== TEST SUMMARY ===")
    print(f"Total tests: {len(test_cases)}")

if __name__ == "__main__":
    run_term_extraction_tests() 