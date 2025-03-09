#!/usr/bin/env python
"""
Cursor Rules Demo Script

This script demonstrates how to use the cursor rules database to solve a coding problem.
It searches for relevant rules and applies them to generate a solution.
"""
import sys
from agent_tools.cursor_rules.cursor_rules import setup_cursor_rules_db, semantic_search, hybrid_search, get_examples_for_rule

def find_relevant_rules(db, query, limit=3, use_hybrid=True):
    """Find rules relevant to the query."""
    print(f"Searching for rules related to: '{query}'")
    
    if use_hybrid:
        print("Using hybrid search (BM25 + vector similarity)")
        results = hybrid_search(db, query, limit=limit)
    else:
        print("Using semantic search")
        results = semantic_search(db, query, limit=limit)
    
    if not results:
        print("No relevant rules found.")
        return []
    
    print(f"Found {len(results)} relevant rules:")
    for i, result in enumerate(results, 1):
        rule = result["rule"]
        
        # Handle different score formats
        if "hybrid_score" in result:
            score = result["hybrid_score"]
            score_type = "Hybrid Score"
        elif "similarity" in result:
            score = result["similarity"]
            score_type = "Similarity"
        else:
            score = 0
            score_type = "Score"
            
        print(f"{i}. {rule['rule_number']}: {rule['title']} ({score_type}: {score:.4f})")
    
    return results

def get_rule_examples(db, rule_key):
    """Get examples for a rule."""
    examples = get_examples_for_rule(db, rule_key)
    return examples

def solve_problem(db, problem_description, use_hybrid=True):
    """Use rules to solve a coding problem."""
    print("\n=== PROBLEM ===")
    print(problem_description)
    print("\n=== FINDING RELEVANT RULES ===")
    
    # Find relevant rules
    results = find_relevant_rules(db, problem_description, use_hybrid=use_hybrid)
    if not results:
        print("Unable to find relevant rules for this problem.")
        return False
    
    # Get the most relevant rule
    top_result = results[0]
    rule = top_result["rule"]
    
    print("\n=== APPLYING RULE ===")
    print(f"Applying rule: {rule['rule_number']} - {rule['title']}")
    print(f"Rule description: {rule['description']}")
    
    # Get examples for the rule
    examples = get_rule_examples(db, rule["_key"])
    
    print("\n=== SOLUTION ===")
    if examples:
        print("Based on the examples for this rule, here's the solution:")
        example = examples[0]
        print(f"\nGood pattern to follow from '{example['title']}':")
        print("```python")
        print(example["good_example"])
        print("```")
        
        print("\nPattern to avoid:")
        print("```python")
        print(example["bad_example"])
        print("```")
    else:
        # Extract guidance from the rule content if no examples
        print("Based on the rule content, here's the solution guideline:")
        content_lines = rule["content"].split("\n")
        in_code_block = False
        code_blocks = []
        current_block = []
        
        for line in content_lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if not in_code_block and current_block:
                    code_blocks.append("\n".join(current_block))
                    current_block = []
            elif in_code_block:
                current_block.append(line)
        
        if code_blocks:
            print("```python")
            print(code_blocks[0])
            print("```")
        else:
            # Extract bullet points if no code blocks
            bullet_points = [line for line in content_lines if line.strip().startswith("- ")]
            if bullet_points:
                for point in bullet_points:
                    print(point)
    
    return True

def main():
    """Main function."""
    # Connect to database
    config = {
        "arango_config": {
            "hosts": ["http://localhost:8529"],
            "username": "root",
            "password": "openSesame"
        }
    }
    
    db_name = "cursor_rules_test"
    db = setup_cursor_rules_db(config, db_name=db_name)
    if not db:
        print("Failed to connect to database")
        return 1
    
    # Define problems
    problems = [
        {
            "description": "How should I handle asynchronous operations in my Python code?",
            "method": "hybrid"
        },
        {
            "description": "What's the best way to handle errors in my Python code?",
            "method": "hybrid"
        },
        {
            "description": "How should I work with databases in this project?",
            "method": "semantic" 
        }
    ]
    
    # Solve each problem
    for i, problem in enumerate(problems):
        if i > 0:
            print("\n" + "="*80 + "\n")
        
        use_hybrid = problem["method"] == "hybrid"
        solve_problem(db, problem["description"], use_hybrid=use_hybrid)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 