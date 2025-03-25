"""
Example demonstrating integration with QA systems.

This example shows how to format extraction output for QA systems.
"""

import os
import json
from pathlib import Path

# Import integration components
from agent_tools.dualipa.extraction.integration import QAIntegration, QuestionGenerator

def format_for_qa_system(extraction_file: str, output_dir: str):
    """
    Format extraction output for QA system consumption.
    
    Args:
        extraction_file: Path to extraction output JSON file
        output_dir: Directory to save QA-formatted output
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load extraction output
    print(f"Loading extraction output from {extraction_file}...")
    with open(extraction_file, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    
    # Format for QA system
    print("Formatting for QA system...")
    qa_integration = QAIntegration()
    qa_data = qa_integration.format_for_qa(blocks)
    
    # Save QA-formatted output
    qa_output_file = os.path.join(output_dir, "qa_formatted_output.json")
    with open(qa_output_file, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2)
    
    print(f"QA formatting completed. Output saved to {qa_output_file}")
    
    # Generate sample questions
    print("Generating sample questions...")
    question_generator = QuestionGenerator()
    questions = question_generator.generate_questions(blocks, num_questions=5)
    
    # Save sample questions
    questions_file = os.path.join(output_dir, "sample_questions.json")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated {len(questions)} sample questions. Saved to {questions_file}")
    
    return qa_data, questions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format extraction output for QA systems")
    parser.add_argument("extraction_file", help="Path to extraction output JSON file")
    parser.add_argument("--output-dir", default="qa_output", help="Output directory")
    
    args = parser.parse_args()
    
    format_for_qa_system(args.extraction_file, args.output_dir)
