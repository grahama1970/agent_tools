"""Quality assurance validators for question-answer pairs.

This module provides utilities to validate and enhance question-answer pairs
using similarity matching, content validation, and filtering techniques.

Official documentation:
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
- Pydantic: https://docs.pydantic.dev/
"""

import re
import sys
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path
import json
import asyncio

import random
from pydantic import BaseModel, Field, validator
from rapidfuzz import fuzz, process
from loguru import logger

# Constants for QA validation
MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 15
MIN_SIMILARITY_THRESHOLD = 80.0  # Minimum similarity threshold for deduplication (0-100)
MIN_CODE_COMPLETENESS_THRESHOLD = 80  # For ensuring code completeness (0-100 scale)
MIN_FUNCTION_NAME_SIMILARITY = 85  # For function name matching (0-100 scale)
MIN_FUNCTION_CODE_COMPLETENESS = 0.7  # Minimum fraction of function code required in answers

# Strings that indicate questions about implementation
IMPLEMENTATION_INDICATORS = [
    "implementation", "how is", "how does", "code for", "source code",
    "how would you implement", "how to implement", "show me", "example of"
]

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


class QAValidationResult(BaseModel):
    """Result of QA pair validation with quality metrics."""
    valid: bool = Field(False, description="Whether the QA pair is valid")
    confidence: float = Field(0.0, description="Confidence score for the validation (0-1)")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    enhanced_question: Optional[str] = Field(None, description="Enhanced question if applicable")
    enhanced_answer: Optional[str] = Field(None, description="Enhanced answer if applicable")
    
    def add_issue(self, issue: str) -> None:
        """Add an issue to the validation result."""
        self.issues.append(issue)
        self.valid = False


def detect_duplicate_pairs(
    qa_pairs: List[Dict[str, str]], 
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD
) -> List[Dict[str, str]]:
    """Detect and remove duplicate QA pairs based on similarity.
    
    Args:
        qa_pairs: List of question-answer pair dictionaries
        similarity_threshold: Similarity threshold for duplicate detection (0-100)
        
    Returns:
        List of unique QA pairs
    """
    if not qa_pairs:
        return []
    
    unique_pairs = []
    existing_questions = []
    existing_answers = []
    
    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        
        # Check for duplicate questions
        q_matches = process.extract(
            question, 
            existing_questions, 
            scorer=fuzz.partial_ratio, 
            limit=3
        )
        
        is_duplicate = any(score >= similarity_threshold for _, score in q_matches)
        
        # Check for duplicate answers if not already a duplicate
        if not is_duplicate and existing_answers:
            a_matches = process.extract(
                answer, 
                existing_answers, 
                scorer=fuzz.partial_ratio, 
                limit=3
            )
            is_duplicate = any(score >= similarity_threshold for _, score in a_matches)
        
        if not is_duplicate:
            unique_pairs.append(pair)
            existing_questions.append(question)
            existing_answers.append(answer)
    
    logger.debug(f"Removed {len(qa_pairs) - len(unique_pairs)} duplicate QA pairs")
    return unique_pairs


def validate_function_qa_pair(
    qa_pair: Dict[str, str], 
    original_code: str,
    function_name: Optional[str] = None
) -> QAValidationResult:
    """Validate and enhance QA pairs for function-related questions.
    
    Args:
        qa_pair: Question-answer pair dictionary
        original_code: Original code content for reference
        function_name: Optional function name to validate against
        
    Returns:
        Validation result with enhanced content if applicable
    """
    result = QAValidationResult()
    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")
    
    if not question or len(question) < MIN_QUESTION_LENGTH:
        result.add_issue("Question is too short or empty")
        return result
        
    if not answer or len(answer) < MIN_ANSWER_LENGTH:
        result.add_issue("Answer is too short or empty")
        return result
    
    # Check if this is a function-related question
    is_function_question = False
    detected_function_name = None
    
    # If a function name is provided, check for it in the question
    if function_name:
        if function_name.lower() in question.lower():
            is_function_question = True
            detected_function_name = function_name
    else:
        # Try to detect function names in the question
        for pattern in FUNCTION_PATTERNS:
            matches = re.findall(pattern, original_code)
            for match in matches:
                # Check if any detected function name appears in the question
                for fn_name in matches:
                    if fn_name.lower() in question.lower():
                        is_function_question = True
                        detected_function_name = fn_name
                        break
                
                if is_function_question:
                    break
    
    # Check for implementation-related questions
    is_implementation_question = any(indicator.lower() in question.lower() 
                                     for indicator in IMPLEMENTATION_INDICATORS)
    
    # Enhance function-related questions or implementation questions
    if (is_function_question or is_implementation_question) and detected_function_name:
        # Extract the complete function code
        fn_pattern = re.compile(
            r'(def\s+' + re.escape(detected_function_name) + r'\s*\(.*?(?:return|pass|raise|break).*?)(?=\n\S|\Z)', 
            re.DOTALL
        )
        js_fn_pattern = re.compile(
            r'(function\s+' + re.escape(detected_function_name) + r'\s*\(.*?})', 
            re.DOTALL
        )
        
        fn_match = fn_pattern.search(original_code) or js_fn_pattern.search(original_code)
        
        if fn_match:
            complete_function_code = fn_match.group(1).strip()
            
            # Check if the answer already contains the complete function code
            similarity = fuzz.partial_ratio(complete_function_code, answer)
            
            if similarity < MIN_CODE_COMPLETENESS_THRESHOLD:
                # Enhance the answer with the complete function code
                enhanced_answer = f"{answer}\n\nHere is the complete implementation:\n\n```\n{complete_function_code}\n```"
                result.enhanced_answer = enhanced_answer
                
            result.valid = True
            result.confidence = 0.9
        else:
            # Function name detected but couldn't extract code
            result.add_issue(f"Function '{detected_function_name}' mentioned but code extraction failed")
    else:
        # Not a function question or no function detected
        result.valid = True
        result.confidence = 0.7
    
    return result


@cached_retry(
    retries=3,
    cache_ttl=3600,  # 1 hour
    exceptions=(ConnectionError, TimeoutError)
)
async def validate_and_enhance_qa_pairs(
    qa_pairs: List[Dict[str, str]], 
    original_content: str,
    function_name: Optional[str] = None,
    deduplicate: bool = True
) -> List[Dict[str, str]]:
    """Validate and enhance a list of QA pairs using RapidFuzz.
    
    Args:
        qa_pairs: List of question-answer pairs to validate
        original_content: Original content (code or markdown) for reference
        function_name: Optional function name for targeted validation
        deduplicate: Whether to remove duplicate pairs
        
    Returns:
        List of validated and enhanced QA pairs
    """
    if not qa_pairs:
        return []
    
    # Use CPU-bound operations in separate thread following 005-async-patterns.mdc
    return await asyncio.to_thread(
        _validate_and_enhance_qa_pairs_sync,
        qa_pairs,
        original_content,
        function_name,
        deduplicate
    )

def _validate_and_enhance_qa_pairs_sync(
    qa_pairs: List[Dict[str, str]], 
    original_content: str,
    function_name: Optional[str] = None,
    deduplicate: bool = True
) -> List[Dict[str, str]]:
    """Synchronous implementation of validate_and_enhance_qa_pairs.
    
    This function runs in a separate thread via asyncio.to_thread.
    """
    # First, validate and clean the pairs
    validator = QAValidator()
    validated_pairs = []
    
    for pair in qa_pairs:
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        
        if not question or not answer:
            logger.warning("Skipping QA pair with empty question or answer")
            continue
            
        # Validate question and answer
        question_result = validator.validate_question(question)
        answer_result = validator.validate_answer(answer, question)
        
        # Only add if both are valid
        if question_result.valid and answer_result.valid:
            validated_pair = {
                "question": question,
                "answer": answer,
                "validation": {
                    "question": question_result.to_dict(),
                    "answer": answer_result.to_dict()
                }
            }
            
            # Additional function validation if requested
            if function_name and function_name.lower() in question.lower():
                function_result = validator.validate_function_answer(
                    answer, function_name, original_content
                )
                validated_pair["validation"]["function"] = function_result.to_dict()
            
            validated_pairs.append(validated_pair)
        else:
            logger.debug(f"Skipping invalid QA pair: {question[:30]}...")
    
    # Deduplicate if requested
    if deduplicate and RAPIDFUZZ_AVAILABLE:
        validated_pairs = detect_duplicate_pairs(validated_pairs, similarity_threshold=validator.similarity_threshold)
    
    # Extract just the question and answer fields for the final result
    return [{"question": pair["question"], "answer": pair["answer"]} for pair in validated_pairs]


async def demo_qa_validation() -> None:
    """Demonstrate the QA validation functionality with examples."""
    try:
        logger.info("Beginning QA Validation demonstration")
        
        # Example code for testing validation
        example_code = """
def calculate_average(numbers, weights=None):
    \"\"\"Calculate the average of a list of numbers with optional weights.\"\"\"
    if not numbers:
        return 0
    
    if weights:
        if len(weights) != len(numbers):
            raise ValueError("Weights and numbers must have the same length")
        return sum(n * w for n, w in zip(numbers, weights)) / sum(weights)
    
    return sum(numbers) / len(numbers)
"""
        
        # Example QA pairs
        example_pairs = [
            {
                "question": "What does the calculate_average function do?",
                "answer": "The calculate_average function computes the mean of a list of numbers, with optional weights."
            },
            {
                "question": "What happens if I pass an empty list to calculate_average?",
                "answer": "If you pass an empty list, the function returns 0."
            },
            {
                "question": "What error is raised if the weights and numbers have different lengths?",
                "answer": "A ValueError is raised with the message 'Weights and numbers must have the same length'."
            },
            {
                "question": "What does the calculate_average function do?", # Duplicate question
                "answer": "It calculates the average of a list of numbers."
            },
            {
                "question": "Show me the implementation of calculate_average",
                "answer": "Here's the code for calculate_average:"
            }
        ]
        
        # Demonstrate duplicate detection
        logger.info("Demonstrating duplicate QA pair detection")
        if RAPIDFUZZ_AVAILABLE:
            # Should identify and remove one duplicate
            deduplicated = detect_duplicate_pairs(example_pairs)
            logger.info(f"Removed {len(example_pairs) - len(deduplicated)} duplicate pair(s)")
        else:
            logger.warning("RapidFuzz not available, skipping duplicate detection demo")
        
        # Demonstrate function answer enhancement
        logger.info("Demonstrating function answer enhancement")
        enhanced_pairs = enhance_function_answers(example_pairs, example_code, "calculate_average")
        for pair in enhanced_pairs:
            if "Show me the implementation" in pair["question"]:
                logger.info(f"Enhanced answer with implementation:\n{pair['answer'][:100]}...")
        
        # Demonstrate full validation
        logger.info("Demonstrating full QA validation")
        mixed_pairs = [
            {"question": "", "answer": "This should be rejected due to empty question"},
            {"question": "What does calculate_average do", "answer": "It calculates averages but doesn't handle empty lists properly"},
            {"question": "How do weights work in calculate_average?", "answer": "Weights multiply each number before averaging"}
        ]
        example_pairs.extend(mixed_pairs)
        
        # Should validate and enhance all pairs
        validated = await validate_and_enhance_qa_pairs(mixed_pairs, example_code, "calculate_average")
        logger.info(f"Validated {len(validated)} of {len(mixed_pairs)} pairs")
        
        # Print results
        for i, pair in enumerate(validated):
            logger.info(f"Validated pair {i+1}:")
            logger.info(f"  Q: {pair['question']}")
            logger.info(f"  A: {pair['answer'][:100]}...")
        
        logger.info("QA Validation demonstration completed")
    except Exception as e:
        logger.error(f"Error in QA validation demo: {e}")

def demo_qa_validation_sync() -> None:
    """Synchronous wrapper for the QA validation demo."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_qa_validation())

if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    asyncio.run(demo_qa_validation())
    
    # Example of processing a QA file
    if len(sys.argv) > 1:
        try:
            input_file = sys.argv[1]
            logger.info(f"Processing QA pairs from: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'qa_pairs' in data and 'original_content' in data:
                qa_pairs = data['qa_pairs']
                original_content = data['original_content']
                
                validated_pairs = validate_and_enhance_qa_pairs(
                    qa_pairs, 
                    original_content,
                    function_name=data.get('function_name')
                )
                
                output_file = input_file.replace('.json', '_validated.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({'qa_pairs': validated_pairs}, f, indent=2)
                
                logger.info(f"Validated {len(validated_pairs)} out of {len(qa_pairs)} pairs")
                logger.info(f"Results saved to: {output_file}")
            else:
                logger.error("Input file must contain 'qa_pairs' and 'original_content' fields")
        except Exception as e:
            logger.error(f"Error processing file: {e}") 