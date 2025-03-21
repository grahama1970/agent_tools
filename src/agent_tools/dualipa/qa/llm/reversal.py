"""Bidirectional QA generation functions.

This module provides functionality for generating reverse QA pairs.
It enables creating bidirectional knowledge by generating new questions
for existing answers, complementing the forward generation process.

Official documentation:
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
"""

import json
import asyncio
import logging
import textwrap
from typing import Dict, List, Any, Optional, Union, Tuple

from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
from agent_tools.dualipa.qa.models.config import QAGenerationConfig
from .retry_llm_call import retry_llm_call
from agent_tools.dualipa.qa.utils.validation import validate_qa_pair

logger = logging.getLogger(__name__)


async def generate_reversed_pair(
    original: QAPair,
    temperature: float = 0.7,
    max_retries: int = 2
) -> Optional[QAPair]:
    """Create a reverse Q&A pair from an original pair.
    
    This function takes an existing QA pair and generates a new question
    for the same answer, creating a "reverse" pair that approaches the
    knowledge from a different angle.
    
    Args:
        original: The original QA pair
        temperature: The temperature to use for generation (higher = more creative)
        max_retries: Maximum number of retries if generation fails
        
    Returns:
        A new QA pair with a different question but the same answer
    """
    # Create the reverse prompt using textwrap.dedent for clean formatting
    reverse_prompt = textwrap.dedent(f"""
        Given the following answer: 
        
        "{original.answer}"
        
        Generate a different question that would have this as its answer.
        Your question must be different from: "{original.question}"
        The question should explore the answer from a new angle.
        
        Include an "Oh wait?!" moment in your reasoning that shows your thought process.
        
        Format as a JSON object with keys:
        - question: The new question (must end with a question mark and be at least 10 characters)
        - reasoning: Your thought process with an "Oh wait?!" moment
    """).strip()
    
    # Prepare API call config
    config = {
        "model": "gpt-4-turbo",
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates insightful questions for given answers."},
            {"role": "user", "content": reverse_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Try generation with retries
    retries = 0
    while retries <= max_retries:
        try:
            # Make the API call
            response = await retry_llm_call(config)
            
            # Extract content from response
            content = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            data = json.loads(content)
            
            # Create the reversed pair
            reversed_pair = QAPair(
                question=data["question"],
                answer=original.answer,
                reasoning=data["reasoning"],
                source_section_uuid=original.source_section_uuid,
                temperature_used=temperature,
                direction=Direction.REVERSE,  # Explicitly use enum value
                complexity_level=original.complexity_level,
                confidence_score=original.confidence_score  # Preserve confidence
            )
            
            # Validate the pair
            if validate_qa_pair(reversed_pair):
                # Ensure question is different
                if reversed_pair.question != original.question:
                    logger.info(f"Successfully generated reversed QA pair: {reversed_pair.question}")
                    return reversed_pair
                else:
                    logger.warning("Generated question is identical to original, retrying...")
            else:
                logger.warning(f"Invalid reversed QA pair generated")
            
            # Increase temperature slightly for more variation in next attempt
            config["temperature"] = min(1.0, temperature + 0.1 * (retries + 1))
            retries += 1
            
        except Exception as e:
            logger.error(f"Error generating reversed QA pair: {e}")
            retries += 1
    
    logger.error(f"Failed to generate valid reversed QA pair after {max_retries} retries")
    return None


async def generate_reversed_qa_pairs(
    original_pairs: List[QAPair],
    reverse_ratio: float = 0.3,
    temperature: float = 0.7,
    max_concurrent_requests: int = 2
) -> List[QAPair]:
    """Generate reversed QA pairs for a subset of original pairs.
    
    This function selects a portion of the original QA pairs (based on
    the reverse_ratio) and generates new questions for their answers.
    It prioritizes pairs with higher confidence scores for reversal.
    
    Args:
        original_pairs: The original QA pairs
        reverse_ratio: Ratio of pairs to reverse (0.0 to 1.0)
        temperature: The temperature to use for generation
        max_concurrent_requests: Maximum number of concurrent generation requests
        
    Returns:
        List of reversed QA pairs
    """
    if not original_pairs:
        logger.warning("No original pairs provided for reversal")
        return []
    
    # Validate and adjust parameters
    reverse_ratio = max(0.0, min(1.0, reverse_ratio))  # Ensure in range [0.0, 1.0]
    
    # Filter out any pairs that are already reversed
    forward_pairs = [p for p in original_pairs if p.direction == Direction.FORWARD]
    if not forward_pairs:
        logger.warning("No forward pairs available for reversal")
        return []
    
    # Calculate how many pairs to reverse
    count = max(1, int(len(forward_pairs) * reverse_ratio))
    count = min(count, len(forward_pairs))  # Can't reverse more than we have
    
    logger.info(f"Generating reversed pairs for {count}/{len(forward_pairs)} original pairs (ratio: {reverse_ratio})")
    
    # Select pairs to reverse (prioritize higher confidence scores)
    pairs_to_reverse = sorted(
        forward_pairs,
        key=lambda p: p.confidence_score if p.confidence_score is not None else 0.5,
        reverse=True
    )[:count]
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def generate_with_rate_limit(pair: QAPair) -> Optional[QAPair]:
        """Run reversal with rate limiting applied."""
        async with semaphore:
            logger.debug(f"Generating reversed pair for: {pair.question}")
            return await generate_reversed_pair(pair, temperature)
    
    # Create tasks for each pair to reverse
    tasks = [generate_with_rate_limit(pair) for pair in pairs_to_reverse]
    
    # Execute all tasks with controlled concurrency
    results = await asyncio.gather(*tasks)
    
    # Filter out None values
    reversed_pairs = [pair for pair in results if pair is not None]
    
    logger.info(f"Successfully generated {len(reversed_pairs)} reversed QA pairs")
    return reversed_pairs


async def generate_bidirectional_qa_pairs(
    content: str,
    content_type: str,
    config: Optional[QAGenerationConfig] = None,
    section_uuid: Optional[str] = None
) -> Tuple[List[QAPair], List[QAPair]]:
    """Generate both forward and reverse QA pairs for content.
    
    This is a high-level function that:
    1. Generates forward QA pairs from content
    2. Selects high-quality forward pairs
    3. Generates reverse pairs from selected forward pairs
    
    Args:
        content: The content to generate QA pairs from
        content_type: Type of content (code, markdown, etc.)
        config: Configuration settings
        section_uuid: UUID of the content section
        
    Returns:
        Tuple of (forward_pairs, reverse_pairs)
    """
    if config is None:
        config = QAGenerationConfig()
    
    # Import here to avoid circular imports
    from .generation import generate_qa_pairs_with_temperature
    
    # Generate forward pairs
    forward_pairs = await generate_qa_pairs_with_temperature(
        content=content,
        content_type=content_type,
        temperature=config.temperature_range[0],  # Use lowest temperature for precision
        section_uuid=section_uuid,
        max_pairs=config.max_qa_pairs_per_section
    )
    
    # Generate reverse pairs from forward pairs
    reverse_pairs = await generate_reversed_qa_pairs(
        original_pairs=forward_pairs,
        reverse_ratio=config.bidirectional_ratio,
        temperature=config.temperature_range[-1],  # Use highest temperature for creativity
        max_concurrent_requests=config.max_concurrent_requests
    )
    
    return forward_pairs, reverse_pairs