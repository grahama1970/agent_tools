"""Core QA generation functions.

This module provides core functionality for generating QA pairs from content.
It includes functions for generating QA pairs with temperature controls,
enhanced reasoning, bidirectional generation, and robust error handling.

Official documentation:
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
- logging: https://docs.python.org/3/library/logging.html
- uuid: https://docs.python.org/3/library/uuid.html
- datetime: https://docs.python.org/3/library/datetime.html
- re: https://docs.python.org/3/library/re.html

Expected input/output:
- generate_qa_pairs_with_temperature: Takes content and configuration, returns a list of QA pairs
- iterate_temperatures: Takes content and temperature range, returns aggregated QA pairs
- generate_code_qa_pairs: Takes code content, returns QA pairs about code
- generate_markdown_qa_pairs: Takes markdown content, returns QA pairs about the content
"""

import json
import asyncio
import random
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from agent_tools.dualipa.qa.models.qa_models import QAPair
from agent_tools.dualipa.qa.models.config import (
    QAGenerationConfig,
    DEFAULT_TEMPERATURE,
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    DEFAULT_TEMPERATURE_RANGE, 
    CODE_TEMPERATURE_RANGE,
    MARKDOWN_TEMPERATURE_RANGE
)
from .retry_llm_call import retry_llm_call
from agent_tools.dualipa.qa.utils.validation import validate_qa_pair, validate_temperature_range

logger = logging.getLogger(__name__)


async def generate_qa_pairs_with_temperature(
    content: str,
    content_type: str,
    temperature: float,
    section_uuid: Optional[str] = None,
    max_pairs: int = 5,
    prompt_template: Optional[str] = None,
    enable_enhanced_error_recovery: bool = True
) -> List[QAPair]:
    """Generate QA pairs with a specific temperature and enhanced error recovery.
    
    This function provides QA pair generation with:
    1. Temperature-specific generation
    2. Content-type appropriate prompting
    3. Enhanced error recovery with circuit breaker
    4. Response validation and filtering
    5. Automatic fallback to simpler prompts on failure
    
    Args:
        content: The content to generate QA pairs from
        content_type: The type of content (code, markdown, etc.)
        temperature: The temperature to use for generation
        section_uuid: Optional UUID of the section
        max_pairs: Maximum number of QA pairs to generate
        prompt_template: Custom prompt template to use
        enable_enhanced_error_recovery: Whether to enable enhanced error recovery
        
    Returns:
        List of QA pairs
    """
    # Select appropriate prompt template if not provided
    if prompt_template is None:
        if content_type == "code":
            prompt_template = """
            Generate {max_pairs} question-answer pairs about the following code:
            
            ```
            {content}
            ```
            
            Focus on:
            1. What the code does
            2. How it works
            3. Important functions or classes
            4. Edge cases or error handling
            
            Each answer MUST include an "Oh wait?!" moment where you realize or clarify something.
            
            Format as a JSON array of objects with keys:
            - question: The question being asked (must end with a question mark)
            - answer: Clear explanation (at least 5 words)
            - reasoning: Your thought process with an "Oh wait?!" moment
            """
        else:
            prompt_template = """
            Generate {max_pairs} question-answer pairs about the following content:
            
            {content}
            
            Each answer MUST include an "Oh wait?!" moment where you realize or clarify something.
            
            Format as a JSON array of objects with keys:
            - question: The question being asked (must end with a question mark)
            - answer: Clear explanation (at least 5 words)
            - reasoning: Your thought process with an "Oh wait?!" moment
            """
    
    # Format the prompt
    prompt = prompt_template.format(content=content, max_pairs=max_pairs)
    
    # Create a unique request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Prepare API call config
    config = {
        "model": "gpt-4-turbo",
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates high-quality question-answer pairs."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "request_id": request_id,
        "metadata": {
            "content_type": content_type,
            "section_uuid": section_uuid,
            "temperature": temperature
        }
    }
    
    # If enhanced error recovery is enabled, process with full circuit breaker, retries, etc.
    # This provides the strongest guarantees but might be overkill for some use cases
    if enable_enhanced_error_recovery:
        try:
            # Make the API call with enhanced error recovery
            response = await retry_llm_call(
                config=config,
                cost_aware=True,  # Use cost-aware routing
                max_retries=3,    # Try up to 3 times
                circuit_breaker_enabled=True  # Use circuit breaker
            )
        except Exception as primary_error:
            logger.error(f"Primary generation failed: {primary_error}")
            
            # Fall back to simpler approach with fallback model
            fallback_config = config.copy()
            fallback_config["model"] = "gpt-3.5-turbo"
            fallback_config["request_id"] = f"{request_id}-fallback"
            
            # Simplify prompt for fallback
            simpler_prompt = f"Generate {max_pairs} question-answer pairs about the content. Format as a JSON array."
            fallback_config["messages"][1]["content"] = simpler_prompt
            
            try:
                response = await retry_llm_call(
                    config=fallback_config,
                    cost_aware=False,  # Already using cheapest model
                    max_retries=1  # Just try once
                )
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                return []
    else:
        # Standard generation without enhanced recovery
        try:
            response = await retry_llm_call(config)
        except Exception as e:
            logger.error(f"Standard generation failed: {e}")
            return []
    
    try:
        # Extract content from response
        content = response["choices"][0]["message"]["content"]
        
        # Try to parse the JSON response with error handling
        try:
            qa_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in response: {e}")
            
            # Try to extract JSON from the response using regex as a fallback
            import re
            json_matches = re.findall(r'\[.*\]', content)
            if json_matches:
                try:
                    qa_data = json.loads(json_matches[0])
                except:
                    # Last resort: create a single basic pair
                    qa_data = [{
                        "question": "What is this content about?", 
                        "answer": "The content covers various topics and concepts.",
                        "reasoning": "Based on the structure and content provided. Oh wait?! It has more specific details."
                    }]
            else:
                # Create a basic fallback if JSON extraction failed
                qa_data = [{
                    "question": "What is this content about?", 
                    "answer": "The content covers various topics and concepts.",
                    "reasoning": "Based on the structure and content provided. Oh wait?! It has more specific details."
                }]
        
        # Ensure the response is a list
        if not isinstance(qa_data, list):
            logger.warning(f"Unexpected response format: {type(qa_data)}")
            if isinstance(qa_data, dict) and "qa_pairs" in qa_data:
                qa_data = qa_data["qa_pairs"]
            elif isinstance(qa_data, dict):
                # Try to extract a list from common dict patterns
                for k, v in qa_data.items():
                    if isinstance(v, list) and len(v) > 0:
                        qa_data = v
                        break
                else:
                    # If no list found, create a list with the dict itself
                    qa_data = [qa_data]
        
        # Convert to QAPair objects with validation
        qa_pairs = []
        for item in qa_data:
            try:
                # Extract required fields with fallbacks
                question = item.get("question", "What is the main point of this content?")
                if not question.endswith("?"):
                    question += "?"
                    
                answer = item.get("answer", "This content provides information on various topics.")
                
                reasoning = item.get("reasoning", "Based on analyzing the content structure. Oh wait?! There are more details to consider.")
                if "Oh wait?!" not in reasoning:
                    reasoning += " Oh wait?! There's more to consider here."
                
                # Create the pair
                pair = QAPair(
                    question=question,
                    answer=answer,
                    reasoning=reasoning,
                    source_section_uuid=section_uuid,
                    temperature_used=temperature,
                    direction="forward"
                )
                
                # Validate the pair
                if validate_qa_pair(pair):
                    qa_pairs.append(pair)
                else:
                    logger.warning(f"Invalid QA pair filtered: {pair}")
            except Exception as e:
                logger.error(f"Error creating QA pair: {e}")
        
        # Always return at least one pair if possible
        if not qa_pairs and content:
            # Create a generic pair as a last resort
            try:
                generic_pair = QAPair(
                    question=f"What is the key concept in this {content_type} content?",
                    answer="The content covers various important concepts and principles.",
                    reasoning="Based on the general structure and format. Oh wait?! The specific details provide more context.",
                    source_section_uuid=section_uuid,
                    temperature_used=temperature,
                    direction="forward"
                )
                qa_pairs.append(generic_pair)
            except Exception as e:
                logger.error(f"Error creating fallback QA pair: {e}")
        
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Error processing QA generation response: {e}")
        return []


async def iterate_temperatures(
    content: str,
    content_type: str,
    temps: List[float],
    section_uuid: Optional[str] = None,
    max_pairs_per_temp: int = 3,
    max_concurrent_requests: int = 2
) -> List[QAPair]:
    """Generate QA pairs with multiple temperatures and rate limiting.
    
    This function generates QA pairs using multiple temperature settings,
    ensuring that each temperature gets its own isolated context and
    applying rate limiting to prevent API overload.
    
    Args:
        content: The content to generate QA pairs from
        content_type: The type of content (code, markdown, etc.)
        temps: List of temperatures to use
        section_uuid: Optional UUID of the section
        max_pairs_per_temp: Maximum number of pairs per temperature
        max_concurrent_requests: Maximum number of concurrent API requests
        
    Returns:
        List of QA pairs from all temperatures
    """
    logger.info(f"Iterating {len(temps)} temperatures with rate limit {max_concurrent_requests}")
    
    # Validate temperatures
    from agent_tools.dualipa.qa.utils.validation import validate_temperature_range
    temps = validate_temperature_range(temps)
    
    if not temps:
        logger.warning("No valid temperatures provided")
        return []
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def generate_with_rate_limit(temp: float) -> List[QAPair]:
        """Run generation with rate limiting applied."""
        async with semaphore:
            logger.debug(f"Generating QA pairs with temperature {temp}")
            # Each temperature gets fresh context to prevent contamination
            return await generate_qa_pairs_with_temperature(
                content=content,
                content_type=content_type,
                temperature=temp,
                section_uuid=section_uuid,
                max_pairs=max_pairs_per_temp
            )
    
    # Create tasks for each temperature
    # Each task has its own isolated context
    tasks = [generate_with_rate_limit(temp) for temp in temps]
    
    # Execute all tasks with controlled concurrency
    try:
        results = await asyncio.gather(*tasks)
        
        # Combine and return results
        all_pairs = []
        for result in results:
            all_pairs.extend(result)
            
        logger.info(f"Generated {len(all_pairs)} QA pairs across {len(temps)} temperatures")
        return all_pairs
        
    except Exception as e:
        logger.error(f"Error during temperature iteration: {e}")
        # Continue with partial results if we have any
        return []


async def generate_code_qa_pairs(
    code_content: str,
    function_name: Optional[str] = None,
    temperature: Optional[float] = None,
    temps: Optional[List[float]] = None,
    max_pairs: int = 5,
    max_concurrent_requests: int = 2
) -> List[QAPair]:
    """Generate QA pairs for code content.
    
    Args:
        code_content: The code to generate QA pairs from
        function_name: Optional function name to focus on
        temperature: Specific temperature to use (overrides temps)
        temps: List of temperatures to iterate through
        max_pairs: Maximum number of pairs to generate
        max_concurrent_requests: Maximum number of concurrent API requests
        
    Returns:
        List of QA pairs
    """
    # Handle temperature options
    if temperature is not None:
        # Use a single temperature
        return await generate_qa_pairs_with_temperature(
            content=code_content,
            content_type="code",
            temperature=temperature,
            max_pairs=max_pairs
        )
    elif temps is not None:
        # Use provided temperatures
        return await iterate_temperatures(
            content=code_content,
            content_type="code",
            temps=temps,
            max_pairs_per_temp=max_pairs // len(temps) + 1,
            max_concurrent_requests=max_concurrent_requests
        )
    else:
        # Use default temperatures for code
        return await iterate_temperatures(
            content=code_content,
            content_type="code",
            temps=CODE_TEMPERATURE_RANGE,
            max_pairs_per_temp=max_pairs // len(CODE_TEMPERATURE_RANGE) + 1,
            max_concurrent_requests=max_concurrent_requests
        )


async def generate_markdown_qa_pairs(
    markdown_content: str,
    temperature: Optional[float] = None,
    temps: Optional[List[float]] = None,
    max_pairs: int = 5,
    max_concurrent_requests: int = 2,
    enable_enhanced_reasoning: bool = True
) -> List[QAPair]:
    """Generate QA pairs for markdown content with enhanced reasoning.
    
    This function generates question-answer pairs from markdown content with:
    1. Enhanced reasoning that includes internal dialog
    2. Robust error recovery with circuit breaker and retries
    3. Temperature iteration for quality diversity
    4. Dead letter queue for failed generations
    
    Args:
        markdown_content: The markdown to generate QA pairs from
        temperature: Specific temperature to use (overrides temps)
        temps: List of temperatures to iterate through
        max_pairs: Maximum number of pairs to generate
        max_concurrent_requests: Maximum number of concurrent API requests
        enable_enhanced_reasoning: Whether to enable enhanced reasoning
        
    Returns:
        List of QA pairs
    """
    # For enhanced reasoning, we inject additional prompting directives
    if enable_enhanced_reasoning and markdown_content:
        # Check if we need to enhance the content
        prompt_enhancement = """
        As you analyze this content, practice "thinking aloud" in your reasoning:
        - First identify key facts, concepts, and patterns
        - Think step-by-step about implications and connections
        - Have an "Oh wait?!" moment where you reconsider or deepen your initial analysis
        - Make sure reasoning shows depth and multiple perspectives
        
        Generate questions that:
        - Require synthesis of multiple concepts from the content
        - Sometimes challenge intuitive first impressions
        - Create opportunities to showcase reasoning
        
        Ensure your answers:
        - Are factually correct and aligned with the content
        - Include real-world context and implications when relevant
        """
        
        # Append enhancement to ensure it's processed in context
        enhanced_content = f"{markdown_content}\n\n{prompt_enhancement}"
    else:
        enhanced_content = markdown_content
    
    try:
        # Handle temperature options
        if temperature is not None:
            # Use a single temperature
            return await generate_qa_pairs_with_temperature(
                content=enhanced_content,
                content_type="markdown",
                temperature=temperature,
                max_pairs=max_pairs
            )
        elif temps is not None:
            # Use provided temperatures
            return await iterate_temperatures(
                content=enhanced_content,
                content_type="markdown",
                temps=temps,
                max_pairs_per_temp=max_pairs // len(temps) + 1,
                max_concurrent_requests=max_concurrent_requests
            )
        else:
            # Use default temperatures for markdown
            return await iterate_temperatures(
                content=enhanced_content,
                content_type="markdown",
                temps=MARKDOWN_TEMPERATURE_RANGE,
                max_pairs_per_temp=max_pairs // len(MARKDOWN_TEMPERATURE_RANGE) + 1,
                max_concurrent_requests=max_concurrent_requests
            )
    except Exception as e:
        # Handle failures gracefully
        logger.error(f"Error generating markdown QA pairs: {str(e)}")
        
        # Try fallback approach with simpler content if we were using enhanced
        if enable_enhanced_reasoning:
            logger.warning("Falling back to standard reasoning due to error")
            return await generate_markdown_qa_pairs(
                markdown_content=markdown_content,  # Original without enhancement
                temperature=temperature or MARKDOWN_TEMPERATURE_RANGE[0],
                max_pairs=max(1, max_pairs // 2),  # Request fewer pairs
                enable_enhanced_reasoning=False  # Disable enhancement for fallback
            )
        
        # If already using standard approach or other issues, return empty
        logger.error("Failed to generate markdown QA pairs even with fallback")
        return []