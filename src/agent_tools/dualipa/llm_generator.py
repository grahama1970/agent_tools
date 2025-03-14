"""
LLM-based Q&A Pair Generator for DuaLipa.

This module provides functions to generate question-answer pairs
from code and documentation using LLMs. It integrates with LiteLLM
for flexible LLM provider selection.

Official Documentation References:
- LiteLLM: https://docs.litellm.ai/docs/
- Loguru: https://loguru.readthedocs.io/en/stable/
- Pydantic: https://docs.pydantic.dev/latest/
- Tenacity: https://tenacity.readthedocs.io/en/stable/
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
- re: https://docs.python.org/3/library/re.html
"""

import os
import re
import json
import random
import sys
import asyncio
import time
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable, TypeVar, Generic
from loguru import logger
import tempfile
from pathlib import Path
import async_timeout

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Pydantic imports
from pydantic import BaseModel, Field, validator

# Initialize flag for litellm availability
LITELLM_AVAILABLE = False

try:
    import litellm
    from litellm import completion
    from tenacity import (
        retry,
        stop_after_attempt, 
        wait_exponential,
        retry_if_exception_type
    )
    LITELLM_AVAILABLE = True
    logger.info("LiteLLM is available for enhanced Q&A generation")
except ImportError:
    logger.warning("LiteLLM not installed. Enhanced Q&A generation will be limited.")

# Check if we're in the root project directory or subdirectory
current_dir = Path(__file__).parent
root_dir = current_dir
while "agent_tools" not in str(root_dir.name) and root_dir != root_dir.parent:
    root_dir = root_dir.parent

# Add LiteLLM paths
if "agent_tools" in str(root_dir):
    llm_dir = root_dir / "cursor_rules" / "llm"
    if llm_dir.exists() and str(llm_dir) not in sys.path:
        sys.path.append(str(llm_dir))

# Import litellm components
try:
    from litellm_call import call_litellm, CallOptions
    from initialize_litellm_cache import init_litellm, get_cache_key
    from retry_llm_call import retry_llm_call
    from multimodal_utils import process_image_for_llm, extract_image_from_base64
    from snippets.caching_tenacity import cached_retry
    
    # Initialize litellm with caching
    init_litellm()
    LITELLM_AVAILABLE = True
    logger.info("LiteLLM initialized and available for use")
except ImportError as e:
    LITELLM_AVAILABLE = False
    logger.warning(f"LiteLLM not available. Falling back to basic Q&A generation. Error: {e}")
    
    # Define placeholder for typing
    class CallOptions:
        pass

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Pydantic models for validation
class QAPair(BaseModel):
    """Question and answer pair with validation."""
    question: str = Field(..., min_length=10, description="The question text")
    answer: str = Field(..., min_length=10, description="The answer text")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v
    
    @validator('answer')
    def validate_answer(cls, v):
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v

class QAResponse(BaseModel):
    """Response model for QA generation."""
    pairs: List[QAPair] = Field(..., description="List of QA pairs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

class CodeQARequest(BaseModel):
    """Code QA generation request model."""
    code_content: str = Field(..., min_length=10, description="The source code content")
    function_name: Optional[str] = Field(None, description="Optional name of the function for targeted QA")
    max_pairs: int = Field(5, ge=1, le=20, description="Maximum number of QA pairs to generate")
    timeout: float = Field(30.0, ge=1.0, le=120.0, description="Timeout in seconds")

class MarkdownQARequest(BaseModel):
    """Markdown QA generation request model."""
    markdown_content: str = Field(..., min_length=10, description="The markdown content")
    section_title: Optional[str] = Field(None, description="Optional title of the section for targeted QA")
    max_pairs: int = Field(5, ge=1, le=20, description="Maximum number of QA pairs to generate")
    timeout: float = Field(30.0, ge=1.0, le=120.0, description="Timeout in seconds")

class BatchQAItem(BaseModel):
    """Item for batch QA generation."""
    content_type: str = Field(..., description="Type of content: 'code' or 'markdown'")
    content: str = Field(..., min_length=10, description="The content to generate QA pairs from")
    identifier: Optional[str] = Field(None, description="Optional identifier for the content")
    max_pairs: int = Field(5, ge=1, le=20, description="Maximum number of QA pairs to generate")

class BatchQARequest(BaseModel):
    """Batch QA generation request model."""
    items: List[BatchQAItem] = Field(..., min_items=1, max_items=50, description="Items to process")
    model: str = Field("gpt-3.5-turbo", description="LLM model to use")
    concurrency_limit: int = Field(3, ge=1, le=10, description="Maximum concurrent LLM requests")
    batch_size: int = Field(5, ge=1, le=20, description="Number of items to process in each batch")
    timeout_per_item: float = Field(30.0, ge=1.0, le=120.0, description="Timeout per item in seconds")

# Helper functions for timeouts and retries
async def with_timeout(coro, timeout_seconds: float, fallback_function: Callable = None, fallback_args: Tuple = None):
    """Execute a coroutine with a timeout and optional fallback.
    
    Args:
        coro: The coroutine to execute
        timeout_seconds: Timeout in seconds
        fallback_function: Function to call if timeout occurs
        fallback_args: Arguments to pass to fallback function
        
    Returns:
        Result of the coroutine or fallback function
    """
    try:
        async with async_timeout.timeout(timeout_seconds):
            return await coro
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds} seconds")
        if fallback_function and fallback_args:
            logger.info("Executing fallback function")
            return fallback_function(*fallback_args)
        raise

@cached_retry(
    retries=3,
    cache_ttl=3600,  # 1 hour
    exceptions=(ConnectionError, TimeoutError)
)
async def generate_code_qa_pairs(
    code_content: str,
    function_name: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,  # Now optional to allow for random variation
    max_pairs: int = 5,
    timeout: float = 30.0
) -> List[Dict[str, str]]:
    """Generate question-answer pairs from code content using LiteLLM.
    
    Args:
        code_content: The source code content
        function_name: Optional name of the function for targeted QA
        model: LLM model to use
        temperature: Sampling temperature (if None, will use random variation)
        max_pairs: Maximum number of QA pairs to generate
        timeout: Timeout in seconds
        
    Returns:
        List of question-answer dictionaries
    """
    # Validate timeout parameter
    if timeout <= 0:
        logger.warning(f"Invalid timeout value: {timeout}. Using default 30s.")
        timeout = 30.0
        
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available, using basic Q&A generation")
        return await asyncio.to_thread(_generate_basic_code_qa, code_content, function_name, max_pairs)
    
    # Use the with_timeout helper with fallback to basic generation
    return await with_timeout(
        _generate_code_qa_pairs_internal(
            code_content=code_content,
            function_name=function_name,
            model=model,
            temperature=temperature,
            max_pairs=max_pairs
        ),
        timeout_seconds=timeout,
        fallback_function=_generate_basic_code_qa,
        fallback_args=(code_content, function_name, max_pairs)
    )

async def _generate_code_qa_pairs_internal(
    code_content: str,
    function_name: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Internal implementation for generate_code_qa_pairs without timeout handling."""
    try:
        # Validate request
        request = CodeQARequest(
            code_content=code_content,
            function_name=function_name,
            max_pairs=max_pairs,
            timeout=30.0  # Default timeout
        )
        
        # If temperature is None, use random variation
        if temperature is None:
            # Use a lower temperature for more focused code questions
            temperature = random.uniform(0.1, 0.5)
            logger.debug(f"Using random temperature: {temperature:.2f}")
        
        # Extract function or class if provided
        extracted_code = code_content
        if function_name:
            extracted_code = _extract_function_or_class(code_content, function_name)
        
        # Prepare the prompt
        system_prompt = f"""You are an expert code analyzer. Your task is to generate {max_pairs} high-quality 
        question-answer pairs from the provided code. Each question should be precise and directly related to the code.
        Focus on the function's purpose, parameters, return values, usage patterns, and error handling.
        
        For each question-answer pair:
        1. Make the question specific enough that the answer is unambiguous
        2. Include code examples where appropriate in the answers
        3. Include both basic usage questions and more advanced conceptual questions
        4. Include at least one question about error handling if applicable
        5. Include at least one implementation question that requires showing the complete code
        6. For questions about implementation, always include the full implementation code in the answer
        
        Format your response as a JSON array of objects, each with 'question' and 'answer' fields.
        """
        
        user_prompt = f"""Generate question-answer pairs for the following code:
        
        ```
        {extracted_code}
        ```
        
        Return ONLY a JSON array of objects, each with 'question' and 'answer' fields.
        Remember that questions about implementation should always include the complete function code in the answer.
        """
        
        # Configure call options
        options = CallOptions(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            cache=True,
            metadata={
                "source": "dualipa",
                "content_type": "code",
                "function": function_name or "full"
            }
        )
        
        # Call LiteLLM with retry
        response = await retry_llm_call(
            options=options,
            max_retries=2,
            retry_delay=1,
            exponential_backoff=True
        )
        
        if not response:
            logger.error("Failed to get response from LiteLLM for code QA")
            return await asyncio.to_thread(_generate_basic_code_qa, code_content, function_name, max_pairs)
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from the response
        json_str = _extract_json_from_text(content)
        
        # Validate QA pairs
        qa_pairs = _validate_qa_pairs(json_str)
        
        # Validate and enhance QA pairs using RapidFuzz
        validated_pairs = await validate_and_enhance_qa_pairs(
            qa_pairs=qa_pairs,
            original_content=code_content,
            function_name=function_name,
            deduplicate=True
        )
        
        return validated_pairs
        
    except Exception as e:
        logger.error(f"Error generating code QA pairs: {str(e)}")
        return await asyncio.to_thread(_generate_basic_code_qa, code_content, function_name, max_pairs)

@cached_retry(
    retries=3,
    cache_ttl=3600,  # 1 hour
    exceptions=(ConnectionError, TimeoutError)
)
async def generate_markdown_qa_pairs(
    markdown_content: str,
    section_title: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,  # Now optional to allow for random variation
    max_pairs: int = 5,
    timeout: float = 30.0
) -> List[Dict[str, str]]:
    """Generate question-answer pairs from markdown content using LiteLLM.
    
    Args:
        markdown_content: The markdown content
        section_title: Optional title of the section for targeted QA
        model: LLM model to use
        temperature: Sampling temperature (if None, will use random variation)
        max_pairs: Maximum number of QA pairs to generate
        timeout: Timeout in seconds
        
    Returns:
        List of question-answer dictionaries
    """
    # Validate timeout parameter
    if timeout <= 0:
        logger.warning(f"Invalid timeout value: {timeout}. Using default 30s.")
        timeout = 30.0
        
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available, using basic Q&A generation")
        return await asyncio.to_thread(_generate_basic_markdown_qa, markdown_content, section_title, max_pairs)
    
    # Use the with_timeout helper with fallback to basic generation
    return await with_timeout(
        _generate_markdown_qa_pairs_internal(
            markdown_content=markdown_content,
            section_title=section_title,
            model=model,
            temperature=temperature,
            max_pairs=max_pairs
        ),
        timeout_seconds=timeout,
        fallback_function=_generate_basic_markdown_qa,
        fallback_args=(markdown_content, section_title, max_pairs)
    )

async def _generate_markdown_qa_pairs_internal(
    markdown_content: str,
    section_title: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Internal implementation for generate_markdown_qa_pairs without timeout handling."""
    try:
        # Validate request
        request = MarkdownQARequest(
            markdown_content=markdown_content,
            section_title=section_title,
            max_pairs=max_pairs,
            timeout=30.0  # Default timeout
        )
        
        # If temperature is None, use random variation
        if temperature is None:
            # Use a higher temperature for more creative questions
            temperature = random.uniform(0.3, 0.7)
            logger.debug(f"Using random temperature: {temperature:.2f}")
        
        # Prepare the section header if specified
        section_header = f" specifically for the section titled '{section_title}'" if section_title else ""
        
        # Prepare the prompt
        system_prompt = f"""You are an expert in technical documentation. Your task is to generate {max_pairs} 
        high-quality question-answer pairs from the provided markdown content{section_header}.
        
        Each question should be clear and specific to a certain aspect of the documentation. For question-answer pairs:
        1. Focus on key concepts, features, parameters, and usage patterns
        2. Include both factual questions and conceptual questions
        3. Make questions that someone unfamiliar with the documentation might ask
        4. Include code examples in answers where appropriate
        5. Ensure questions and answers are technically accurate and reflect the content
        
        Format your response as a JSON array of objects, each with 'question' and 'answer' fields.
        """
        
        user_prompt = f"""Generate question-answer pairs for the following markdown documentation:
        
        ```markdown
        {markdown_content}
        ```
        
        Return ONLY a JSON array of objects, each with 'question' and 'answer' fields.
        """
        
        # Configure call options
        options = CallOptions(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            cache=True,
            metadata={
                "source": "dualipa",
                "content_type": "markdown",
                "section": section_title or "full"
            }
        )
        
        # Call LiteLLM with retry
        response = await retry_llm_call(
            options=options,
            max_retries=2,
            retry_delay=1,
            exponential_backoff=True
        )
        
        if not response:
            logger.error("Failed to get response from LiteLLM for markdown QA")
            return await asyncio.to_thread(_generate_basic_markdown_qa, markdown_content, section_title, max_pairs)
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from the response
        json_str = _extract_json_from_text(content)
        
        # Validate QA pairs
        qa_pairs = _validate_qa_pairs(json_str)
        
        # Validate and enhance QA pairs using RapidFuzz
        validated_pairs = await validate_and_enhance_qa_pairs(
            qa_pairs=qa_pairs,
            original_content=markdown_content,
            deduplicate=True
        )
        
        return validated_pairs
        
    except Exception as e:
        logger.error(f"Error generating markdown QA pairs: {str(e)}")
        return await asyncio.to_thread(_generate_basic_markdown_qa, markdown_content, section_title, max_pairs)

# Sync wrapper functions following the pattern in 005-async-patterns.mdc
def generate_code_qa_pairs_sync(
    code_content: str,
    function_name: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    max_pairs: int = 5,
    timeout: float = 30.0
) -> List[Dict[str, str]]:
    """Synchronous wrapper for generate_code_qa_pairs.
    
    This function provides a synchronous interface to the async QA generation function.
    
    Args:
        code_content: The source code content
        function_name: Optional name of the function for targeted QA
        model: LLM model to use
        temperature: Sampling temperature (if None, will use random variation)
        max_pairs: Maximum number of QA pairs to generate
        timeout: Timeout in seconds
        
    Returns:
        List of question-answer dictionaries
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        generate_code_qa_pairs(
            code_content=code_content,
            function_name=function_name,
            model=model,
            temperature=temperature,
            max_pairs=max_pairs,
            timeout=timeout
        )
    )

def generate_markdown_qa_pairs_sync(
    markdown_content: str,
    section_title: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = None,
    max_pairs: int = 5,
    timeout: float = 30.0
) -> List[Dict[str, str]]:
    """Synchronous wrapper for generate_markdown_qa_pairs.
    
    This function provides a synchronous interface to the async QA generation function.
    
    Args:
        markdown_content: The markdown content
        section_title: Optional title of the section for targeted QA
        model: LLM model to use
        temperature: Sampling temperature (if None, will use random variation)
        max_pairs: Maximum number of QA pairs to generate
        timeout: Timeout in seconds
        
    Returns:
        List of question-answer dictionaries
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        generate_markdown_qa_pairs(
            markdown_content=markdown_content,
            section_title=section_title,
            model=model,
            temperature=temperature,
            max_pairs=max_pairs,
            timeout=timeout
        )
    )

# Implementing batch processing with semaphore for concurrency control
async def batch_generate_qa_pairs(
    items: List[BatchQAItem],
    model: str = "gpt-3.5-turbo",
    concurrency_limit: int = 3,
    batch_size: int = 5,
    timeout_per_item: float = 30.0
) -> List[Dict[str, Any]]:
    """Generate QA pairs for multiple content items in batches with concurrency control.
    
    This function implements the batch processing pattern from 005-async-patterns.mdc,
    using semaphores to limit concurrency and processing items in batches for efficiency.
    
    Args:
        items: List of content items to process
        model: LLM model to use
        concurrency_limit: Maximum number of concurrent LLM requests
        batch_size: Number of items to process in each batch
        timeout_per_item: Timeout per item in seconds
        
    Returns:
        List of dictionaries with content identifier and QA pairs
    """
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = []
    
    async def process_with_semaphore(item: BatchQAItem) -> Dict[str, Any]:
        """Process a single item with semaphore for concurrency control."""
        async with semaphore:
            try:
                if item.content_type.lower() == "code":
                    qa_pairs = await generate_code_qa_pairs(
                        code_content=item.content,
                        function_name=item.identifier,
                        model=model,
                        max_pairs=item.max_pairs,
                        timeout=timeout_per_item
                    )
                elif item.content_type.lower() == "markdown":
                    qa_pairs = await generate_markdown_qa_pairs(
                        markdown_content=item.content,
                        section_title=item.identifier,
                        model=model,
                        max_pairs=item.max_pairs,
                        timeout=timeout_per_item
                    )
                else:
                    logger.warning(f"Unsupported content type: {item.content_type}")
                    return {
                        "identifier": item.identifier,
                        "content_type": item.content_type,
                        "error": f"Unsupported content type: {item.content_type}",
                        "qa_pairs": []
                    }
                
                return {
                    "identifier": item.identifier,
                    "content_type": item.content_type,
                    "qa_pairs": qa_pairs
                }
            except Exception as e:
                logger.error(f"Error processing item {item.identifier}: {str(e)}")
                return {
                    "identifier": item.identifier,
                    "content_type": item.content_type,
                    "error": str(e),
                    "qa_pairs": []
                }
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size} with {len(batch)} items")
        
        # Create tasks for the batch
        batch_tasks = [process_with_semaphore(item) for item in batch]
        
        # Process batch with error handling
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any exceptions and add results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {str(result)}")
                results.append({
                    "identifier": "unknown",
                    "content_type": "unknown",
                    "error": str(result),
                    "qa_pairs": []
                })
            else:
                results.append(result)
    
    return results

def batch_generate_qa_pairs_sync(
    items: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    concurrency_limit: int = 3,
    batch_size: int = 5,
    timeout_per_item: float = 30.0
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for batch_generate_qa_pairs.
    
    This function provides a synchronous interface to the async batch processing function.
    
    Args:
        items: List of content items to process (each with content_type, content, and optional identifier)
        model: LLM model to use
        concurrency_limit: Maximum number of concurrent LLM requests
        batch_size: Number of items to process in each batch
        timeout_per_item: Timeout per item in seconds
        
    Returns:
        List of dictionaries with content identifier and QA pairs
    """
    # Convert dictionary items to BatchQAItem objects
    batch_items = []
    for item in items:
        try:
            batch_items.append(BatchQAItem(
                content_type=item["content_type"],
                content=item["content"],
                identifier=item.get("identifier"),
                max_pairs=item.get("max_pairs", 5)
            ))
        except Exception as e:
            logger.error(f"Error converting item to BatchQAItem: {str(e)}")
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        batch_generate_qa_pairs(
            items=batch_items,
            model=model,
            concurrency_limit=concurrency_limit,
            batch_size=batch_size,
            timeout_per_item=timeout_per_item
        )
    )

def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain non-JSON content.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Extracted JSON string or empty string if not found
    """
    # Try to find JSON between markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find content between square brackets or curly braces
    bracket_match = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", text)
    
    if bracket_match:
        return bracket_match.group(0).strip()
    
    # Return the original text if no JSON-like content is found
    return text

def _validate_qa_pairs(json_str: str) -> List[Dict[str, str]]:
    """Validate QA pairs from JSON string using Pydantic.
    
    Args:
        json_str: JSON string containing QA pairs
        
    Returns:
        List of validated QA pairs
    """
    try:
        # Parse the JSON
        if not json_str:
            return []
            
        qa_data = json.loads(json_str)
        
        # Handle both list and dictionary formats
        qa_pairs = qa_data if isinstance(qa_data, list) else [qa_data]
        
        # Validate each pair with Pydantic
        validated_pairs = []
        
        for pair in qa_pairs:
            try:
                # Validate the QA pair using Pydantic model
                qa_pair = QAPair(**pair)
                validated_pairs.append({"question": qa_pair.question, "answer": qa_pair.answer})
            except Exception as e:
                logger.warning(f"Invalid QA pair: {pair}. Error: {str(e)}")
        
        return validated_pairs
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return []

def _extract_function_or_class(code_content: str, name: str) -> str:
    """Extract a function or class from code content by name.
    
    Args:
        code_content: The source code content
        name: Name of the function or class to extract
        
    Returns:
        Extracted function or class code, or the original content if not found
    """
    import re
    lines = code_content.split('\n')
    
    # Patterns for function and class definitions
    function_pattern = rf'def\s+{name}\s*\('
    class_pattern = rf'class\s+{name}\s*'
    
    found_idx = -1
    indent_level = 0
    is_function = False
    
    # Find the definition
    for i, line in enumerate(lines):
        if re.search(function_pattern, line):
            found_idx = i
            is_function = True
            indent_level = len(line) - len(line.lstrip())
            break
        elif re.search(class_pattern, line):
            found_idx = i
            is_function = False
            indent_level = len(line) - len(line.lstrip())
            break
    
    if found_idx == -1:
        # Not found, return the original content
        return code_content
    
    # Extract the code block
    extracted_lines = [lines[found_idx]]
    
    for i in range(found_idx + 1, len(lines)):
        line = lines[i]
        
        # If we've reached a line with the same or less indentation,
        # and it's not empty, we've reached the end of the function/class
        if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
            if not is_function or not line.strip().startswith("@"):
                break
        
        extracted_lines.append(line)
    
    return '\n'.join(extracted_lines)

def _generate_basic_code_qa(
    code_content: str,
    function_name: Optional[str] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Generate basic question-answer pairs for code without using LiteLLM.
    
    Args:
        code_content: The source code content
        function_name: Optional name of the function for targeted QA
        max_pairs: Maximum number of QA pairs to generate
        
    Returns:
        List of question-answer dictionaries
    """
    qa_pairs = []
    
    # Extract function or class if provided
    if function_name:
        code_content = _extract_function_or_class(code_content, function_name)
    
    # Basic extraction of docstrings and function signatures
    import re
    
    # Extract function docstring
    docstring_pattern = r'def\s+(\w+)\s*\(([^)]*)\)[^"\']*(["\']){3}(.*?)\3'
    docstring_matches = re.finditer(docstring_pattern, code_content, re.DOTALL)
    
    for match in docstring_matches:
        if len(qa_pairs) >= max_pairs:
            break
            
        fn_name = match.group(1)
        fn_params = match.group(2)
        docstring = match.group(4).strip()
        
        qa_pairs.append({
            "question": f"What does the function `{fn_name}` do?",
            "answer": docstring
        })
        
        qa_pairs.append({
            "question": f"What are the parameters of `{fn_name}`?",
            "answer": f"The parameters are: {fn_params}"
        })
    
    # Extract class docstring
    class_pattern = r'class\s+(\w+)[^"\']*(["\']){3}(.*?)\2'
    class_matches = re.finditer(class_pattern, code_content, re.DOTALL)
    
    for match in class_matches:
        if len(qa_pairs) >= max_pairs:
            break
            
        class_name = match.group(1)
        docstring = match.group(3).strip()
        
        qa_pairs.append({
            "question": f"What is the purpose of the `{class_name}` class?",
            "answer": docstring
        })
    
    # If we still need more QA pairs, add generic ones
    if len(qa_pairs) < max_pairs:
        qa_pairs.append({
            "question": "Can you show me the implementation?",
            "answer": code_content
        })
    
    return qa_pairs[:max_pairs]

def _generate_basic_markdown_qa(
    markdown_content: str,
    section_title: Optional[str] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Generate basic question-answer pairs for markdown without using LiteLLM.
    
    Args:
        markdown_content: The markdown content
        section_title: Optional title of the section for targeted QA
        max_pairs: Maximum number of QA pairs to generate
        
    Returns:
        List of question-answer dictionaries
    """
    qa_pairs = []
    
    # Split by headers to find sections
    import re
    headers = re.findall(r'^(#+)\s+(.*?)$', markdown_content, re.MULTILINE)
    
    if not headers:
        # No headers, treat all content as one section
        qa_pairs.append({
            "question": f"What is described in the documentation?",
            "answer": markdown_content
        })
        
        qa_pairs.append({
            "question": f"Can you summarize the key points of the documentation?",
            "answer": f"The documentation covers: {markdown_content[:100]}..."
        })
    else:
        # Process each section
        for i, (hashes, title) in enumerate(headers):
            if len(qa_pairs) >= max_pairs:
                break
                
            if section_title and title != section_title:
                continue
                
            level = len(hashes)
            
            # Find the next header with the same or higher level
            section_end = len(markdown_content)
            for j in range(i + 1, len(headers)):
                next_level, next_title = headers[j]
                if len(next_level) <= level:
                    # Find position of this header in the content
                    next_pos = markdown_content.find(f"{next_level} {next_title}")
                    if next_pos != -1:
                        section_end = next_pos
                        break
            
            # Find the start of this section
            section_start = markdown_content.find(f"{hashes} {title}")
            if section_start == -1:
                continue
                
            # Skip the header line
            section_start = markdown_content.find('\n', section_start) + 1
            
            if section_start >= section_end:
                continue
                
            section_content = markdown_content[section_start:section_end].strip()
            
            qa_pairs.append({
                "question": f"What is covered in the section '{title}'?",
                "answer": section_content
            })
            
            # If this is the section we're specifically looking for, add more questions
            if section_title and title == section_title:
                qa_pairs.append({
                    "question": f"Can you explain the main points of '{title}'?",
                    "answer": section_content
                })
                
                # Look for lists in the section
                list_items = re.findall(r'^[\s]*[*-]\s+(.*?)$', section_content, re.MULTILINE)
                if list_items:
                    qa_pairs.append({
                        "question": f"What are the key items listed in the '{title}' section?",
                        "answer": "The key items are:\n" + "\n".join([f"- {item}" for item in list_items])
                    })
    
    # If we still need more QA pairs, add generic ones
    if len(qa_pairs) < max_pairs:
        qa_pairs.append({
            "question": "Can you show me the full documentation?",
            "answer": markdown_content
        })
    
    return qa_pairs[:max_pairs]

async def process_code_with_images(
    code_content: str,
    image_path: Optional[str] = None,
    function_name: Optional[str] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Process code with optional image context to generate QA pairs.
    
    Args:
        code_content: The source code content
        image_path: Optional path to a related image (e.g., diagram)
        function_name: Optional name of the function for targeted QA
        max_pairs: Maximum number of QA pairs to generate
        
    Returns:
        List of question-answer dictionaries
    """
    if not LITELLM_AVAILABLE or not image_path or not os.path.exists(image_path):
        # Fall back to regular code QA if image processing isn't available
        return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)
    
    try:
        # Process the image for LLM
        image_data = process_image_for_llm(image_path)
        
        # Extract function or class if provided
        if function_name:
            code_content = _extract_function_or_class(code_content, function_name)
        
        # Prepare the prompt
        system_prompt = f"""You are an expert code analyzer. You have been provided with both code and a relevant diagram or image.
        Your task is to generate {max_pairs} high-quality question-answer pairs that relate the code to the image where appropriate.
        
        For each question-answer pair:
        1. Make the question specific enough that the answer is unambiguous
        2. Include code examples where appropriate in the answers
        3. When relevant, reference how the code relates to elements in the diagram
        4. Include at least one question about how the code implementation relates to the visual representation
        
        Format your response as a JSON array of objects, each with 'question' and 'answer' fields.
        """
        
        user_prompt = f"""Generate question-answer pairs for the following code and image:
        
        ```
        {code_content}
        ```
        
        Return ONLY a JSON array of objects, each with 'question' and 'answer' fields.
        """
        
        # Configure call options
        options = CallOptions(
            model="gpt-4-vision-preview",  # Use vision model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]}
            ],
            temperature=0.3,
            cache=True,
            metadata={
                "source": "dualipa",
                "content_type": "code_with_image",
                "function_name": function_name or "whole_file" 
            }
        )
        
        # Call LiteLLM with retry
        response = await retry_llm_call(
            options=options,
            max_retries=2,
            retry_delay=1,
            exponential_backoff=True
        )
        
        if not response:
            logger.error("Failed to get response from vision model")
            return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from the response
        json_str = _extract_json_from_text(content)
        
        if not json_str:
            logger.error("Failed to extract JSON from vision model response")
            return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)
        
        # Parse JSON
        try:
            qa_pairs = json.loads(json_str)
            
            # Validate format
            if not isinstance(qa_pairs, list):
                logger.error("Vision model response is not a list")
                return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)
            
            # Validate each QA pair with Pydantic
            valid_pairs = []
            for pair in qa_pairs:
                try:
                    validated_pair = QAPair(**pair)
                    valid_pairs.append(validated_pair.dict())
                except Exception as e:
                    logger.warning(f"Invalid QA pair from vision model: {e}")
                    continue
            
            return valid_pairs[:max_pairs]
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error from vision model: {e}")
            return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)
    
    except Exception as e:
        logger.error(f"Error processing code with images: {e}")
        return await generate_code_qa_pairs(code_content, function_name, max_pairs=max_pairs)

async def debug_llm_generator() -> None:
    """Debug function to test LLM-based Q&A generation.
    
    Tests:
    1. Code QA generation
    2. Markdown QA generation
    3. Reverse QA generation
    4. Cache functionality
    5. Error handling and fallbacks
    """
    # Test code for Q&A generation
    test_code = """
def calculate_average(numbers: list) -> float:
    \"\"\"Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers
        
    Returns:
        The average of the numbers
        
    Raises:
        ValueError: If the list is empty
    \"\"\"
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
"""
    
    # Test markdown for Q&A generation
    test_markdown = """# Test Documentation
    
This is a test document for Q&A generation.

## Installation

To install the package, run:

```
pip install mypackage
```

## Usage

Here's how to use the package:

```python
import mypackage

result = mypackage.calculate_average([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```
"""
    
    print("Testing code Q&A generation...")
    code_qa_start = asyncio.get_event_loop().time()
    code_qa = await generate_code_qa_pairs(test_code, max_pairs=3)
    code_qa_time = asyncio.get_event_loop().time() - code_qa_start
    print(f"Generated {len(code_qa)} code Q&A pairs in {code_qa_time:.2f} seconds:")
    for i, qa in enumerate(code_qa):
        print(f"{i+1}. Q: {qa.get('question')}")
        print(f"   A: {qa.get('answer')[:100]}...")
    
    print("\nTesting markdown Q&A generation...")
    md_qa_start = asyncio.get_event_loop().time()
    md_qa = await generate_markdown_qa_pairs(test_markdown, max_pairs=3)
    md_qa_time = asyncio.get_event_loop().time() - md_qa_start
    print(f"Generated {len(md_qa)} markdown Q&A pairs in {md_qa_time:.2f} seconds:")
    for i, qa in enumerate(md_qa):
        print(f"{i+1}. Q: {qa.get('question')}")
        print(f"   A: {qa.get('answer')[:100]}...")
    
    print("\nTesting reverse Q&A generation...")
    all_qa = code_qa + md_qa
    reverse_qa_start = asyncio.get_event_loop().time()
    reverse_qa = await generate_reverse_qa_pairs(all_qa, max_reverse_pairs=2)
    reverse_qa_time = asyncio.get_event_loop().time() - reverse_qa_start
    print(f"Generated {len(reverse_qa)} reverse Q&A pairs in {reverse_qa_time:.2f} seconds:")
    for i, qa in enumerate(reverse_qa):
        print(f"{i+1}. Q: {qa.get('question')}")
        print(f"   A: {qa.get('answer')[:100]}...")
    
    print("\nTesting cache functionality (should be faster)...")
    cache_test_start = asyncio.get_event_loop().time()
    cached_qa = await generate_code_qa_pairs(test_code, max_pairs=3)
    cache_test_time = asyncio.get_event_loop().time() - cache_test_start
    print(f"Retrieved {len(cached_qa)} code Q&A pairs from cache in {cache_test_time:.2f} seconds")
    
    if LITELLM_AVAILABLE:
        cache_speedup = code_qa_time / max(cache_test_time, 0.001)  # Avoid division by zero
        print(f"Cache speedup factor: {cache_speedup:.2f}x")
    
    print("\nDebug tests completed")

def check_litellm_available() -> bool:
    """
    Check if LiteLLM is available and properly configured.
    
    Returns:
        True if LiteLLM is available and properly configured, False otherwise
    """
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM is not installed")
        return False
    
    try:
        # Check if any provider is configured
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
            return True
        
        # Look for litellm config
        if os.path.exists(os.path.expanduser("~/.litellm.config.yaml")):
            return True
            
        logger.warning("No LLM provider API keys found in environment variables")
        return False
    except Exception as e:
        logger.error(f"Error checking LiteLLM availability: {str(e)}")
        return False

def extract_json_from_markdown(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Extract JSON from markdown text, even if it's not properly formatted.
    
    Args:
        text: The markdown text that should contain JSON
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # Try to extract code blocks with json
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        
        if json_blocks:
            for block in json_blocks:
                try:
                    return json.loads(block.strip())
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON without code blocks
        # Look for arrays of objects with question and answer fields
        json_pattern = r"\[\s*{\s*\"question\".*\"answer\".*}\s*(?:,\s*{\s*\"question\".*\"answer\".*}\s*)*\]"
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        if json_matches:
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try to extract individual QA pairs
        qa_pattern = r'{\s*"question":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer":\s*"((?:[^"\\]|\\.)*)"\s*}'
        qa_matches = re.findall(qa_pattern, text)
        
        if qa_matches:
            result = []
            for question, answer in qa_matches:
                result.append({
                    "question": question.replace('\\"', '"'),
                    "answer": answer.replace('\\"', '"')
                })
            return result
            
        logger.warning("Failed to extract JSON from LLM response")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting JSON from markdown: {str(e)}")
        return None

def extract_function_or_class(content: str, name: str, type_name: str = "function") -> Optional[str]:
    """
    Extract a function or class definition from code content by name.
    
    Args:
        content: The source code
        name: The name of the function or class to extract
        type_name: Either "function" or "class"
        
    Returns:
        The extracted function/class definition with proper indentation,
        or the original content if the function/class is not found
    """
    try:
        if type_name == "function":
            pattern = rf'def\s+{re.escape(name)}\s*\('
        elif type_name == "class":
            pattern = rf'class\s+{re.escape(name)}\s*[:\(]'
        else:
            logger.warning(f"Invalid type_name '{type_name}', must be 'function' or 'class'")
            return content
        
        match = re.search(pattern, content)
        if not match:
            logger.debug(f"{type_name.capitalize()} '{name}' not found in content")
            return content
        
        start_pos = match.start()
        
        # Get the indentation level of the function/class definition
        line_start = content.rfind('\n', 0, start_pos) + 1
        indentation = start_pos - line_start
        
        # Find the end of the function/class by tracking indentation
        lines = content[start_pos:].split('\n')
        end_line = len(lines)
        
        for i, line in enumerate(lines[1:], 1):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if indentation is less than or equal to the function/class indentation
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent <= indentation and line.strip():
                end_line = i
                break
        
        # Extract the function/class definition
        extracted = '\n'.join(lines[:end_line])
        
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting {type_name} '{name}' from content: {str(e)}")
        return content

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_code_related_qa_pairs(
    content: str, 
    name: str, 
    entity_type: str = "function", 
    temperature: Optional[float] = None
) -> List[Dict[str, str]]:
    """
    Generate question-answer pairs related to a specific function or class in code.
    
    Args:
        content: The source code
        name: The name of the function or class
        entity_type: Either "function" or "class"
        temperature: Temperature for LLM generation (0.0 to 1.0)
        
    Returns:
        List of QA pairs in the format {"question": "...", "answer": "..."}
    """
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available, returning empty QA pairs")
        return []
    
    try:
        # Extract the function or class code
        extracted_code = extract_function_or_class(content, name, entity_type)
        
        # Use random temperature if None is provided
        if temperature is None:
            temperature = random.uniform(0.1, 0.5)
        
        # Create prompt
        prompt = f"""You are an expert programming tutor. 
Your task is to create question-answer pairs about the following {entity_type} in Python code:

```python
{extracted_code}
```

Create 3 question-answer pairs about this {entity_type}. The questions should cover:
1. The purpose and functionality of the {entity_type}
2. Implementation details and techniques used
3. How to use the {entity_type} with example code

Format your response as a JSON array with "question" and "answer" fields like this:
```json
[
  {{
    "question": "What is the purpose of the {name} {entity_type}?",
    "answer": "Detailed explanation..."
  }},
  ...
]
```
Only include the JSON in your response, no additional text."""

        # Call LLM
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # Extract and parse JSON from response
        if response and response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            qa_pairs = extract_json_from_markdown(content)
            
            if qa_pairs:
                return qa_pairs
            else:
                logger.warning(f"Failed to parse QA pairs from LLM response for {entity_type} '{name}'")
                return []
        else:
            logger.warning(f"Empty or invalid response from LLM for {entity_type} '{name}'")
            return []
            
    except Exception as e:
        logger.error(f"Error generating code-related QA pairs: {str(e)}")
        return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_markdown_related_qa_pairs(
    content: str, 
    section_name: str = "document", 
    temperature: Optional[float] = None
) -> List[Dict[str, str]]:
    """
    Generate question-answer pairs from markdown content.
    
    Args:
        content: The markdown content
        section_name: Name of the section or document
        temperature: Temperature for LLM generation (0.0 to 1.0)
        
    Returns:
        List of QA pairs in the format {"question": "...", "answer": "..."}
    """
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available, returning empty QA pairs")
        return []
    
    try:
        # Use random temperature if None is provided
        if temperature is None:
            temperature = random.uniform(0.3, 0.7)
        
        # Create prompt
        prompt = f"""You are an expert technical writer. 
Your task is to create question-answer pairs about the following markdown content:

```markdown
{content[:4000]}  # Limit content to avoid token limits
```

Create 3-5 question-answer pairs about this content. The questions should cover:
1. Key concepts explained in the document
2. Technical details and procedures described
3. Practical applications or examples mentioned

Format your response as a JSON array with "question" and "answer" fields like this:
```json
[
  {{
    "question": "What is {section_name} about?",
    "answer": "Detailed explanation..."
  }},
  ...
]
```
Only include the JSON in your response, no additional text."""

        # Call LLM
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # Extract and parse JSON from response
        if response and response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            qa_pairs = extract_json_from_markdown(content)
            
            if qa_pairs:
                return qa_pairs
            else:
                logger.warning(f"Failed to parse QA pairs from LLM response for '{section_name}'")
                return []
        else:
            logger.warning(f"Empty or invalid response from LLM for '{section_name}'")
            return []
            
    except Exception as e:
        logger.error(f"Error generating markdown-related QA pairs: {str(e)}")
        return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_reverse_qa_pairs(
    answer_text: str,
    num_questions: int = 2,
    temperature: Optional[float] = None
) -> List[Dict[str, str]]:
    """
    Generate questions for a given answer text.
    
    Args:
        answer_text: The answer for which to generate questions
        num_questions: Number of questions to generate
        temperature: Temperature for LLM generation (0.0 to 1.0)
        
    Returns:
        List of QA pairs in the format {"question": "...", "answer": "..."}
    """
    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available, returning empty QA pairs")
        return []
    
    try:
        # Use random temperature if None is provided
        if temperature is None:
            temperature = random.uniform(0.3, 0.6)
        
        # Create prompt
        prompt = f"""You are an expert tutor.
Your task is to create {num_questions} different questions for the following answer:

ANSWER:
```
{answer_text[:4000]}  # Limit content to avoid token limits
```

Create {num_questions} questions that would have this as their answer. The questions should be:
1. Specific and clearly linked to the answer content
2. Varied in their focus and wording
3. Natural and conversational

Format your response as a JSON array with "question" and "answer" fields like this:
```json
[
  {{
    "question": "Ask a question that would have the provided text as its answer?",
    "answer": "{answer_text[:50]}..."
  }},
  ...
]
```
Only include the JSON in your response, no additional text."""

        # Call LLM
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # Extract and parse JSON from response
        if response and response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            qa_pairs = extract_json_from_markdown(content)
            
            # Fix answers to use the full original answer
            if qa_pairs:
                for qa_pair in qa_pairs:
                    qa_pair["answer"] = answer_text
                return qa_pairs
            else:
                logger.warning("Failed to parse QA pairs from LLM response for reverse generation")
                return []
        else:
            logger.warning("Empty or invalid response from LLM for reverse generation")
            return []
            
    except Exception as e:
        logger.error(f"Error generating reverse QA pairs: {str(e)}")
        return []

def demo_llm_generator() -> None:
    """Demonstrate the LLM generator functionality with examples.
    
    This function shows how to use the main components of the LLM generator:
    1. Checking LiteLLM availability
    2. Generating code-related QA pairs
    3. Generating markdown-related QA pairs
    4. Generating reverse QA pairs
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("LLM Generator Demo")
        logger.info("=================")
        
        # Check LiteLLM availability
        llm_available = check_litellm_available()
        logger.info(f"LiteLLM available: {llm_available}")
        
        if not llm_available:
            logger.warning("LiteLLM is not available. Demo will show example responses only.")
            
            # Show example responses
            logger.info("\nExample Code QA Pairs:")
            example_code_qa = [
                {
                    "question": "What is the purpose of the calculate_average function?",
                    "answer": "The calculate_average function calculates the arithmetic mean of a list of numbers. It returns the sum of all numbers divided by the count of numbers. If the input list is empty, it returns 0 to avoid division by zero errors."
                },
                {
                    "question": "How is the calculate_average function implemented?",
                    "answer": "The calculate_average function checks if the input list is empty using 'if not numbers:' and returns 0 in that case. Otherwise, it calculates the sum of all numbers using Python's built-in sum() function and divides by the length of the list, returning the result."
                }
            ]
            
            for i, qa_pair in enumerate(example_code_qa, 1):
                logger.info(f"\nExample {i}:")
                logger.info(f"Q: {qa_pair['question']}")
                logger.info(f"A: {qa_pair['answer']}")
                
            logger.info("\nDemo completed with examples only.")
            return
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Create a sample Python file
            logger.info("\n1. Generating QA pairs from Python code:")
            python_file = temp_path / "utils.py"
            with open(python_file, "w") as f:
                f.write("""
def calculate_average(numbers):
    \"\"\"
    Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers to average
        
    Returns:
        The average value
    \"\"\"
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class DataProcessor:
    \"\"\"A class for processing data collections.\"\"\"
    
    def __init__(self, data):
        \"\"\"Initialize with a data collection.\"\"\"
        self.data = data
    
    def process(self):
        \"\"\"Process the data and return results.\"\"\"
        return [item * 2 for item in self.data]
""")
            
            # Read the file content
            with open(python_file, "r") as f:
                python_content = f.read()
            
            # Generate QA pairs for a function
            logger.info("\nGenerating QA pairs for 'calculate_average' function:")
            function_qa_pairs = generate_code_related_qa_pairs(
                python_content, 
                "calculate_average", 
                "function",
                temperature=0.3
            )
            
            if function_qa_pairs:
                for i, qa_pair in enumerate(function_qa_pairs, 1):
                    logger.info(f"\nPair {i}:")
                    logger.info(f"Q: {qa_pair['question']}")
                    logger.info(f"A: {qa_pair['answer'][:150]}..." if len(qa_pair['answer']) > 150 else f"A: {qa_pair['answer']}")
            else:
                logger.warning("No function QA pairs generated")
            
            # Generate QA pairs for a class
            logger.info("\nGenerating QA pairs for 'DataProcessor' class:")
            class_qa_pairs = generate_code_related_qa_pairs(
                python_content, 
                "DataProcessor", 
                "class",
                temperature=0.4
            )
            
            if class_qa_pairs:
                for i, qa_pair in enumerate(class_qa_pairs, 1):
                    logger.info(f"\nPair {i}:")
                    logger.info(f"Q: {qa_pair['question']}")
                    logger.info(f"A: {qa_pair['answer'][:150]}..." if len(qa_pair['answer']) > 150 else f"A: {qa_pair['answer']}")
            else:
                logger.warning("No class QA pairs generated")
            
            # 2. Create a sample Markdown file
            logger.info("\n2. Generating QA pairs from Markdown:")
            markdown_file = temp_path / "readme.md"
            with open(markdown_file, "w") as f:
                f.write("""# Data Processing Library

This library provides utilities for processing data collections.

## Installation

Install the package using pip:

```bash
pip install data-processor
```

## Usage

```python
from data_processor import calculate_average, DataProcessor

# Calculate average
result = calculate_average([1, 2, 3, 4, 5])
print(f"Average: {result}")  # Output: Average: 3.0

# Process data
processor = DataProcessor([1, 2, 3])
processed = processor.process()
print(processed)  # Output: [2, 4, 6]
```
""")
            
            # Read the file content
            with open(markdown_file, "r") as f:
                markdown_content = f.read()
            
            # Generate QA pairs for markdown
            logger.info("\nGenerating QA pairs from markdown content:")
            markdown_qa_pairs = generate_markdown_related_qa_pairs(
                markdown_content, 
                "README",
                temperature=0.5
            )
            
            if markdown_qa_pairs:
                for i, qa_pair in enumerate(markdown_qa_pairs, 1):
                    logger.info(f"\nPair {i}:")
                    logger.info(f"Q: {qa_pair['question']}")
                    logger.info(f"A: {qa_pair['answer'][:150]}..." if len(qa_pair['answer']) > 150 else f"A: {qa_pair['answer']}")
            else:
                logger.warning("No markdown QA pairs generated")
            
            # 3. Generate reverse QA pairs
            logger.info("\n3. Generating reverse QA pairs:")
            
            # Create a sample answer
            sample_answer = """
The `calculate_average` function takes a list of numbers as input and returns their arithmetic mean. 
It first checks if the list is empty using `if not numbers:` and returns 0 in that case to avoid division by zero errors.
If the list contains numbers, it calculates the sum using Python's built-in `sum()` function and divides by the
length of the list using `len(numbers)`.

Example usage:
```python
result = calculate_average([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```
"""
            
            # Generate reverse QA pairs
            reverse_qa_pairs = generate_reverse_qa_pairs(
                sample_answer,
                num_questions=2,
                temperature=0.4
            )
            
            if reverse_qa_pairs:
                for i, qa_pair in enumerate(reverse_qa_pairs, 1):
                    logger.info(f"\nPair {i}:")
                    logger.info(f"Q: {qa_pair['question']}")
                    logger.info(f"A: {qa_pair['answer'][:150]}..." if len(qa_pair['answer']) > 150 else f"A: {qa_pair['answer']}")
            else:
                logger.warning("No reverse QA pairs generated")
        
        logger.info("\nLLM Generator Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in LLM generator demo: {e}")

if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_llm_generator()
    
    # Process command line arguments if provided
    if len(sys.argv) > 1:
        try:
            # Check for specific command line options
            if sys.argv[1] == "--check-availability":
                available = check_litellm_available()
                print(f"LiteLLM available: {available}")
                sys.exit(0)
                
            # Generate QA pairs for a file
            if sys.argv[1] == "--generate" and len(sys.argv) > 2:
                file_path = sys.argv[2]
                output_path = sys.argv[3] if len(sys.argv) > 3 else "qa_pairs.json"
                
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    sys.exit(1)
                
                # Read the file
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Determine content type and generate QA pairs
                qa_pairs = []
                if file_path.endswith(".py") or file_path.endswith(".js") or file_path.endswith(".java"):
                    # Extract function names for Python
                    if file_path.endswith(".py"):
                        function_matches = list(re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content))
                        function_names = [match.group(1) for match in function_matches]
                        
                        for func_name in function_names[:3]:  # Limit to 3 functions
                            func_pairs = generate_code_related_qa_pairs(content, func_name, "function")
                            qa_pairs.extend(func_pairs)
                    
                    # Generic code QA pairs
                    qa_pairs.append({
                        "question": f"What does the code in {os.path.basename(file_path)} do?",
                        "answer": f"The code implements various functionality including...\n\n```\n{content[:500]}...\n```"
                    })
                    
                elif file_path.endswith(".md") or file_path.endswith(".txt"):
                    # Generate markdown QA pairs
                    md_pairs = generate_markdown_related_qa_pairs(content, os.path.basename(file_path))
                    qa_pairs.extend(md_pairs)
                
                # Save QA pairs
                with open(output_path, "w") as f:
                    json.dump(qa_pairs, f, indent=2)
                
                logger.info(f"Generated {len(qa_pairs)} QA pairs and saved to {output_path}")
                
            else:
                logger.error("Invalid command line arguments")
            logger.info("Usage:")
            logger.info("  --check-availability: Check if LiteLLM is available")
            logger.info("  --generate <file_path> [output_path]: Generate QA pairs for a file")
            
        except Exception as e:
            logger.error(f"Error processing command line arguments: {e}") 