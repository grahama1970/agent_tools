#!/usr/bin/env python3
"""
Basic integration test for LiteLLM functionality.

Documentation References:
- LiteLLM: https://docs.litellm.ai/
- OpenAI Vision: https://platform.openai.com/docs/guides/vision
- Redis Caching: https://redis.io/docs/manual/
- Pydantic v2: https://docs.pydantic.dev/latest/

This test verifies that LiteLLM works correctly with:
1. Basic completion calls
2. Redis caching
3. Retry mechanisms
4. Response validation
5. Multimodal (vision) capabilities

Similar to test_hybrid_search.py, this uses real calls without mocks.
"""

import os
import sys
import asyncio
import pytest
from loguru import logger
from pydantic import BaseModel, Field, field_validator
import copy
import base64
from pathlib import Path
import redis
import litellm

# Import the components we need to test
from agent_tools.cursor_rules.llm.litellm_call import litellm_call
from agent_tools.cursor_rules.llm.retry_llm_call import retry_llm_call
from agent_tools.cursor_rules.llm.initialize_litellm_cache import initialize_litellm_cache, test_litellm_cache
from agent_tools.cursor_rules.llm.multimodal_utils import is_multimodal, format_multimodal_messages
from agent_tools.cursor_rules.utils.image_utils import process_image_input

class PersonInfo(BaseModel):
    """A model representing basic information about a person"""
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="The person's age in years")
    occupation: str = Field(..., description="The person's current job or role")
    
    @field_validator('age')
    def validate_age(cls, v):
        if not 0 <= v <= 150:
            raise ValueError('Age must be between 0 and 150')
        return v

async def test_basic_litellm_completion():
    """
    Test basic LiteLLM completion without any caching or retries.
    This is the simplest possible test to verify LiteLLM works.
    """
    # Initialize cache first
    initialize_litellm_cache()
    
    # Simple completion request
    llm_config = {
        "llm_config": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            "temperature": 0.2,
            "caching": True
        }
    }
    
    # Make the call
    logger.info("Making basic LiteLLM call...")
    response = await litellm_call(llm_config)
    
    # Basic validation
    assert response is not None
    assert response.choices[0].message.content.strip() == "4"
    logger.info("Basic LiteLLM call successful")
    
    # Test cache hit
    logger.info("Testing cache hit...")
    cached_response = await litellm_call(llm_config)
    assert cached_response.choices[0].message.content == response.choices[0].message.content
    assert cached_response._hidden_params.get("cache_hit") is True
    logger.info("Cache hit verified")

async def test_litellm_with_validation():
    """
    Test LiteLLM with response validation using retry_llm_call.
    """
    initialize_litellm_cache()
    
    llm_config = {
        "llm_config": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You must respond with only a single number between 1 and 10, with no additional text."},
                {"role": "user", "content": "Generate a number between 1 and 10"}
            ],
            "temperature": 0.2,
            "caching": True
        }
    }
    
    # Define validation strategy
    def validate_number(response):
        try:
            num = int(response.choices[0].message.content.strip())
            return 1 <= num <= 10 or "Number must be between 1 and 10"
        except ValueError:
            return "Response must be a number"
    
    logger.info("Testing LiteLLM with validation...")
    response = await retry_llm_call(
        llm_call=litellm_call,
        llm_config=llm_config,
        validation_strategies=[validate_number],
        max_retries=3
    )
    
    # Verify response meets validation criteria
    num = int(response.choices[0].message.content.strip())
    assert 1 <= num <= 10
    logger.info(f"Validated response received: {num}")

def test_litellm_cache_persistence():
    """
    Test that LiteLLM cache persists between calls and handles cache misses/hits correctly.
    """
    # Clear Redis cache first
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.flushall()
    
    # First, run the test_litellm_cache function from initialize_litellm_cache
    logger.info("Testing LiteLLM cache with provided test function...")
    test_litellm_cache()
    
    # Base config that we'll copy to ensure we don't modify the original
    base_messages = [
        {"role": "system", "content": "You must respond with only the city name, no other text."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # First call should miss cache
    logger.info("Making first call (should miss cache)...")
    response1 = litellm.completion(
        model="gpt-4o-mini",
        messages=base_messages,
        temperature=0.0,  # Use 0 temperature for deterministic responses
        cache={"no-cache": False},
    )
    assert response1.choices[0].message.content.strip().lower() == "paris"
    assert not response1._hidden_params.get("cache_hit", False)
    
    # Second call with same config should hit cache
    logger.info("Making second call (should hit cache)...")
    response2 = litellm.completion(
        model="gpt-4o-mini",
        messages=base_messages,
        temperature=0.0,
        cache={"no-cache": False},
    )
    assert response2._hidden_params.get("cache_hit") is True
    assert response2.choices[0].message.content == response1.choices[0].message.content
    
    # Different question should miss cache
    logger.info("Making call with different question (should miss cache)...")
    different_messages = [
        {"role": "system", "content": "You must respond with only the city name, no other text."},
        {"role": "user", "content": "What is the capital of Spain?"}
    ]
    response3 = litellm.completion(
        model="gpt-4o-mini",
        messages=different_messages,
        temperature=0.0,
        cache={"no-cache": False},
    )
    assert not response3._hidden_params.get("cache_hit", False)
    assert response3.choices[0].message.content.strip().lower() == "madrid"
    
    logger.info("Cache persistence test completed successfully")

async def test_litellm_structured_validation():
    """
    Test LiteLLM with structured response validation using a Pydantic model.
    This test verifies that:
    1. The model initially returns an invalid response
    2. The validation catches the incorrect response and adds correction message
    3. The retry mechanism makes another call with the updated messages array
    4. The final response meets all validation criteria
    """
    initialize_litellm_cache()
    
    base_config = {
        "llm_config": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that returns information about people. You MUST respond with a valid JSON object containing name (string), age (integer), and occupation (string) fields."},
                {"role": "user", "content": "Tell me about John who is twenty five years old and works as an engineer."}
            ],
            "temperature": 0.2,  # Lower temperature for more consistent output
            "caching": False  # Disable caching as we're testing validation
        }
    }
    
    # Track number of calls made
    call_count = 0
    original_litellm_call = litellm_call
    
    async def counting_litellm_call(config):
        nonlocal call_count
        call_count += 1
        # Print current state of messages array before each call
        print(f"\nCall #{call_count} - Current messages array:")
        for msg in config["llm_config"]["messages"]:
            print(f"{msg['role'].upper()}: {msg['content']}")
        
        # Make the call and capture the response
        response = await original_litellm_call(config)
        
        # Add the model's response to the message history
        config["llm_config"]["messages"].append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        
        return response
    
    # Define validation strategies that provide clear correction instructions
    def validate_json_structure(response):
        try:
            content = response.choices[0].message.content
            # Try to parse as PersonInfo model
            person = PersonInfo.model_validate_json(content)
            return True
        except Exception as e:
            return ("Your response must be a valid JSON object. Please format your response like this: "
                   '{"name": "John", "age": 25, "occupation": "engineer"}')
    
    def validate_age_format(response):
        try:
            content = response.choices[0].message.content
            person = PersonInfo.model_validate_json(content)
            return True
        except Exception as e:
            return "The age field must be a number (not a string). Please provide the age as an integer."
    
    logger.info("Testing LiteLLM with structured validation...")
    
    # Store initial message count
    initial_message_count = len(base_config["llm_config"]["messages"])
    
    response = await retry_llm_call(
        llm_call=counting_litellm_call,  # Use our wrapped version that counts calls
        llm_config=base_config,
        validation_strategies=[validate_json_structure, validate_age_format],
        max_retries=3
    )
    
    # Verify multiple calls were made (at least one retry)
    assert call_count > 1, f"Expected multiple calls due to validation failures, but got {call_count} calls"
    
    # Verify messages were added between calls
    final_messages = base_config["llm_config"]["messages"]
    assert len(final_messages) > initial_message_count, (
        f"Expected messages to be added for validation feedback. Started with {initial_message_count}, "
        f"ended with {len(final_messages)}"
    )
    
    # Verify the structure of added messages
    added_messages = final_messages[initial_message_count:]
    print("\nValidation and correction flow:")
    print("--------------------------------")
    for msg in added_messages:
        print(f"{msg['role'].upper()}: {msg['content']}\n")
    
    # There should be at least:
    # 1. First model response (invalid)
    # 2. Validation error message
    # 3. Second model response (valid)
    assert len(added_messages) >= 3, "Expected at least one model response, one validation message, and one corrected response"
    
    # Verify message sequence
    assert any(msg["role"] == "assistant" and "John" in msg["content"] for msg in added_messages), "No model response found"
    assert any(msg["role"] == "assistant" and "validation" in msg["content"].lower() for msg in added_messages), "No validation message found"
    
    # Verify the final response is valid
    assert response is not None
    content = response.choices[0].message.content
    person = PersonInfo.model_validate_json(content)
    assert person.name == "John"
    assert person.age == 25
    assert "engineer" in person.occupation.lower()
    
    logger.info(f"Structured validation test completed successfully after {call_count} calls")

async def test_litellm_multimodal():
    """Test that multimodal LLM calls work correctly with image input."""
    # Initialize cache (synchronous function, no await needed)
    initialize_litellm_cache()

    # Get the absolute path to the image and image directory
    project_root = Path(__file__).parent.parent.parent
    image_path = project_root / "images" / "spartan.png"
    image_directory = project_root / "images" / "processed"
    assert image_path.exists(), f"Image not found at {image_path}"

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Configure the LLM call with the exact same format as OpenAI example
    llm_config = {
        "llm_config": {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }],
            "temperature": 0.2,
            "max_tokens": 300,
            "caching": False  # Disable caching for multimodal queries
        }
    }
    
    logger.info("Testing multimodal LLM call...")
    response = await litellm_call(llm_config)
    
    # Verify response
    assert response is not None
    content = response.choices[0].message.content.lower()
    logger.info(f"Response content: {content}")
    
    # Check for relevant keywords in response
    relevant_keywords = ["sparta", "spartan", "ancient", "greece", "greek", "warrior"]
    matching_keywords = [word for word in relevant_keywords if word in content]
    assert matching_keywords, (
        f"Response should mention at least one of: {relevant_keywords}. "
        f"Got response: {content}"
    )
    
    logger.info("Multimodal test completed successfully")

if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"])) 