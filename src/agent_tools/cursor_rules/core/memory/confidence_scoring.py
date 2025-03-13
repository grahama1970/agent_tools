#!/usr/bin/env python3
"""
Confidence Scoring Module for Agent Memory System

This module implements the confidence scoring functionality for the agent memory system.
It uses LiteLLM to evaluate confidence in facts, allowing the system to update
confidence scores based on new evidence or context.

Documentation references:
- LiteLLM: https://docs.litellm.ai/docs/
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
- Pydantic: https://docs.pydantic.dev/latest/
"""

import litellm
import os
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import asyncio

from agent_tools.cursor_rules.core.memory.agent_memory import AgentMemorySystem, MemoryFact
from agent_tools.cursor_rules.llm.litellm_call import litellm_call
from agent_tools.cursor_rules.llm.retry_llm_call import retry_llm_call
from agent_tools.cursor_rules.llm.initialize_litellm_cache import initialize_litellm_cache
from agent_tools.cursor_rules.utils.helpers.file_utils import load_env_file

# Default model
DEFAULT_MODEL = "gpt-4o-mini"

# Initialize LiteLLM cache
initialize_litellm_cache()

# Default prompt template for confidence evaluation
DEFAULT_CONFIDENCE_PROMPT = """
You are an expert evaluator of factual information. Your task is to evaluate the confidence
we should have in a given fact based on the evidence provided.

FACT: {fact}

EVIDENCE:
{evidence}

Please evaluate the confidence we should have in the fact on a scale from 0.0 to 1.0, where:
- 0.0 means completely false or contradicted by evidence
- 0.5 means uncertain or mixed evidence
- 1.0 means completely true and fully supported by evidence

Provide your assessment by analyzing the evidence in relation to the fact.
First, analyze whether the evidence supports, contradicts, or is neutral toward the fact.
Then, consider the reliability, recency, and comprehensiveness of the evidence.
Finally, give your numerical confidence score between 0.0 and 1.0.

CONFIDENCE SCORE: 
"""

# --- New Function: create_association ---
def create_association(
    memory_system: AgentMemorySystem,
    fact_id_1: Union[str, Dict[str, Any]],
    fact_id_2: Union[str, Dict[str, Any]],
    association_type: str = "related"
) -> None:
    """
    Create an association between two facts. The fact identifiers can be passed either as strings or as fact documents.
    If a fact document is provided, extract the '_key' from the 'new' key.
    """
    # Extract fact IDs from fact documents if necessary
    if isinstance(fact_id_1, dict) and "new" in fact_id_1:
        fact_id_1 = fact_id_1["new"]["_key"]
    if isinstance(fact_id_2, dict) and "new" in fact_id_2:
        fact_id_2 = fact_id_2["new"]["_key"]

    aql_query = """
    FOR fact1 IN facts
        FILTER fact1._key == @fact_id_1
        FOR fact2 IN facts
            FILTER fact2._key == @fact_id_2
            INSERT { _from: fact1._id, _to: fact2._id, association_type: @association_type }
            INTO associations
    """
    bind_vars = {
        "fact_id_1": fact_id_1,
        "fact_id_2": fact_id_2,
        "association_type": association_type
    }
    memory_system.db.aql.execute(aql_query, bind_vars=bind_vars)
# --- End of create_association ---

class ConfidenceEvaluation(BaseModel):
    """Response model for confidence evaluation."""
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Explanation for the confidence score")

class EvidenceAnalysis(BaseModel):
    """Response model for evidence analysis."""
    support_score: float = Field(..., description="Score indicating evidence support")
    contradiction_score: float = Field(..., description="Score indicating contradictions")
    uncertainty_factors: List[str] = Field(default_factory=list, description="Factors contributing to uncertainty")

class ConfidenceUpdate(BaseModel):
    """Response model for confidence updates."""
    fact_id: str = Field(..., description="ID of the updated fact")
    original_confidence: float = Field(..., description="Original confidence score")
    new_confidence: float = Field(..., description="New confidence score")
    evidence: List[str] = Field(..., description="Evidence used for update")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of update")
    reasoning: str = Field(..., description="Reasoning for confidence update")

async def evaluate_confidence(
    fact_content: str,
    evidence: List[str],
    model: str = DEFAULT_MODEL,
) -> ConfidenceEvaluation:
    """
    Evaluate confidence in a fact using LiteLLM with proper retry handling.
    """
    system_message = (
        "You are an expert evaluator of factual information. "
        "Analyze the evidence and provide a confidence score for the given fact."
    )

    formatted_evidence = "\n".join([f"{i+1}. {item}" for i, item in enumerate(evidence)])
    
    user_message = (
        f"Evaluate the confidence we should have in this fact based on the evidence:\n\n"
        f"FACT: {fact_content}\n\n"
        f"EVIDENCE:\n{formatted_evidence}\n\n"
        "Provide your confidence score (0.0 to 1.0) and reasoning."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    llm_config = {
        "llm_config": {  # Wrap in llm_config as expected by litellm_call
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": ConfidenceEvaluation,
            "caching": True  # Use caching as configured by initialize_litellm_cache
        }
    }

    try:
        response = await retry_llm_call(
            llm_call=litellm_call,
            llm_config=llm_config,
            validation_strategies=[
                lambda x: 0.0 <= x.choices[0].message.content.confidence <= 1.0 or "Confidence must be between 0.0 and 1.0",
                lambda x: len(x.choices[0].message.content.reasoning) > 0 or "Reasoning must not be empty"
            ],
            max_retries=3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in confidence evaluation: {str(e)}")
        return ConfidenceEvaluation(confidence=0.5, reasoning=f"Error in evaluation: {str(e)}")

async def analyze_evidence(
    fact_content: str,
    evidence: List[str],
    model: str = DEFAULT_MODEL,
) -> EvidenceAnalysis:
    """
    Analyze evidence for a fact using LiteLLM with proper retry handling.
    """
    system_message = (
        "You are an expert evidence analyst. "
        "Analyze the evidence and assess support and contradictions for the given fact."
    )

    formatted_evidence = "\n".join([f"{i+1}. {item}" for i, item in enumerate(evidence)])
    
    user_message = (
        f"Analyze the evidence regarding this fact:\n\n"
        f"FACT: {fact_content}\n\n"
        f"EVIDENCE:\n{formatted_evidence}\n\n"
        "Provide support score (0.0-1.0), contradiction score (0.0-1.0), and list uncertainty factors."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    llm_config = {
        "llm_config": {  # Wrap in llm_config as expected by litellm_call
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": EvidenceAnalysis,
            "caching": True  # Use caching as configured by initialize_litellm_cache
        }
    }

    try:
        response = await retry_llm_call(
            llm_call=litellm_call,
            llm_config=llm_config,
            validation_strategies=[
                lambda x: 0.0 <= x.choices[0].message.content.support_score <= 1.0 or "Support score must be between 0.0 and 1.0",
                lambda x: 0.0 <= x.choices[0].message.content.contradiction_score <= 1.0 or "Contradiction score must be between 0.0 and 1.0"
            ],
            max_retries=3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in evidence analysis: {str(e)}")
        return EvidenceAnalysis(
            support_score=0.5,
            contradiction_score=0.5,
            uncertainty_factors=[f"Error in analysis: {str(e)}"]
        )

async def update_fact_confidence(
    memory_system,
    fact_id: str,
    evidence: List[str],
    model: str = DEFAULT_MODEL,
) -> ConfidenceUpdate:
    """
    Update the confidence score of a fact based on new evidence.
    
    Args:
        memory_system: The memory system instance
        fact_id: ID of the fact to update
        evidence: List of evidence strings to evaluate against
        model: LLM model to use for evaluation
    
    Returns:
        ConfidenceUpdate object with update results
    """
    # Get the fact
    fact = await memory_system.get_fact(fact_id)
    if not fact:
        raise ValueError(f"Fact with ID {fact_id} not found")

    # Evaluate confidence
    evaluation = await evaluate_confidence(
        fact_content=fact["content"],
        evidence=evidence,
        model=model
    )

    # Create update record
    update = ConfidenceUpdate(
        fact_id=fact_id,
        original_confidence=fact.get("confidence", 0.5),
        new_confidence=evaluation.confidence,
        evidence=evidence,
        reasoning=evaluation.reasoning
    )

    # Update the fact in the database
    await memory_system.update_fact_confidence(
        fact_id=fact_id,
        confidence=evaluation.confidence,
        update_record=update.dict()
    )

    return update

async def batch_confidence_evaluation(
    memory_system: AgentMemorySystem,
    query: str,
    evidence: List[str],
    limit: int = 10,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate confidence for multiple facts matching a query.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        query: Search query to find facts to evaluate
        evidence: List of evidence strings to evaluate against
        limit: Maximum number of facts to evaluate
        model: LLM model to use for evaluation
        api_key: API key for the LLM service (if required)
        
    Returns:
        List of evaluation results
    """
    if not memory_system.initialized:
        await memory_system.initialize()
    
    # Find facts matching the query
    facts = await memory_system.recall(query, limit=limit)
    
    results = []
    for fact in facts:
        try:
            result = await update_fact_confidence(
                memory_system=memory_system,
                fact_id=fact["_key"],
                evidence=evidence,
                model=model,
                api_key=api_key
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating fact {fact['_key']}: {str(e)}")
            results.append({
                "fact_id": fact["_key"],
                "error": str(e),
                "updated": False
            })
    
    return results

async def evaluate_contradictions(
    memory_system: AgentMemorySystem,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    contradiction_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find and evaluate facts with contradictions.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        model: LLM model to use for evaluation
        api_key: API key for the LLM service (if required)
        contradiction_threshold: Threshold for detecting contradictions
        
    Returns:
        List of facts with contradictions and evaluation results
    """
    if not memory_system.initialized:
        await memory_system.initialize()
    
    # Use knowledge_correction module to find facts with contradictions
    from agent_tools.cursor_rules.core.memory.knowledge_correction import analyze_contradictions
    
    contradictions = await analyze_contradictions(memory_system)
    
    results = []
    for item in contradictions:
        if item.get("contradiction_score", 0) >= contradiction_threshold:
            # Prepare evidence from alternatives
            alternatives = item.get("alternatives", [])
            evidence = [f"Alternative statement: {alt['content']} (confidence: {alt['confidence']})" 
                      for alt in alternatives]
            
            # Add the original fact as evidence
            evidence.append(f"Original statement: {item['content']} (confidence: {item['confidence']})")
            
            try:
                # Use evaluate_confidence with proper cache config
                evaluation = await evaluate_confidence(
                    fact_content=item['content'],
                    evidence=evidence,
                    model=model
                )
                
                results.append({
                    "fact_id": item["fact_id"],
                    "content": item["content"],
                    "alternatives": alternatives,
                    "original_confidence": item["confidence"],
                    "evaluated_confidence": evaluation.confidence,
                    "reasoning": evaluation.reasoning,
                    "contradiction_score": item.get("contradiction_score", 0)
                })
            except Exception as e:
                logger.error(f"Error evaluating contradiction for fact {item['fact_id']}: {str(e)}")
                continue
    
    return results

async def get_confidence_history(
    memory_system: AgentMemorySystem,
    fact_id: str
) -> List[Dict[str, Any]]:
    """
    Get the confidence history for a fact.
    
    Args:
        memory_system: The initialized AgentMemorySystem
        fact_id: ID of the fact to get history for
        
    Returns:
        List of confidence history entries, each containing confidence score,
        timestamp, evidence, and any notes
    """
    if not memory_system.initialized:
        await memory_system.initialize()
    
    # Get the fact
    fact = await memory_system.get_fact(fact_id)
    if not fact:
        raise ValueError(f"Fact with ID {fact_id} not found")
    
    # Get confidence history from the fact
    history = fact.get("confidence_history", [])
    
    # Add the current confidence as the latest entry if not already present
    latest_entry = {
        "confidence": fact["confidence"],
        "timestamp": fact.get("last_updated", datetime.now().isoformat()),
        "evidence": fact.get("latest_evidence", []),
        "notes": fact.get("latest_confidence_notes", None)
    }
    
    # Only add if the timestamp is different from the last entry
    if not history or history[-1]["timestamp"] != latest_entry["timestamp"]:
        history.append(latest_entry)
    
    return history
