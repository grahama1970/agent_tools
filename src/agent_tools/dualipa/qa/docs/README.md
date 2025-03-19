# DuaLipa LLM Q&A Generation Module

This document defines the standardized interface and implementation for the DuaLipa LLM Q&A Generation module, which generates question-answer pairs from structured JSON content produced by the DuaLipa Extraction Module. The output is formatted for use with Unsloth to create an adapter.

## Workflow Overview

The LLM Q&A Generation Module processes the JSON object produced by the Extraction Module following these steps:

1. **Input Validation and Normalization:** Validates and cleans the input JSON for consistency
2. **Input JSON Parsing:** Reads the structured extraction JSON
3. **Temperature Iteration:** Generates Q&A pairs across multiple temperature settings
4. **Bidirectional Generation:** Creates both forward and reverse Q&A pairs (addressing the reversal curse)
5. **Reasoning Enrichment:** Adds a single reasoning string with reconsideration moments
6. **Deduplication:** Removes duplicate or near-duplicate Q&A pairs
7. **Output Validation:** Ensures Q&A pairs meet quality standards
8. **Output Serialization:** Produces the final validated JSON with enriched metadata

## Proposed File Structure
src/agent_tools/dualipa/
├── qa/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── qa_models.py  
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── generation.py
│   │   └── reversal.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── deduplication.py
│   ├── processor.py
│   └── cli.py
│
├── tests/
│   └── qa/  # Tests are here instead of qa_generation
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_models.py
│       ├── test_llm/
│       │   ├── __init__.py
│       │   ├── test_cache.py
│       │   ├── test_generation.py
│       │   └── test_reversal.py
│       ├── test_utils/
│       │   ├── __init__.py
│       │   ├── test_validation.py
│       │   └── test_deduplication.py
│       └── test_processor.py


## Implementation Components

### 1. Pydantic Models for Structured Data

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Optional

class QAPair(BaseModel):
    """Model for a single question-answer pair with merged reasoning."""
    uuid: str = Field(..., description="Unique identifier")
    source_section_uuid: str = Field(..., description="UUID of the source section")
    question: str = Field(..., description="The generated question")
    reasoning: str = Field(..., description="Merged reasoning content including pivots")
    answer: str = Field(..., description="The generated answer")
    temperature_used: float = Field(..., description="Temperature setting used for generation")
    direction: str = Field(default="forward", description="Direction of generation: 'forward' (Q→A) or 'reverse' (A→Q)")
    confidence_score: float = Field(default=1.0, description="Estimated confidence in pair quality (0.0-1.0)")
    complexity_level: str = Field(default="medium", description="Complexity: 'low', 'medium', or 'high'")

class QAResponse(BaseModel):
    """Model for the complete response from the Q&A generation module."""
    qa_pairs: List[QAPair] = Field(..., description="Generated QA pairs")
    generation_metadata: Dict[str, Any] = Field(..., description="Metadata about the generation process")
```

### 2. Input Validation and Normalization

```python
import logging
logging.basicConfig(level=logging.INFO)

def validate_input_json(data: Dict[str, Any]) -> bool:
    """Validates and normalizes the input JSON from the Extraction Module."""
    if "sections" not in data or not isinstance(data["sections"], list):
        raise ValueError("Input JSON must contain a 'sections' list")
    valid_sections = []
    for section in data["sections"]:
        if not all(k in section for k in ["uuid", "type", "content"]):
            logging.warning(f"Skipping section {section.get('uuid', 'unknown')}: missing required fields")
            continue
        if not section["content"].strip():
            logging.warning(f"Skipping empty section {section['uuid']}")
            continue
        valid_sections.append(section)
    data["sections"] = valid_sections
    return len(valid_sections) > 0
```

### 3. Temperature Iteration Implementation

```python
import asyncio
from agent_tools.cursor_rules.llm.litellm_call import litellm_call
from agent_tools.cursor_rules.llm.initialize_litellm_cache import initialize_litellm_cache

async def iterate_temperatures(section: Dict[str, Any], 
                              prompt: str,
                              temperature_values: list = [0.3, 0.4, 0.5, 0.6, 0.7]) -> Dict[float, Any]:
    responses = {}
    for temp in temperature_values:
        config = {
            "llm_config": {
                "api_base": "http://api.example.com/v1/completions",
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert technical writer generating question-answer pairs with reasoning."},
                    {"role": "user", "content": f"{prompt}\n\nTemperature: {temp}"}
                ],
                "response_format": QAPair,
                "temperature": temp,
                "stream": False,
                "caching": True,
            },
            "directories": {}
        }
        try:
            response = await litellm_call(config)
            responses[temp] = response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Failed to generate Q&A at temperature {temp}: {str(e)}")
            responses[temp] = None
    return responses
```

### 4. Reversal Curse Handling

```python
async def generate_reversed_qa_pairs(qa_pairs: List[QAPair]) -> List[QAPair]:
    reversed_pairs = []
    for pair in qa_pairs:
        reverse_prompt = f"""
        Given this answer: "{pair.answer}"
        
        Generate a question that would produce this exact answer.
        The question should be different from the original question: "{pair.question}"
        
        Include reasoning as a single paragraph showing how you formulate the reversed question.
        Include at least one "Oh wait?!" moment in your reasoning.
        
        Output as a JSON object with "question", "reasoning", and "answer" fields.
        """
        config = {
            "llm_config": {
                "api_base": "http://api.example.com/v1/completions",
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert at bidirectional question-answer generation."},
                    {"role": "user", "content": reverse_prompt}
                ],
                "temperature": 0.7,
                "stream": False,
                "caching": True,
            },
            "directories": {}
        }
        try:
            response = await litellm_call(config)
            content = response.choices[0].message.content
            data = json.loads(content)
            reversed_pair = QAPair(
                uuid=str(uuid.uuid4()),
                source_section_uuid=pair.source_section_uuid,
                question=data["question"],
                reasoning=data["reasoning"],
                answer=pair.answer,
                temperature_used=0.7,
                direction="reverse",
                confidence_score=0.9,
                complexity_level="medium"
            )
            reversed_pairs.append(reversed_pair)
        except Exception as e:
            logging.warning(f"Error processing reversed pair for {pair.uuid}: {str(e)}")
    return reversed_pairs
```

### 5. Generating Markdown Q&A Pairs with Reasoning

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
async def generate_markdown_qa_pairs(section: Dict[str, Any], temperature: float = 0.4) -> List[QAPair]:
    content = section.get("content", "")
    focus = section.get("extraction_focus", "general")
    instructions = section.get("summary_instructions", "")
    
    prompt = f"""
    You are an expert technical writer tasked with generating 3 high-quality Q&A pairs based on the markdown content provided.
    
    Extraction Focus: "{focus}"
    Summary Instructions: "{instructions}"
    
    CONTENT:
    ```
    {content}
    ```
    
    For each question-answer pair:
    1. Generate a clear, specific question based on the content
    2. Provide reasoning as a single paragraph showing how you derive the answer, including at least one "Oh wait?!" moment
    3. Provide a concise, accurate answer
    
    Output as JSON array of {{"question": "...", "reasoning": "...", "answer": "..."}}.
    """
    try:
        config = {
            "llm_config": {
                "api_base": "http://api.example.com/v1/completions",
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert technical writer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "stream": False,
                "caching": True,
            },
            "directories": {}
        }
        response = await litellm_call(config)
        content = response.choices[0].message.content
        
        json_str = content.strip()
        if "```
            json_str = json_str.split("```json")[1].split("```
        elif "```" in json_str:
            json_str = json_str.split("```
        qa_list = json.loads(json_str)
        
        result = []
        for qa in qa_list:
            reasoning_len = len(qa["reasoning"].split())
            result.append(QAPair(
                uuid=str(uuid.uuid4()),
                question=qa["question"],
                reasoning=qa["reasoning"],
                answer=qa["answer"],
                source_section_uuid=section["uuid"],
                temperature_used=temperature,
                direction="forward",
                confidence_score=1.0 if reasoning_len >= 15 else 0.8,
                complexity_level="high" if reasoning_len > 20 else "medium"
            ))
        return result
    except Exception as e:
        logging.warning(f"Error generating QA pairs for section {section['uuid']}: {str(e)}")
        raise
```

### 6. Deduplication

```
def deduplicate_qa_pairs(pairs: List[QAPair]) -> List[QAPair]:
    seen = set()
    unique_pairs = []
    for pair in pairs:
        key = (pair.question.lower(), pair.answer.lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    return unique_pairs
```

### 7. Validation

```
def validate_qa_pair(pair: QAPair) -> bool:
    if not pair.question.strip().endswith("?"):
        return False
    if "Oh wait?!" not in pair.reasoning or len(pair.reasoning.split())  None:
    initialize_litellm_cache()
    with open(extraction_json_path, 'r') as f:
        data = json.load(f)
    
    if not validate_input_json(data):
        raise ValueError("No valid sections found in input JSON")
    
    all_qa_pairs = []
    start_time = time.time()
    temperature_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    async def batch_process_sections(sections: List[Dict], temp_range: List[float]) -> List[QAPair]:
        tasks = []
        for section in sections:
            for temp in temp_range:
                if section["type"] == "documentation":
                    tasks.append(generate_markdown_qa_pairs(section, temperature=temp))
                else:
                    tasks.append(generate_code_qa_pairs(section, temperature=temp))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logging.warning(f"Batch task failed: {str(result)}")
            else:
                batch_pairs.extend(result)
        return batch_pairs
    
    sections = data.get("sections", [])
    all_qa_pairs = await batch_process_sections(sections, temperature_range)
    reversed_pairs = await generate_reversed_qa_pairs(all_qa_pairs)
    all_qa_pairs.extend(reversed_pairs)
    all_qa_pairs = deduplicate_qa_pairs(all_qa_pairs)
    validated_pairs = [pair for pair in all_qa_pairs if validate_qa_pair(pair)]
    
    output = {
        "qa_pairs": [qa.dict() for qa in validated_pairs],
        "generation_metadata": {
            "version": "1.0.0",
            "model_used": "gpt-4-turbo",
            "temperature_range": temperature_range,
            "duration_seconds": round(time.time() - start_time, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_extraction_file": extraction_json_path,
            "total_qa_pairs": len(validated_pairs),
            "forward_pairs": len([p for p in validated_pairs if p.direction == "forward"]),
            "reverse_pairs": len([p for p in validated_pairs if p.direction == "reverse"]),
            "error_count": sum(1 for r in logging.getLogger().handlers.buffer if r.levelno >= logging.WARNING)
        }
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Generated {len(validated_pairs)} validated Q&A pairs. Output saved to {output_path}")
```

## Example Output

```
{
  "qa_pairs": [
    {
      "uuid": "b7c8d9e0-f1a2-4b3c-95d6-e7f8a9b0c1d2",
      "source_section_uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
      "question": "What languages does the code extractor support?",
      "reasoning": "Looking at the Features section, it lists supported languages. Oh wait?! The bullet points specifically mention Python, JavaScript, and TypeScript. After reviewing the content again, I confirm only these are mentioned.",
      "answer": "The code extractor supports Python, JavaScript, and TypeScript.",
      "temperature_used": 0.4,
      "direction": "forward",
      "confidence_score": 1.0,
      "complexity_level": "medium"
    }
  ],
  "generation_metadata": {
    "version": "1.0.0",
    "model_used": "gpt-4-turbo",
    "temperature_range": [0.3, 0.4, 0.5, 0.6, 0.7],
    "duration_seconds": 45.23,
    "timestamp": "2025-03-19T11:48:32Z",
    "source_extraction_file": "extracted_repo.json",
    "total_qa_pairs": 58,
    "forward_pairs": 35,
    "reverse_pairs": 23,
    "error_count": 2
  }
}
```

## Usage Notes and Best Practices

1. **Cache Initialization**:
   ```
   initialize_litellm_cache(cache_type="redis")
   ```

2. **Error Handling**:
   - Use logging to track warnings and errors.

3. **Temperature Exploration**:
   - Use a range of temperatures (0.3-0.7) for diversity.

4. **Bidirectional Generation**:
   - Generate both forward and reverse pairs.

5. **Reasoning Quality**:
   - Ensure reasoning includes "Oh wait?!" and is at least 15 words.

6. **Validation**:
   - Enforce: question mark, 15+ word reasoning, 5+ word answers, non-identical Q&A.
