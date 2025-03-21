# DuaLipa LLM Q&A Generation Module

This document defines a secure, enterprise-ready interface for generating question-answer pairs from structured content, with production-grade validation, error recovery, and compliance checks.

## Implementation Using Test-Driven Development

This module is being implemented using a strict TDD approach, following these steps:

1. **Start with MVP** - Begin with a minimal working solution that passes a smoke test
2. **Add Tests First** - Write tests for each new feature before implementation
3. **Minimal Implementation** - Implement only what's needed for tests to pass
4. **Follow Technical Patterns** - Adhere to project-specific patterns like textwrap.dedent()
5. **Iterative Expansion** - Build functionality in phases as outlined in task.md

See [agent_learnings.md](agent_learnings.md), [tdd_instructions.md](tdd_instructions.md), and [implementation_lessons.md](implementation_lessons.md) for detailed TDD guidance and lessons learned.

## Workflow Overview

```
%%{init: {'theme': 'neutral'}}%%
flowchart TD
    A[Start: Extraction JSON] --> B(Input Validation & Sanitization)
    B --> C{Valid?}
    C -->|Yes| D[Temperature Iteration\n0.3, 0.5, 0.7]
    C -->|No| Z[Log Error]
    D --> E(Bidirectional Generation\nForward + Reverse)
    E --> F(Reasoning Enrichment\n"Oh wait?!" required)
    F --> G(Deduplication\nExact + Semantic)
    G --> H{QA Spec Compliant?}
    H -->|Yes| I[Serialize Output]
    H -->|No| J[Adjust & Retry]
    I --> K[End: Validated QA JSON]
    
    style A fill:#4CAF50,stroke:#388E3C
    style K fill:#2196F3,stroke:#1976D2
    style Z fill:#F44336,stroke:#D32F2F
```

## Input/Output Specifications

### Expected Input Format
```
{
  "sections": [
    {
      "uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
      "type": "documentation",
      "content": "## Feature Overview\n...",
      "extraction_focus": "technical details",
      "summary_instructions": "Generate 3 QA pairs focusing on API usage"
    }
  ],
  "extraction_metadata": {
    "model_used": "gpt-4-turbo",
    "timestamp": "2025-03-19T14:49:00Z"
  }
}
```

### Output Format
```
{
  "qa_pairs": [
    {
      "uuid": "b7c8d9e0-f1a2-4b3c-95d6-e7f8a9b0c1d2",
      "question": "What languages does the code extractor support?",
      "answer": "Python, JavaScript, and TypeScript",
      "reasoning": "The documentation lists three languages... Oh wait?! The 'Advanced Features' section confirms no others are supported.",
      "source_section_uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
      "temperature_used": 0.5,
      "direction": "forward",
      "confidence_score": 0.92,
      "complexity_level": "medium"
    }
  ],
  "generation_metadata": {
    "model_used": "gpt-4-turbo",
    "temperature_range": [0.3, 0.7],
    "timestamp": "2025-03-19T15:22:00Z",
    "validation_checks": {
      "sanitized_inputs": true,
      "reverse_pairs_ratio": 0.23,
      "avg_reasoning_length": 18.7
    }
  }
}
```

## Implementation Components

### 1. Security & Validation
```python
from pydantic import BaseModel, Field
import bleach
import logging
import textwrap

class QAPair(BaseModel):
    question: str = Field(..., min_length=10, regex="^.*\?$")
    answer: str = Field(..., min_length=5)
    reasoning: str = Field(..., min_length=15, regex="Oh wait\?!")

def sanitize_input(content: str) -> str:
    """Sanitize HTML/JS and detect prompt injections"""
    cleaned = bleach.clean(content, tags=[], strip=True)
    if any(inj in cleaned.lower() for inj in ["ignore", "override", "system"]):
        raise ValueError("Potential prompt injection detected")
    return cleaned
```

### 2. Error Recovery & Retry
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def generate_with_fallback(prompt: str, config: dict):
    """Generate with circuit breaker and fallback model"""
    try:
        return await litellm_call(config)
    except APIError as e:
        logging.warning(f"API failure: {e}, retrying...")
        config["llm_config"]["model"] = "gpt-3.5-turbo"  # Fallback
        raise
```

### 3. Bidirectional Generation
```python
async def generate_reversed_pair(original: QAPair) -> QAPair:
    """Create reverse Q&A pair with validation"""
    reverse_prompt = textwrap.dedent(f"""
        Given answer: {original.answer}
        Generate a different question. Include "Oh wait?!" moment.
    """)
    
    response = await generate_with_fallback(reverse_prompt, {
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    })
    
    return QAPair(
        question=response['question'],
        answer=original.answer,
        reasoning=response['reasoning'],
        direction="reverse",
        source_section_uuid=original.source_section_uuid
    )
```

### 4. Semantic Deduplication
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_deduplicate(pairs: list[QAPair], threshold=0.85) -> list[QAPair]:
    """Remove similar QA pairs using cosine similarity"""
    # Always dedent multiline strings
    prompt = textwrap.dedent("""
        This is a deduplicated list of QA pairs.
    """).strip()
    
    embeddings = model.encode([f"{p.question} {p.answer}" for p in pairs])
    sim_matrix = np.inner(embeddings, embeddings)
    unique_indices = set()
    
    for i in range(len(pairs)):
        if i not in unique_indices:
            for j in np.where(sim_matrix[i] > threshold):
                unique_indices.add(j)
                
    return [pairs[i] for i in unique_indices]
```

## File Structure
```
src/agent_tools/dualipa/
├── qa/
│   ├── __init__.py
│   ├── models/
│   │   ├── qa_models.py       # Pydantic models
│   │   └── config.py          # Temperature ranges
│   ├── llm/
│   │   ├── generation.py      # Core logic
│   │   └── reversal.py        # Bidirectional handling
│   ├── utils/
│   │   ├── validation.py      # Schema checks
│   │   ├── security.py        # Sanitization
│   │   └── deduplication.py   # Semantic similarity
│   ├── processor.py           # Main pipeline
│   ├── cli.py                 # Command interface
│   └── docs/
│       ├── README.md          # This file
│       ├── agent_learnings.md # TDD lessons
│       └── tdd_instructions.md # TDD guidelines
```

## Production Features
1. **Input Sanitization**
   - HTML/JS removal
   - Prompt injection detection
   - Minimum length enforcement

2. **Error Recovery**
   - 3 retries with exponential backoff
   - GPT-3.5 Turbo fallback
   - Circuit breaker pattern

3. **Compliance**
   - PII detection in answers
   - License validation for generated content
   - EAR99 export control checks

4. **Validation**
   ```python
   def validate_qa_pair(pair: QAPair) -> bool:
       return all([
           "?" in pair.question,
           "Oh wait?!" in pair.reasoning,
           len(pair.answer.split()) >= 5,
           pair.confidence_score >= 0.5
       ])
   ```

## Usage Example
```python
from dualipa.qa import process_extraction_json

# Generate from cleaned input
output = await process_extraction_json(
    input_file="extracted_content.json",
    output_file="qa_pairs.json",
    temps=[0.3, 0.5, 0.7],
    min_reasoning_words=15
)

# Verify output
assert len(output.qa_pairs) >= 50
assert sum(1 for p in output.qa_pairs if p.direction == "reverse") >= 10
```

## CI/CD Requirements
```yaml
# .github/workflows/qa_ci.yml
jobs:
  test:
    steps:
      - name: Run Validation
        run: pytest --cov=dualipa.qa --cov-fail-under=95
      - name: Security Scan
        run: python -m dualipa.qa.utils.security audit qa_pairs_sample.json
```

## TDD Implementation Progress

- [x] Phase 0: Preparation and Documentation Foundation
  - [x] Task 0.1: Review Official Documentation
  - [x] Task 0.2: Establish Real-World Baseline (MVP)
- [x] Phase 1: Setup and Core Model Validation
  - [x] Task 1.1: Set Up Project Structure and Testing Framework
  - [x] Task 1.2: Implement and Test Pydantic Models with Constraints
  - [x] Task 1.3: Configuration Management
  - [x] Task 1.4: Security Validation
- [ ] Phase 2: Core Business Logic Components
  - [x] Task 2.1: Input Validation and Normalization
  - [x] Task 2.2: Temperature Iteration Logic with Rate-Limiting
  - [x] Task 2.3: Bidirectional Generation
  - [ ] Task 2.4: Reasoning Enrichment with Enhanced Error Recovery
  - [ ] Task 2.5: Advanced Deduplication with Tuning
  - [ ] Task 2.6: Output Validation
  - [ ] Task 2.7: Model Fallback Strategy
  - [x] Task 2.8: Constraint Tracking for Phase 2
- [ ] Phase 3: Infrastructure and Integration
- [ ] Phase 4: Optimization and CLI
- [ ] Phase 5: Final Verification and CI

**Current Status:** Working on Phase 2 - Core Business Logic Components

This README:
1. Specifies exact input/output formats with validation rules
2. Includes security measures (sanitization, injection detection)
3. Implements error recovery with fallback model
4. Uses semantic deduplication
5. Enforces compliance requirements
6. Integrates with CI/CD
7. Provides clear usage examples
8. Documents TDD approach and progress