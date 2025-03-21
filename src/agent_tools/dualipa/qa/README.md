# DuaLipa LLM Q&A Generation Module

This is a secure, enterprise-ready interface for generating question-answer pairs from structured content, with production-grade validation, error recovery, and compliance checks.

## Features

- **Input Sanitization**: Removes HTML/JS and detects prompt injections
- **Temperature Iteration**: Tests multiple temperatures (0.3, 0.5, 0.7)
- **Bidirectional Generation**: Creates both forward (Q→A) and reverse (A→Q) pairs
- **Reasoning Enrichment**: All answers include an "Oh wait?!" moment for deeper insights
- **Semantic Deduplication**: Removes similar QA pairs using embedding similarity
- **Error Recovery**: Implements retry logic with exponential backoff and fallback models
- **Validation**: Enforces quality standards on all generated content

## Installation

```bash
# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from agent_tools.dualipa.qa import process_extraction_json

# Generate QA pairs from extraction JSON
output = await process_extraction_json(
    input_data="extracted_content.json", 
    output_file="qa_pairs.json"
)

# Access the generated pairs
for pair in output.qa_pairs:
    print(f"Q: {pair.question}")
    print(f"A: {pair.answer}")
    print(f"Reasoning: {pair.reasoning}")
    print()
```

### CLI Usage

```bash
# Generate QA pairs from a file
python -m agent_tools.dualipa.qa.cli extracted_content.json -o qa_pairs.json

# Customize temperature range
python -m agent_tools.dualipa.qa.cli extracted_content.json --temps 0.3 0.7

# Enable verbose logging
python -m agent_tools.dualipa.qa.cli extracted_content.json -v
```

## Input/Output Format

### Expected Input Format

```json
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

```json
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
    "processing_time_seconds": 42.1,
    "sections_processed": 5,
    "forward_pairs": 15,
    "reverse_pairs": 8,
    "validation_checks": {
      "sanitized_inputs": true,
      "avg_reasoning_length": 18.7,
      "reverse_pairs_ratio": 0.23
    }
  }
}
```

## Testing

```bash
# Run the tests
pytest tests/qa/

# Run with coverage
pytest --cov=agent_tools.dualipa.qa tests/qa/
```

## Implementation Status

This is a production-ready MVP with:

- Full input validation and sanitization
- Bidirectional generation 
- Advanced error handling
- Comprehensive test coverage