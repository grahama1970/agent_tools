
# DuaLipa LLM Q&A Generation Module: Updated TDD Strategy

## Testing Framework Setup

```
# conftest.py
import pytest
import json
from unittest.mock import patch, MagicMock
from dualipa.qa.models import QAPair, QAResponse
```

## 1. Enhanced Model Validation Tests

```
# test_models/test_qaresponse.py
def test_metadata_structure():
    response = QAResponse(
        qa_pairs=[QAPair(...)],
        generation_metadata={
            "temperature_range": [0.3, 0.7],
            "forward_pairs": 35,
            "reverse_pairs": 23
        }
    )
    assert response.generation_metadata["forward_pairs"] == 35
    assert "reverse_pairs" in response.generation_metadata
```

## 2. Temperature Iteration Tests

```
# test_llm/test_generation.py
@pytest.mark.parametrize("temps", [
    [0.3, 0.5, 0.7],
    [0.4, 0.6]
])
def test_temperature_iteration(temps):
    test_data = {"sections": [...]}
    result = process_extraction_json(test_data, temps)
    assert len(result.qa_pairs) == len(temps) * 2  # Forward + reverse
```

## 3. Bidirectional Generation Tests

```
# test_llm/test_reversal.py
def test_reversal_quality():
    original = QAPair(question="What is X?", answer="X is...")
    reversed = generate_reversed_qa_pairs([original])
    
    assert reversed.question != original.question
    assert reversed.answer == original.answer
    assert "Oh wait?!" in reversed.reasoning
```

## 4. Validation Pipeline Tests

```
# test_utils/test_validation.py
@pytest.mark.parametrize("reasoning,expected", [
    ("Valid reasoning with Oh wait?! moment", True),
    ("Missing pivot moment", False),
    ("Short", False)
])
def test_reasoning_validation(reasoning, expected):
    pair = QAPair(reasoning=reasoning, ...)
    assert validate_qa_pair(pair) == expected
```

## 5. Cache Implementation Tests

```
# test_llm/test_cache.py
def test_cache_hit_rates():
    initialize_litellm_cache()
    generate_markdown_qa_pairs(section)
    generate_markdown_qa_pairs(section)  
    assert cache_hit_rate() > 0.5
```

## 6. Async Processing Tests

```
# test_processor.py
@pytest.mark.asyncio
async def test_parallel_processing():
    sections = [doc_section, code_section, config_section]
    results = await batch_process_sections(sections, [0.3, 0.5, 0.7])
    assert len(results) >= len(sections) * 3 * 2
```

## Implementation Validation Checklist

```
[
    ("Directory structure alignment", True),
    ("Temperature iteration coverage", [0.3, 0.7]),
    ("Bidirectional validation", "reverse" in directions),
    ("Reasoning quality gates", "Oh wait?!" in all_reasoning),
    ("Metadata completeness", {"forward_pairs", "reverse_pairs"} <= metadata_keys)
]
```

## Multi-Layer Testing Strategy

| Layer                | Test Coverage                          | Example Tests                          |
|----------------------|----------------------------------------|----------------------------------------|
| Model Validation     | Pydantic model constraints             | Field types, validation rules          |
| LLM Interactions     | Temperature handling, reversal logic   | API call formatting, error handling    | 
| Business Logic       | Deduplication, validation              | Edge cases, threshold enforcement      |
| Integration          | Full pipeline execution                | End-to-end JSON processing             |

## Updated Directory Structure

```
dualipa/
├── qa/
│   ├── __init__.py
│   ├── models/
│   ├── llm/
│   ├── utils/
│   └── processor.py
└── tests/
    └── qa/
        ├── test_models/
        ├── test_llm/
        │   ├── test_cache.py
        │   ├── test_generation.py
        │   └── test_reversal.py
        ├── test_utils/
        ├── test_processor.py
        └── conftest.py
```

## Execution Plan

1. **Phase 1: Core Components**
   - Implement model validation tests
   - Add temperature iteration tests
   - Develop bidirectional generation tests

2. **Phase 2: Infrastructure**
   - Implement cache testing
   - Add async processing validation
   - Develop CLI integration tests

3. **Phase 3: Full Pipeline**
   - End-to-end processing tests
   - Metadata validation
   - Error condition testing

4. **Phase 4: Optimization**
   - Performance benchmarking
   - Cache efficiency tests
   - Failure mode analysis

## Test Coverage Requirements

1. **Model Layer**: 100% field validation
2. **LLM Layer**: 95% API interaction coverage
3. **Business Logic**: 100% edge case coverage
4. **Integration**: Full pipeline validation

```
# pytest.ini
[pytest]
asyncio_mode = auto
min_cov = 95
addopts = --cov=dualipa.qa --cov-report=term-missing
```

## Updated Best Practices

1. Test bidirectional generation first
2. Validate metadata structure early
3. Use real temperature ranges in tests
4. Test cache integration holistically
5. Enforce "Oh wait?!" in all reasoning tests
```
