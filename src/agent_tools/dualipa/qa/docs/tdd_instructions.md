# TDD Instructions for AI Assistants

This guide provides specific instructions for AI assistants implementing code using Test-Driven Development (TDD). Following these instructions will improve code quality and adherence to requirements.

## Step-by-Step TDD Process

### 1. Understand Requirements First
```
- Read ALL documentation files before starting
- Create a list of technical patterns required (e.g., textwrap.dedent())
- Note specific function signatures and parameters
- Identify the smallest testable unit of functionality
```

### 2. Create Test Before Implementation
```python
# Example: Creating a test for process_extraction_json
async def test_minimal_pipeline_real_data(sample_data, tmp_path):
    """Test the minimal pipeline with real data.
    
    Input: Real JSON.
    Expect: One Q&A pair written to file.
    """
    # Import the function to test (may not exist yet)
    try:
        from agent_tools.dualipa.qa.processor import process_extraction_json
    except ImportError:
        pytest.fail("Missing implementation: process_extraction_json")
    
    # Set up test parameters
    output_file = tmp_path / "test_output.json"
    
    # Run the function
    response = await process_extraction_json(
        input_data=sample_data,
        output_file=output_file
    )
    
    # Verify basic expectations
    assert response is not None
    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)
        assert "qa_pairs" in data
        assert len(data["qa_pairs"]) >= 1
```

### 3. Run Test to Verify Failure
```
- Expect test to fail since implementation doesn't exist
- Confirm failure is for expected reason
```

### 4. Implement Minimal Solution
```python
# Example: Minimal implementation
async def process_extraction_json(
    input_data: Union[Dict[str, Any], str, Path],
    output_file: Optional[Union[str, Path]] = None
) -> QAResponse:
    """Process extraction JSON to generate QA pairs.
    
    Args:
        input_data: Input JSON data or file path
        output_file: Optional path to write output JSON
        
    Returns:
        QA response with generated pairs
    """
    # Load data
    if isinstance(input_data, (str, Path)):
        with open(input_data, 'r') as f:
            input_json = json.load(f)
    else:
        input_json = input_data
    
    # Create minimal QA pair
    pair = QAPair(
        question="What is this module for?",
        answer=textwrap.dedent("""
            This module generates QA pairs from content.
        """).strip()
    )
    
    # Create response
    response = QAResponse(
        qa_pairs=[pair],
        generation_metadata={"timestamp": datetime.now().isoformat()}
    )
    
    # Write output
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(response.model_dump(), f, indent=2)
    
    return response
```

### 5. Apply Technical Patterns
```
- Use textwrap.dedent() for all multiline strings
- Follow error handling patterns
- Use asyncio.to_thread() if specified
- Match function signatures exactly
```

### 6. Verify Test Passes
```
- Run test again to verify implementation works
- Do not add features beyond what's needed for the test
```

### 7. Document Implementation
```
- Add docstrings explaining implementation
- Note any design decisions
- Update documentation if needed
```

## Common Mistakes to Avoid

1. **Implementing Too Much**
   - ❌ Creating complex implementations beyond test requirements
   - ✅ Implementing only what's needed for current test to pass

2. **Skipping Technical Patterns**
   - ❌ Ignoring project-specific patterns (textwrap.dedent, etc.)
   - ✅ Explicitly checking each pattern during implementation

3. **Changing Test After Implementation**
   - ❌ Modifying tests to match implementation
   - ✅ Modifying implementation to match tests

4. **Assuming Future Requirements**
   - ❌ Adding features "we might need later"
   - ✅ Following YAGNI principle (You Aren't Gonna Need It)

## AI Assistant Prompt Guidelines

When prompted to implement functionality using TDD:

1. First respond with: "I'll implement this using TDD. Let me start by creating a test."
2. Create and share the test first
3. Explain what the test is verifying
4. Create minimal implementation meeting all technical requirements
5. Verify implementation against test
6. Only after test passes, discuss possible improvements

## Technical Pattern Checklist

- [ ] Use `textwrap.dedent()` for all multiline strings
- [ ] Handle file paths correctly (str vs Path)
- [ ] Implement proper error handling
- [ ] Use asyncio.to_thread() for blocking operations
- [ ] Follow project naming conventions
- [ ] Maintain type hints throughout
- [ ] Document all public functions and classes

Following these instructions will result in higher quality, more maintainable code that precisely meets requirements through TDD methodology.