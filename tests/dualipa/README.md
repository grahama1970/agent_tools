# DuaLipa Tests

This directory contains tests for the DuaLipa pipeline, organized by pipeline stages:

## Pipeline Stages

DuaLipa operates in the following sequential stages:

1. Repository Download [SOURCE ACQUISITION] - Located in [`stage1/`](./stage1/)
2. Code and Documentation Extraction [DATA EXTRACTION] - Located in [`stage2/`](./stage2/)
3. QA Pair Generation and Dataset Formatting [DATA PREPARATION]
4. Model Fine-tuning with Unsloth [MODEL TRAINING]
5. LoRA Adapter Merging [MODEL OPTIMIZATION]
6. Deployment to Hugging Face [DISTRIBUTION]
7. Usage for Current Code Generation [APPLICATION]

## Test Organization

- **Stage 1 Tests**: Repository Download tests (github_utils.py)
- **Stage 2 Tests**: Code and Documentation Extraction tests (code_extractor.py, language_detection.py, markdown_parser.py)
- **Common Tests**: Tests for shared functionality across stages
- **Integration Tests**: Tests for full pipeline integration

## Running Tests

Run all dualipa tests:
```bash
python -m pytest tests/dualipa
```

Run tests for a specific stage:
```bash
python -m pytest tests/dualipa/stage1  # Stage 1 tests
python -m pytest tests/dualipa/stage2  # Stage 2 tests
```

Run a specific test:
```bash
python -m pytest tests/dualipa/stage1/test_github_utils.py
```

## Test Requirements

See [`requirements-test.txt`](./tests/requirements-test.txt) for test dependencies. 