# QA Module CI Utilities

This directory contains utilities for Continuous Integration (CI) and quality assurance for the DuaLipa QA Generation Module.

## Overview

The CI tools verify that the QA module meets performance and quality requirements, including:

1. **Performance verification**: Ensures there are no performance regressions
2. **Coverage verification**: Validates test coverage meets required thresholds
3. **Baseline metrics**: Stores reference performance metrics for comparison

## Performance Verification

The `verify_performance.py` script measures system performance against established baselines to catch regressions.

### Usage:

```bash
# Generate baseline performance metrics
python -m agent_tools.dualipa.qa.ci.verify_performance --generate-baseline

# Verify performance against baseline
python -m agent_tools.dualipa.qa.ci.verify_performance
```

### Key Features:

- Measures batch processing performance
- Verifies worker scaling efficiency
- Validates adaptive chunk calculation
- Tests cache performance
- Monitors memory usage

### Performance Thresholds:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Batch processing time | 20% | Maximum acceptable slowdown |
| Worker scaling efficiency | 15% | Maximum degradation in parallelization |
| Adaptive chunk calculation | 5% | Maximum deviation in chunk sizing |
| Cache hit rate | 10% | Maximum degradation in cache effectiveness |
| Memory usage | 25% | Maximum increase in memory consumption |

## Coverage Verification

The `verify_coverage.py` script ensures the test suite provides adequate code coverage.

### Usage:

```bash
# Verify coverage against thresholds
python -m agent_tools.dualipa.qa.ci.verify_coverage --coverage-file coverage.xml

# Generate a detailed coverage report
python -m agent_tools.dualipa.qa.ci.verify_coverage --generate-report
```

### Coverage Thresholds:

| Component | Threshold | Description |
|-----------|-----------|-------------|
| Overall | 80% | Overall module coverage |
| Processor | 90% | Main processor module |
| Utils | 80% | Utility functions |
| Models | 90% | Data models |
| LLM | 75% | LLM integration |
| CLI | 85% | Command-line interface |
| Monitoring | 80% | Monitoring components |
| Performance | 75% | Performance components |
| Cache | 85% | Caching components |

## GitHub Actions Integration

These tools are integrated into the CI workflow in `.github/workflows/qa_ci.yml` to ensure all pull requests maintain quality standards.

### CI Jobs:

1. **Test**: Runs the test suite with coverage
2. **Performance**: Verifies performance against baselines
3. **Lint**: Ensures code style consistency

## Setup for Local Development

To set up the CI tools for local development:

1. Install dependencies:
   ```bash
   pip install pytest pytest-asyncio pytest-cov pytest-benchmark
   ```

2. Generate baseline metrics:
   ```bash
   python -m agent_tools.dualipa.qa.ci.verify_performance --generate-baseline
   ```

3. Run performance verification:
   ```bash
   python -m agent_tools.dualipa.qa.ci.verify_performance
   ```

4. Run tests with coverage:
   ```bash
   pytest tests/qa/ --cov=src/agent_tools/dualipa/qa --cov-report=xml -v
   ```

5. Verify coverage:
   ```bash
   python -m agent_tools.dualipa.qa.ci.verify_coverage
   ```