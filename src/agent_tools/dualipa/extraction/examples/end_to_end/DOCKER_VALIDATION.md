# Docker Validation for Documentation Extraction

This document explains how to use the Docker-based validation system for the DuaLipa documentation extraction framework.

## Overview

The Docker validation system allows you to:

1. Run validation tests on documentation extraction results
2. View detailed validation reports in a web browser
3. Verify both structural and semantic correctness of extraction

## Setup and Usage

### Option 1: Using Docker Compose

```bash
# Start the validation server
docker-compose up -d

# Access the validation dashboard at:
# http://localhost:8000/
```

### Option 2: Using Direct Docker Command

```bash
# Build and run the validation container
./run_docker.sh

# Or specify a custom port
./run_docker.sh 8080
```

### Option 3: Local Execution (No Docker)

```bash
# Create necessary directories
mkdir -p ./test_results ./test_results_dashboard

# Run validation and start server
./run_validation_server.sh
```

## Validation Dashboard

The validation dashboard provides:

1. **Summary View**: Overall statistics and test results
2. **Detailed Reports**: Per-test validation results with scores for:
   - Structure validation
   - Content validation
   - Format consistency

## Available Tests

The validation framework can test various extraction outputs:

- LENGTH function documentation
- ARRAY_INTERSECTION function documentation
- ArangoDB API documentation
- ReadTheDocs extractions

## Custom Validation

To validate your own extraction results:

```bash
# Convert to validation format
python convert_for_validation.py --input your_extraction.json --output ./test_results/your_extraction_converted.json

# Validate (with auto-detection)
python test_validation_framework.py --extraction ./test_results/your_extraction_converted.json --expected deepseek_length_format.json --output ./test_results/your_validation_results.json

# Generate HTML reports
python generate_validation_report.py --input ./test_results --output ./test_results_dashboard
```

## Troubleshooting

### Container Not Starting

Check for errors in the Docker logs:

```bash
docker logs dualipa-validation-container
```

### Failed Validation

If validation fails:

1. Check the detailed report for specific errors
2. Ensure the extraction format matches the expected format
3. Use the conversion tool to transform the extraction if needed

### Port Conflicts

If the default port (8000) is in use:

```bash
# Using Docker Compose
PORT=8080 docker-compose up -d

# Using run_docker.sh
./run_docker.sh 8080
```