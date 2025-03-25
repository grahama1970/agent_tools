#!/bin/bash

# DuaLipa Extraction Test Runner
# This script runs tests in the proper order and stops on first failure

set -e  # Exit on first error

echo "=== DuaLipa Extraction Test Runner ==="
echo ""

# Function to run tests for a stage
run_stage() {
    local stage_name=$1
    local test_path=$2
    
    echo "===== Testing: ${stage_name} ====="
    
    # Run tests for this stage
    if ! python -m pytest ${test_path} -v --tb=short; then
        echo "❌ ${stage_name} tests FAILED"
        exit 1
    fi
    
    echo "✅ ${stage_name} tests PASSED"
    echo ""
}

# Core tests first
run_stage "Core - Simple Tests" "tests/dualipa/extraction/core/test_simple.py"
run_stage "Core - Import Tests" "tests/dualipa/extraction/core/test_import.py"

# Utils tests
run_stage "Utils - Language Utils" "tests/dualipa/extraction/extractors/utils/test_language_utils.py"
run_stage "Utils - Validation Utils" "tests/dualipa/extraction/extractors/utils/test_validation_utils.py"
run_stage "Utils - Verification Utils" "tests/dualipa/extraction/extractors/utils/test_verification_utils.py"
run_stage "Utils - Stats Utils" "tests/dualipa/extraction/extractors/utils/test_stats_utils.py"

# Code extractor tests
run_stage "Code - Python Extractor" "tests/dualipa/extraction/extractors/code/test_python_extractor.py"
run_stage "Code - JS/TS Extractor" "tests/dualipa/extraction/extractors/code/test_js_ts_extractor.py"
run_stage "Code - Generic Extractor" "tests/dualipa/extraction/extractors/code/test_generic_extractor.py"
run_stage "Code - Hierarchy" "tests/dualipa/extraction/extractors/code/test_hierarchy.py"
run_stage "Code - Code Extractor" "tests/dualipa/extraction/extractors/code/test_code_extractor.py"

# Markdown extractor tests
run_stage "Markdown - Parser" "tests/dualipa/extraction/extractors/markdown/test_parser.py"
run_stage "Markdown - Hierarchy" "tests/dualipa/extraction/extractors/markdown/test_hierarchy.py"
run_stage "Markdown - Markdown Extractor" "tests/dualipa/extraction/extractors/markdown/test_markdown_extractor.py"

# GitHub utils tests
run_stage "GitHub - API Utils" "tests/dualipa/extraction/extractors/github/test_api_utils.py"
run_stage "GitHub - Repo Utils" "tests/dualipa/extraction/extractors/github/test_repo_utils.py"

# Integration tests
run_stage "Integration - Sample Extraction" "tests/dualipa/extraction/integration/test_sample_extraction.py"
run_stage "Integration - Real-world Extraction" "tests/dualipa/extraction/integration/test_realworld_extraction.py"
run_stage "Integration - Repository Integration" "tests/dualipa/extraction/integration/test_repository_integration.py"
run_stage "Integration - Output Examples" "tests/dualipa/extraction/integration/test_output_examples.py"

echo "=== All tests completed successfully! ===" 