#!/bin/bash
# Docker container validation script
# This script runs validation on all test examples and generates HTML reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create validation output directories
mkdir -p ./test_results
mkdir -p ./test_results_dashboard

echo "Running conversion and validation for all test examples..."

# Convert and validate the LENGTH function test
echo "Processing LENGTH function extraction..."
python convert_for_validation.py --input length_function_extraction.json --output ./test_results/length_function_converted.json
python test_validation_framework.py --extraction ./test_results/length_function_converted.json --expected length_function_format.json --output ./test_results/length_function_validation.json

# If array_intersection_test exists, validate it
if [ -f "array_intersection_summary.json" ]; then
    echo "Processing ARRAY INTERSECTION extraction..."
    python convert_for_validation.py --input array_intersection_summary.json --output ./test_results/array_intersection_converted.json
    python test_validation_framework.py --extraction ./test_results/array_intersection_converted.json --expected array_intersection_expected_format.json --output ./test_results/array_intersection_validation.json
fi

# If ArangoDB test exists, validate it
if [ -f "arangodb_extraction_summary.json" ]; then
    echo "Processing ArangoDB extraction..."
    python convert_for_validation.py --input arangodb_extraction_summary.json --output ./test_results/arangodb_converted.json
    python test_validation_framework.py --extraction ./test_results/arangodb_converted.json --expected arangodb_expected_format.json --output ./test_results/arangodb_validation.json
fi

# Run any other specific tests
echo "Processing other extractions with auto-detection..."
python validate_all_tests.py --input-dir ./ --output-dir ./test_results --auto-detect --convert

# Generate HTML reports for all validation results
echo "Generating HTML validation reports..."
python generate_validation_report.py --input ./test_results --output ./test_results_dashboard

echo "Validation complete. Reports generated in ./test_results_dashboard/"
echo "Open http://localhost:8000/test_results_dashboard/summary.html in your browser to view the results"