#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a test and check its exit status
run_test() {
    local test_file=$1
    echo -e "\nRunning test: ${test_file}"
    python -m pytest "${test_file}" -v
    local status=$?
    if [ $status -ne 0 ]; then
        echo -e "${RED}❌ Test failed: ${test_file}${NC}"
        exit $status
    else
        echo -e "${GREEN}✅ Test passed: ${test_file}${NC}"
    fi
}

# Array of test files in order
declare -a test_files=(
    "test_dualipa_stage1.py"
    "test_block_extraction.py"
    "test_extract_direct.py"
    "test_language_detection.py"
    "test_code_extractor.py"
    "test_indented_blocks.py"
)

# Run tests in order
for test_file in "${test_files[@]}"; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    else
        echo -e "${RED}❌ Test file not found: ${test_file}${NC}"
        exit 1
    fi
done

echo -e "\n${GREEN}✅ All tests passed successfully!${NC}" 