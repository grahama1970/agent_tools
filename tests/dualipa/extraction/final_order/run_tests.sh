#!/bin/bash

# DuaLipa Pipeline Test Runner
# This script runs tests in the proper pipeline order

echo "=== DuaLipa Pipeline Test Runner ==="
echo ""

# Function to run tests for a stage
run_stage() {
    local stage_num=$1
    local stage_name=$2
    local test_pattern=$3
    
    echo "===== STAGE ${stage_num}: ${stage_name} ====="
    
    # Run tests for this stage
    python -m pytest ${test_pattern} -v
    
    # Check if tests passed
    if [ $? -eq 0 ]; then
        echo "✅ Stage ${stage_num} tests PASSED"
    else
        echo "❌ Stage ${stage_num} tests FAILED"
        if [ "$4" == "stop" ]; then
            echo "❌ Stopping test execution as requested"
            exit 1
        fi
    fi
    echo ""
}

# Determine if we should stop on failure
STOP_ON_FAILURE=""
if [ "$1" == "--stop-on-failure" ]; then
    STOP_ON_FAILURE="stop"
    echo "Will stop on first test failure"
    echo ""
fi

# Run all stages in order
run_stage "1" "Smoke Tests" "test_0[1-9]_*.py" $STOP_ON_FAILURE
run_stage "2" "Repository Operations" "test_1[0-9]_*.py" $STOP_ON_FAILURE
run_stage "3" "Python AST Extraction" "test_2[0-9]_*.py" $STOP_ON_FAILURE
run_stage "4" "Tree-sitter Extraction" "test_3[0-9]_*.py" $STOP_ON_FAILURE
run_stage "5" "General Extraction" "test_4[0-9]_*.py" $STOP_ON_FAILURE
run_stage "6" "Markdown Extraction" "test_5[0-9]_*.py" $STOP_ON_FAILURE
run_stage "7.1" "Block Verification" "test_6[0-9]_*.py" $STOP_ON_FAILURE
run_stage "7.2" "Multi-language Extraction" "test_7[0-9]_*.py" $STOP_ON_FAILURE
run_stage "7.3" "Output Generation" "test_8[0-9]_*.py" $STOP_ON_FAILURE
run_stage "7.4" "Full Repository Operations" "test_9[0-9]_*.py" $STOP_ON_FAILURE

echo "=== All tests completed ===" 