#!/bin/bash

# Create the directory for the ordered tests
mkdir -p final_order

# Stage 1: Smoke Tests
cp test_simple.py final_order/test_01_simple.py
cp test_import.py final_order/test_02_import.py

# Stage 2: Repository Operations
cp test_github_utils.py final_order/test_10_github_utils.py

# Stage 3: Python AST Extraction
cp test_python_extractor.py final_order/test_20_python_extractor.py

# Stage 4: Tree-sitter Extraction
cp test_js_ts_extraction.py final_order/test_30_js_ts_extraction.py
cp test_tree_sitter_hierarchy.py final_order/test_31_tree_sitter_hierarchy.py

# Stage 5: General Extraction
cp test_code_extractor.py final_order/test_40_code_extractor.py
cp test_block_extractor.py final_order/test_41_block_extractor.py
cp test_block_extraction.py final_order/test_42_block_extraction.py

# Stage 6: Markdown Extraction
cp test_markdown_parser.py final_order/test_50_markdown_parser.py
cp test_markdown_hierarchy.py final_order/test_51_markdown_hierarchy.py
cp test_markdown_it_parser.py final_order/test_52_markdown_it_parser.py

# Stage 7: JSON Conversion, Verification and Integration
cp test_block_verification.py final_order/test_60_block_verification.py
cp test_code_hierarchy.py final_order/test_61_code_hierarchy.py
cp test_multilang_extractor.py final_order/test_70_multilang_extractor.py
cp test_output_examples.py final_order/test_80_output_examples.py
cp test_repo_operations.py final_order/test_90_repo_operations.py

# Copy other necessary files
cp README.md final_order/
cp task_plan.md final_order/
cp __init__.py final_order/

# Copy the markdown_samples directory
cp -r markdown_samples final_order/

echo "Files renamed and copied to final_order directory:"
ls -la final_order/ 