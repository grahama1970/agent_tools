# Test Fixing Task Plan

## Available Repositories
- Python: `/test_repos/requests` ✅
- JavaScript: `/test_repos/react` ✅
- Rust: `/test_repos/rust-analyzer` ✅
- TypeScript: `/test_repos/typescript-sample` ✅ 
- C++: `/test_repos/cpp-sample` ✅
- Go: `/test_repos/go-sample` ✅

## Current Test Status
- test_block_extraction.py: 3/3 passing ✅
- test_block_verification.py: 4/4 passing ✅
- test_multilang_extractor.py: 7/7 passing ✅
- test_output_examples.py: Need to implement missing functions ❌
- test_markdown_parser.py: Need to create local markdown samples ❌

## Tasks to Fix Remaining Tests

### 1. test_output_examples.py
- [ ] Implement missing `format_output_as_json` function
- [ ] Create helper functions for output formatting
- [ ] Use locally available repository files for testing

### 2. test_markdown_parser.py
- [ ] Create local markdown samples for testing
- [ ] Implement fixtures to provide predictable test data
- [ ] Ensure extraction and parsing functions work properly

## Approach
1. Examine each failing test to understand its requirements
2. Identify the missing implementations or broken code paths
3. Create minimal implementations to make tests pass
4. Refactor implementations as needed for robustness
5. Update tests to use concrete files from repositories

## Root Issues to Address
1. Tree-sitter Integration: Use appropriate fallbacks when tree-sitter is unavailable
2. Error Handling: Convert skips to fails with useful messages when appropriate
3. Use specific files from repositories instead of general globbing
4. Implement missing functions needed by the tests 