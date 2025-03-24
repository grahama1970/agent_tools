### How to grep a markdown table and code block in a table 
```find /home/grahama/workspace/experiments/agent_tools/test_repos -name "*.md" -type f -exec grep -l "\`\`\`" {} \; | xargs grep -l "|-" 2>/dev/null```

### How to Run a Pytest in Dualipa
```cd /home/grahama/workspace/experiments/agent_tools && python -m pytest tests/dualipa/stage2/test_block_extractor.py::test_python_block_extraction -vv```

or, with `uv run pytest` (should use)
```clear;cd /home/grahama/workspace/experiments/agent_tools && uv run  pytest tests/dualipa/stage2/test_block_extractor.py::test_python_block_extr
action -vv```

run tests in a directory
```cd /home/grahama/workspace/experiments/agent_tools && python -m pytest tests/dualipa/stage2/ -v```


# Grep for strings in files in a directory
```grep -rlZ 'unsloth' . | xargs -0 grep -l 'training'```

### Run all extraction tests in sequencce
```cd /home/grahama/workspace/experiments/agent_tools && for test in test_01_simple.py test_02_import.py test_05_stats_consistency.py test_10_github_utils.py test_15_language_detection.py test_17_format_validation.py test_20_block_verification.py test_25_tree_sitter_hierarchy.py test_30_python_extractor.py test_31_js_ts_extraction.py test_35_markdown_extraction.py test_41_sample_block_extraction.py test_42_realworld_block_extraction.py test_45_generic_extraction.py test_51_markdown_hierarchy.py test_52_markdown_it_parser.py test_55_code_hierarchy.py test_65_code_extractor.py test_70_multilang_extractor.py test_80_output_examples.py test_85_repository_integration.py test_90_repo_operations.py; do python -m pytest tests/dualipa/extraction/final_order/$test -v || break; done```


# Tailscale Remote Login
ssh -p 2222 -i ~/.ssh/id_ed25519_wsl2 grahama@100.76.171.37 -vv