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

