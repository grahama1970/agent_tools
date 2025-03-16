# Stage 1: Repository Download [SOURCE ACQUISITION]

This directory contains tests for Stage 1 of the DuaLipa pipeline: Repository Download.

## Purpose

These tests verify that the GitHub utilities and repository operations work correctly:

- Repository URL parsing
- Repository cloning
- Repository content retrieval

## Components Tested

- `github_utils.py`: Repository acquisition (`download_github_repo()`)

## Running the Tests

From the project root, run:

```bash
python -m pytest tests/dualipa/stage1
```

## Tests Overview

- `test_github_utils.py`: Tests for GitHub URL parsing and repository operations
- `test_repo_operations.py`: Tests for repository cloning and content retrieval operations 