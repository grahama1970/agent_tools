# LESSONS LEARNED: Building Reliable Extraction Workflows

This document captures critical lessons learned during the DuaLipa extraction project. These principles should be applied to all future extraction tasks to ensure reliability, completeness, and efficiency.

## Repository Analysis First

**ALWAYS start by analyzing what's actually in the repository:**

```python
# Essential first step for any extraction project
def analyze_repository(repo_path):
    """Count all files by type before starting extraction work"""
    file_counts = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1]
            file_counts[ext] = file_counts.get(ext, 0) + 1
    return file_counts
```

This gives us:
- Exact counts of each file type
- Understanding of repository structure
- Baseline to verify extraction completeness

## Verify Complete Coverage

After extraction, always verify coverage against the original counts:

```python
def verify_extraction_coverage(extracted_files, original_counts):
    """Verify extraction coverage against original repository"""
    extracted_counts = Counter()
    for file_path in extracted_files:
        ext = os.path.splitext(file_path)[1]
        extracted_counts[ext] += 1
        
    for ext, count in original_counts.items():
        if ext in extracted_counts:
            coverage = (extracted_counts[ext] / count) * 100
            if coverage < 95:
                print(f"WARNING: Only extracted {coverage:.1f}% of {ext} files")
```

This catches issues like "only 3 of 70 Python files extracted" immediately.

## Simple > Complex

Start with the simplest reliable approach:

✅ `find_all_python_files()` - Complete, reliable, straightforward  
❌ Complex sampling logic - Error-prone, easy to miss files

Don't reinvent when basic utilities work:
- `os.walk()` for file discovery
- `open()` for file reading
- Basic Python data structures

## Step-by-Step Verification

Implement explicit verification at critical steps:

1. **Before extraction:** Verify repository structure and file counts
2. **During extraction:** Log and verify file processing
3. **After extraction:** Compare extracted files to original counts
4. **Final output:** Verify all required fields and structure

Include assertions in code to catch issues early:

```python
# Verify we found the expected number of Python files
assert len(python_files) >= 60, f"Only found {len(python_files)} Python files, expected ~70"
```

## Document Test Cases

Maintain test cases for specific files that must be included:

```
Repository must-extract files:
- tests/js/common/test-data/search/docs/generate_ii_sa_dataset.py
- utils/gantt.py
- scripts/toolbox/modules/HotBackup.py
```

This ensures critical files are always verified.

## Transparency in Reporting

Always report extraction statistics:

```
Extraction Summary:
- Repository: 7,032 total files
- Python: 70/70 files extracted (100%)
- C++: 152/3,157 files extracted (4.8%)
- JavaScript: 83/1,794 files extracted (4.6%)
```

This makes coverage gaps immediately obvious.

## Conclusion

The biggest lesson: **trust but verify**. Never assume extraction is complete without explicit verification against the original repository. The cost of missing files far outweighs the effort of adding proper validation.

*This document should be reviewed before beginning any extraction project to ensure these fundamental steps are not missed.*