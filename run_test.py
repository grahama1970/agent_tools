import subprocess
import sys
import re

def run_test():
    test_path = "src/agent_tools/cursor_rules/tests/memory/test_memory_decay.py::test_domain_preservation_search"
    
    try:
        # Set environment variables to reduce embedding output
        env = {
            "PYTHONUNBUFFERED": "1",
            "LOG_LEVEL": "INFO",  # Reduce logging level
            "PYTEST_ADDOPTS": "--no-header --no-summary",  # Reduce pytest output
        }
        
        # Run pytest with verbose flag and capture output
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v"],
            capture_output=True,
            text=True,
            check=False,
            env=env
        )
        
        # Clean up the output by truncating embedding vectors
        cleaned_stdout = truncate_embedding_vectors(result.stdout)
        cleaned_stderr = truncate_embedding_vectors(result.stderr)
        
        # Print output
        print("STDOUT:")
        print(cleaned_stdout)
        
        print("\nSTDERR:")
        print(cleaned_stderr)
        
        # Return exit code
        print(f"\nExit code: {result.returncode}")
        return result.returncode
        
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

def truncate_embedding_vectors(text):
    """Truncate long embedding vector outputs in the text."""
    # Pattern to match embedding vectors (long sequences of numbers)
    pattern = r'(-?\d+\.\d+,\s*){10,}'
    
    # Replace with a placeholder
    truncated = re.sub(pattern, "[EMBEDDING_VECTOR_TRUNCATED], ", text)
    
    # Also truncate any remaining very long lines
    lines = truncated.split('\n')
    truncated_lines = []
    for line in lines:
        if len(line) > 120:  # If line is longer than 120 chars
            truncated_lines.append(line[:100] + "... [truncated]")
        else:
            truncated_lines.append(line)
    
    return '\n'.join(truncated_lines)

if __name__ == "__main__":
    sys.exit(run_test()) 