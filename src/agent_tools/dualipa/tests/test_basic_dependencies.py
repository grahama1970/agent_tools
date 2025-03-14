"""
Basic dependency tests for DuaLipa.

These tests verify that all core dependencies are working correctly
before proceeding to more complex integration tests.

Official Documentation References:
- pytorch: https://pytorch.org/docs/stable/
- transformers: https://huggingface.co/docs/transformers/
- peft: https://huggingface.co/docs/peft/
- datasets: https://huggingface.co/docs/datasets/
- gitpython: https://gitpython.readthedocs.io/
- markdown-it-py: https://markdown-it-py.readthedocs.io/
- mistune: https://mistune.readthedocs.io/
- litellm: https://docs.litellm.ai/docs/
- tenacity: https://tenacity.readthedocs.io/
- loguru: https://loguru.readthedocs.io/
- rapidfuzz: https://github.com/maxbachmann/RapidFuzz
- click: https://click.palletsprojects.com/
- rich: https://rich.readthedocs.io/
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path to import dualipa modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Set up logger for test output
from loguru import logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="WARNING")


def test_torch_basics():
    """Test that basic PyTorch functionality works.
    
    This verifies core tensor operations and device availability.
    
    Documentation: https://pytorch.org/docs/stable/tensors.html
    """
    try:
        import torch
        
        # Test tensor creation and operations
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        
        assert z.tolist() == [5, 7, 9], "Basic tensor addition failed"
        
        # Test mathematical operations
        squared = x.pow(2)
        assert squared.tolist() == [1, 4, 9], "Tensor power operation failed"
        
        # Test device information (doesn't require GPU)
        device_info = f"CUDA available: {torch.cuda.is_available()}"
        logger.info(device_info)
        
    except ImportError as e:
        pytest.skip(f"PyTorch not installed: {e}")


def test_transformers_basics():
    """Test that basic Transformers functionality works.
    
    This verifies tokenizer loading and basic tokenization.
    
    Documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer
    """
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer - use a small model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokens = tokenizer("Hello, world!")
        
        assert isinstance(tokens, dict), "Tokenizer should return a dictionary"
        assert "input_ids" in tokens, "Tokenizer output should contain input_ids"
        assert len(tokens["input_ids"]) > 0, "Tokenizer should produce non-empty input_ids"
        
        # Test decoding
        decoded = tokenizer.decode(tokens["input_ids"])
        assert "hello" in decoded.lower(), "Decoded text should contain original input"
        
    except ImportError as e:
        pytest.skip(f"Transformers not installed: {e}")


def test_peft_basics():
    """Test that basic PEFT functionality works.
    
    This verifies LoRA configuration and adapter model availability.
    
    Documentation: https://huggingface.co/docs/peft/conceptual_guides/lora
    """
    try:
        from peft import LoraConfig, TaskType
        
        # Test LoRA config creation
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        assert lora_config.r == 8, "LoRA rank parameter not set correctly"
        assert lora_config.lora_alpha == 32, "LoRA alpha parameter not set correctly"
        assert lora_config.target_modules == ["q", "v"], "LoRA target modules not set correctly"
        assert lora_config.task_type == TaskType.CAUSAL_LM, "Task type not set correctly"
        
    except ImportError as e:
        pytest.skip(f"PEFT not installed: {e}")


def test_datasets_basics():
    """Test that basic Datasets functionality works.
    
    This verifies dataset creation, access, and basic operations.
    
    Documentation: https://huggingface.co/docs/datasets/package_reference/main_classes
    """
    try:
        from datasets import Dataset
        
        # Test dataset creation
        data = {"text": ["Hello", "World"], "label": [0, 1]}
        dataset = Dataset.from_dict(data)
        
        assert len(dataset) == 2, "Dataset should have 2 items"
        assert dataset[0]["text"] == "Hello", "First item text should be 'Hello'"
        assert dataset[1]["label"] == 1, "Second item label should be 1"
        
        # Test dataset filtering
        filtered = dataset.filter(lambda x: x["label"] == 1)
        assert len(filtered) == 1, "Filtered dataset should have 1 item"
        assert filtered[0]["text"] == "World", "Filtered item should be 'World'"
        
        # Test dataset mapping
        mapped = dataset.map(lambda x: {"text_upper": x["text"].upper()})
        assert "text_upper" in mapped.features, "Mapped dataset should have new column"
        assert mapped[0]["text_upper"] == "HELLO", "Mapped text should be uppercase"
        
    except ImportError as e:
        pytest.skip(f"Datasets not installed: {e}")


def test_gitpython_basics():
    """Test that basic GitPython functionality works.
    
    This verifies repository operations like init, add, and status.
    
    Documentation: https://gitpython.readthedocs.io/en/stable/tutorial.html
    """
    try:
        import git
        
        # Test git functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Initialize a new repository
                repo = git.Repo.init(temp_dir)
                
                # Create a test file
                test_file = os.path.join(temp_dir, "test.txt")
                with open(test_file, "w") as f:
                    f.write("test content")
                
                # Add the file to git
                repo.index.add(["test.txt"])
                
                # Assert the file is in the index
                assert "test.txt" in [item.path for item in repo.index.entries], "File should be in git index"
                
                # Check status
                status = repo.git.status()
                assert "test.txt" in status, "File should appear in status output"
                
            except git.GitCommandError as e:
                pytest.skip(f"Git command error: {e}")
    except ImportError as e:
        pytest.skip(f"GitPython not installed: {e}")


def test_markdown_parsing_basics():
    """Test that basic markdown parsing functionality works.
    
    This verifies parsing with both markdown-it-py and mistune if available.
    
    Documentation:
    - markdown-it-py: https://markdown-it-py.readthedocs.io/en/latest/using.html
    - mistune: https://mistune.readthedocs.io/en/latest/
    """
    markdown_test = "# Heading\n\nThis is a paragraph with **bold** text.\n\n```python\nprint('code block')\n```"
    markdown_parsed = False
    
    # Try markdown-it-py
    try:
        import markdown_it
        
        md = markdown_it.MarkdownIt()
        tokens = md.parse(markdown_test)
        
        # Check tokens were generated
        assert len(tokens) > 0, "Should generate tokens from markdown"
        
        # Find heading token
        heading_tokens = [t for t in tokens if t.type == "heading_open"]
        assert len(heading_tokens) > 0, "Should parse heading correctly"
        assert heading_tokens[0].tag == "h1", "Should identify h1 tag"
        
        # Find code block
        code_tokens = [t for t in tokens if t.type == "fence"]
        assert len(code_tokens) > 0, "Should parse code block correctly"
        assert "python" in code_tokens[0].info, "Should identify language"
        
        markdown_parsed = True
    except ImportError:
        pass
    
    # Try mistune
    try:
        import mistune
        
        parsed = mistune.markdown(markdown_test)
        
        # Simple check for HTML output
        assert "<h1>" in parsed, "Should parse heading to h1 tag"
        assert "<strong>" in parsed, "Should parse bold text to strong tag"
        assert "<code>" in parsed, "Should parse code block"
        
        markdown_parsed = True
    except ImportError:
        pass
    
    # If neither markdown parser is available, skip
    if not markdown_parsed:
        pytest.skip("No markdown parser installed")


def test_litellm_imports():
    """Test that litellm imports work correctly.
    
    This verifies the litellm package is available and basic functionality can be imported.
    
    Documentation: https://docs.litellm.ai/docs/
    """
    try:
        import litellm
        
        # Just check that basic attributes exist
        assert hasattr(litellm, 'completion'), "litellm.completion should exist"
        assert hasattr(litellm, 'acompletion'), "litellm.acompletion should exist"
        
        # Check configuration options
        assert hasattr(litellm, 'set_verbose'), "litellm.set_verbose should exist"
        assert hasattr(litellm, 'cache'), "litellm.cache should exist"
        
    except ImportError as e:
        pytest.skip(f"LiteLLM not installed: {e}")


def test_tenacity_basics():
    """Test that tenacity basics work correctly.
    
    This verifies retry decorators and retry strategy configuration.
    
    Documentation: https://tenacity.readthedocs.io/en/latest/
    """
    try:
        import tenacity
        from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
        
        # Check basic functionality
        assert callable(retry), "retry should be callable"
        assert callable(stop_after_attempt), "stop_after_attempt should be callable"
        
        # Create a test function with retry
        counter = {"count": 0}
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01),
            retry=retry_if_exception_type(ValueError)
        )
        def test_function():
            counter["count"] += 1
            if counter["count"] < 3:
                raise ValueError("Test error")
            return "success"
        
        # Call the function and check it retries correctly
        result = test_function()
        assert result == "success", "Function should eventually succeed"
        assert counter["count"] == 3, "Function should be called exactly 3 times"
        
    except ImportError as e:
        pytest.skip(f"Tenacity not installed: {e}")


def test_caching_tenacity():
    """Test the caching_tenacity functionality if available.
    
    This verifies that the cached_retry decorator works correctly.
    
    Documentation: See package_usage rules for caching_tenacity pattern
    """
    try:
        # First try to import from snippets
        try:
            from snippets.caching_tenacity import cached_retry
            caching_source = "snippets.caching_tenacity"
        except ImportError:
            # Then try to import from llm_generator
            from llm_generator import cached_retry
            caching_source = "llm_generator"
            
        # Create a counter to track calls
        counter = {"count": 0}
        
        # Define a function with cached retry
        @cached_retry(
            retries=3,
            cache_ttl=10
        )
        def cached_function(arg1, arg2):
            counter["count"] += 1
            return f"{arg1}_{arg2}"
        
        # Call the function twice with the same arguments
        result1 = cached_function("test", 123)
        result2 = cached_function("test", 123)
        
        # Should be called once and cached
        assert counter["count"] == 1, f"Function should be called once (imported from {caching_source})"
        assert result1 == result2, "Results should be the same"
        
        # Call with different arguments
        result3 = cached_function("test", 456)
        
        # Should be called again
        assert counter["count"] == 2, "Function should be called again for different arguments"
        assert result1 != result3, "Results should be different"
        
        logger.info(f"caching_tenacity successfully tested from {caching_source}")
        
    except ImportError as e:
        pytest.skip(f"cached_retry not available: {e}")


def test_loguru_basics():
    """Test that loguru basics work correctly.
    
    This verifies logger configuration and output.
    
    Documentation: https://loguru.readthedocs.io/
    """
    try:
        from loguru import logger
        import io
        
        # Test logging to a string buffer
        string_io = io.StringIO()
        logger.remove()  # Remove all existing handlers
        logger.add(string_io, format="{message}")
        
        test_message = "This is a test log message"
        logger.info(test_message)
        
        log_output = string_io.getvalue()
        assert test_message in log_output, "Log message should be in output"
        
    except ImportError as e:
        pytest.skip(f"Loguru not installed: {e}")


def test_rapidfuzz_basics():
    """Test that RapidFuzz basics work correctly.
    
    This verifies string similarity comparisons.
    
    Documentation: https://github.com/maxbachmann/RapidFuzz
    """
    try:
        from rapidfuzz import fuzz, process
        
        # Test basic string similarity
        str1 = "This is a test string"
        str2 = "This is a test String"
        similarity = fuzz.ratio(str1, str2)
        
        assert similarity > 90, "Similar strings should have high similarity score"
        
        # Test process.extractOne
        strings = ["apple", "banana", "orange", "pear"]
        best_match = process.extractOne("aple", strings)
        
        assert best_match[0] == "apple", "Should find the closest match"
        assert best_match[1] > 80, "Should have high similarity score"
        
    except ImportError as e:
        pytest.skip(f"RapidFuzz not installed: {e}")


def test_cli_utilities():
    """Test that CLI utilities (Click and Rich) work correctly.
    
    This verifies basic CLI utilities functionality.
    
    Documentation:
    - Click: https://click.palletsprojects.com/
    - Rich: https://rich.readthedocs.io/
    """
    click_available = False
    rich_available = False
    
    # Test Click
    try:
        import click
        from click.testing import CliRunner
        
        # Create a simple CLI function
        @click.command()
        @click.option('--name', default='World', help='Who to greet')
        def hello(name):
            return f"Hello {name}!"
        
        # Test CLI invocation
        runner = CliRunner()
        result = runner.invoke(hello, ['--name', 'Test'])
        
        assert result.exit_code == 0, "CLI command should exit successfully"
        assert "Hello Test!" in result.output, "CLI command should produce expected output"
        
        click_available = True
    except ImportError:
        pass
    
    # Test Rich
    try:
        import rich
        from rich.console import Console
        import io
        
        # Test console output
        console = Console(file=io.StringIO())
        console.print("Test", style="bold red")
        
        output = console.file.getvalue()
        assert "Test" in output, "Console should output the text"
        
        rich_available = True
    except ImportError:
        pass
    
    if not click_available and not rich_available:
        pytest.skip("Neither Click nor Rich is installed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 