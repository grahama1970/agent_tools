"""
Dataset formatting module for DuaLipa.

This module takes the extracted code and documentation from repositories
and formats it into datasets suitable for training language models.

Official Documentation References:
- pathlib: https://docs.python.org/3/library/pathlib.html
- pydantic: https://docs.pydantic.dev/latest/
- loguru: https://loguru.readthedocs.io/en/stable/
- json: https://docs.python.org/3/library/json.html
- random: https://docs.python.org/3/library/random.html
- tqdm: https://tqdm.github.io/docs/
"""

import json
import os
import importlib
import random
from typing import Dict, List, Any, Optional, Tuple
import inspect
import sys
import re
import argparse
import tempfile
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Import method_validator components if available
try:
    from agent_tools.method_validator.analyzer import MethodAnalyzer, MethodInfo
    from agent_tools.method_validator.cache import AnalysisCache
    METHOD_VALIDATOR_AVAILABLE = True
    logger.info("method_validator is available and will be used for enhanced code analysis")
except ImportError:
    METHOD_VALIDATOR_AVAILABLE = False
    logger.warning("method_validator not available. Using basic function detection.")

# Import LLM generator components
try:
    from .llm_generator import (
        generate_code_qa_pairs, 
        generate_markdown_qa_pairs, 
        generate_reverse_qa_pairs,
        generate_qa_pairs_from_text
    )
    # Import QA validator
    from .qa_validator import (
        validate_and_enhance_qa_pairs,
        detect_duplicate_pairs,
        validate_function_qa_pair
    )
    import asyncio
    LLM_GENERATOR_AVAILABLE = True
    logger.info("LLM generator is available and will be used for enhanced QA pair generation")
except ImportError:
    LLM_GENERATOR_AVAILABLE = False
    logger.warning("LLM generator not available. Using basic QA generation.")


def format_for_lora(input_file: str, output_file: str, use_llm: bool = True, max_pairs_per_item: int = 5) -> None:
    """Formats extracted data into structured question-answer pairs for LoRA fine-tuning.
    
    If the method_validator module is available, it uses advanced function inspection
    to generate more detailed and varied question-answer pairs.
    
    If the LLM generator is available and enabled, it uses LLMs to generate
    higher-quality and more diverse QA pairs, including reverse QA pairs.
    
    Args:
        input_file: Path to the JSON file containing extracted repository data
        output_file: Path where the formatted dataset will be saved
        use_llm: Whether to use LLM-based generation if available
        max_pairs_per_item: Maximum number of QA pairs to generate per item
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        PermissionError: If the output file can't be written
    """
    try:
        # Validate input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        formatted_data = {"qa_pairs": []}

        # Determine which method to use for QA generation
        if LLM_GENERATOR_AVAILABLE and use_llm:
            logger.info("Using LLM-based QA pair generation.")
            qa_pairs = asyncio.run(generate_enhanced_llm_qa_pairs(data, max_pairs_per_item))
            formatted_data["qa_pairs"] = qa_pairs
        elif METHOD_VALIDATOR_AVAILABLE:
            # Use method_validator for enhanced function analysis
            logger.info("Using advanced method analysis to generate QA pairs.")
            formatted_data["qa_pairs"] = generate_enhanced_qa_pairs(data)
        else:
            # Fallback to basic function detection
            logger.info("Using basic function detection for QA pair generation.")
            generate_basic_qa_pairs(data, formatted_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=4)

        logger.info(f"Dataset formatted and saved to {output_file}. Generated {len(formatted_data['qa_pairs'])} QA pairs.")
        print(f"Dataset formatted and saved to {output_file}. Generated {len(formatted_data['qa_pairs'])} QA pairs.")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission error when writing output file: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from input file {input_file}: {e}")
        raise ValueError(f"Invalid JSON format in input file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during formatting: {e}")
        raise


def generate_basic_qa_pairs(data: Dict[str, Any], formatted_data: Dict[str, List[Dict[str, str]]]) -> None:
    """Generate basic question-answer pairs without method_validator.
    
    Args:
        data: Repository data containing files and their content
        formatted_data: Dictionary to populate with QA pairs
    """
    for file in data["files"]:
        content = file["content"].split("\n")
        for line in content:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                formatted_data["qa_pairs"].append({
                    "question": f"What does `{line.strip()}` do?",
                    "answer": file["content"]
                })


def generate_enhanced_qa_pairs(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate enhanced question-answer pairs using method_validator.
    
    Args:
        data: Repository data containing files and their content
        
    Returns:
        List of question-answer pairs with varied formats
    """
    qa_pairs = []
    
    # Create a temporary directory to store Python files for analysis
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Write Python files to temp directory with proper structure
        module_files = {}
        for file in data["files"]:
            if not file["path"].endswith(".py"):
                continue
                
            # Skip __init__.py and empty files
            if file["path"].endswith("__init__.py") or not file["content"].strip():
                continue
                
            # Get the module name from file path
            rel_path = os.path.basename(file["path"])
            module_name = os.path.splitext(rel_path)[0]
            
            # Write to temp file
            temp_file = os.path.join(temp_dir, f"{module_name}.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(file["content"])
            
            module_files[module_name] = {
                "path": temp_file,
                "content": file["content"]
            }
        
        # Add temp_dir to sys.path to enable imports
        sys.path.insert(0, temp_dir)
        
        # Analyze each module with method_validator
        analyzer = MethodAnalyzer(include_builtins=False)
        
        for module_name, file_info in module_files.items():
            try:
                # Try to import the module
                spec = importlib.util.spec_from_file_location(module_name, file_info["path"])
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get all functions and classes from the module
                    for name, obj in inspect.getmembers(module):
                        # Skip private methods and attributes
                        if name.startswith("_"):
                            continue
                            
                        # Analyze functions and methods
                        if inspect.isfunction(obj) or inspect.ismethod(obj):
                            method_info = analyze_function(name, obj, module_name)
                            if method_info:
                                qa_pairs.extend(generate_function_qa_pairs(method_info))
                                
                        # Analyze classes
                        elif inspect.isclass(obj):
                            # Add class-level questions
                            class_info = {
                                "name": name,
                                "doc": inspect.getdoc(obj) or "",
                                "module": module_name,
                                "content": extract_class_source(obj, file_info["content"])
                            }
                            qa_pairs.extend(generate_class_qa_pairs(class_info))
                            
                            # Then add method-level questions
                            for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                                if not method_name.startswith("_"):  # Skip private methods
                                    method_info = analyze_function(method_name, method_obj, module_name, class_name=name)
                                    if method_info:
                                        qa_pairs.extend(generate_function_qa_pairs(method_info))
            
            except Exception as e:
                logger.error(f"Error analyzing module {module_name}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error during enhanced QA pair generation: {e}")
    finally:
        # Clean up
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        shutil.rmtree(temp_dir)
    
    return qa_pairs


def analyze_function(name: str, obj: Any, module_name: str, class_name: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a function or method using method_validator if available, otherwise use basic inspection.
    
    Args:
        name: Name of the function
        obj: Function object
        module_name: Name of the containing module
        class_name: Name of the containing class (if any)
        
    Returns:
        Dictionary with function details
    """
    full_name = f"{class_name}.{name}" if class_name else name
    
    try:
        if METHOD_VALIDATOR_AVAILABLE:
            # Use method_validator for advanced analysis
            method_info = MethodInfo(obj, full_name)
            return method_info.to_dict()
        else:
            # Basic function analysis
            return {
                "name": full_name,
                "doc": inspect.getdoc(obj) or "",
                "signature": str(inspect.signature(obj)),
                "module": module_name,
                "summary": (inspect.getdoc(obj) or "").split("\n")[0] if inspect.getdoc(obj) else "",
                "parameters": {
                    name: {"description": ""} 
                    for name in inspect.signature(obj).parameters
                },
                "examples": [],
                "source": inspect.getsource(obj)
            }
    except Exception as e:
        logger.error(f"Error analyzing function {full_name}: {e}")
        return {}


def extract_class_source(cls: Any, file_content: str) -> str:
    """Extract the source code for a class from file content.
    
    Args:
        cls: The class object
        file_content: Content of the file containing the class
        
    Returns:
        Source code of the class
    """
    try:
        # Try to get source directly
        return inspect.getsource(cls)
    except (IOError, TypeError):
        # Fallback: try to extract from file content
        class_name = cls.__name__
        class_pattern = re.compile(rf"class\s+{class_name}\s*(?:\([^)]*\))?\s*:")
        match = class_pattern.search(file_content)
        if match:
            start_pos = match.start()
            # Simple heuristic to find the end of the class definition
            # This is not perfect but works for many cases
            indent = 0
            for i, line in enumerate(file_content[start_pos:].split("\n")):
                if i == 0:
                    indent = len(line) - len(line.lstrip())
                    continue
                
                # If we find a line with the same or less indentation,
                # and it's not empty, consider it the end of the class
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    end_pos = start_pos + file_content[start_pos:].find("\n" + line)
                    return file_content[start_pos:end_pos]
            
            # If we didn't find the end, return the rest of the file
            return file_content[start_pos:]
        
        return ""


def generate_function_qa_pairs(function_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate diverse question-answer pairs for a function.
    
    Args:
        function_info: Dictionary with function details
        
    Returns:
        List of question-answer pairs
    """
    qa_pairs = []
    
    # Skip if no meaningful info
    if not function_info or not function_info.get("name"):
        return []
    
    name = function_info.get("name", "")
    doc = function_info.get("doc", "")
    signature = function_info.get("signature", "()")
    summary = function_info.get("summary", "")
    source = function_info.get("source", "")
    
    # 1. Basic function purpose question
    if doc:
        qa_pairs.append({
            "question": f"What does the function `{name}{signature}` do?",
            "answer": doc
        })
    
    # 2. Parameter usage questions
    parameters = function_info.get("parameters", {})
    if parameters:
        param_descriptions = []
        for param_name, param_info in parameters.items():
            if param_info.get("description"):
                param_descriptions.append(f"- `{param_name}`: {param_info.get('description')}")
        
        if param_descriptions:
            qa_pairs.append({
                "question": f"What are the parameters of `{name}`?",
                "answer": "\n".join(param_descriptions)
            })
    
    # 3. Return value question
    return_info = function_info.get("return_info", {})
    if return_info and return_info.get("description"):
        qa_pairs.append({
            "question": f"What does `{name}` return?",
            "answer": return_info.get("description", "")
        })
    
    # 4. Example usage question
    examples = function_info.get("examples", [])
    if examples:
        qa_pairs.append({
            "question": f"How do I use `{name}`? Show me an example.",
            "answer": "\n".join(examples)
        })
    
    # 5. Error handling question
    exceptions = function_info.get("exceptions", [])
    if exceptions:
        exception_descriptions = []
        for exc in exceptions:
            if exc.get("description"):
                exception_descriptions.append(f"- `{exc.get('type')}`: {exc.get('description')}")
        
        if exception_descriptions:
            qa_pairs.append({
                "question": f"What errors can `{name}` raise?",
                "answer": "\n".join(exception_descriptions)
            })
    
    # 6. Implementation details (for advanced users)
    if source:
        qa_pairs.append({
            "question": f"Show me the implementation of `{name}`.",
            "answer": source
        })
    
    return qa_pairs


def generate_class_qa_pairs(class_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate diverse question-answer pairs for a class.
    
    Args:
        class_info: Dictionary with class details
        
    Returns:
        List of question-answer pairs
    """
    qa_pairs = []
    
    # Skip if no meaningful info
    if not class_info or not class_info.get("name"):
        return []
    
    name = class_info.get("name", "")
    doc = class_info.get("doc", "")
    content = class_info.get("content", "")
    
    # 1. Class purpose question
    if doc:
        qa_pairs.append({
            "question": f"What is the purpose of the `{name}` class?",
            "answer": doc
        })
    
    # 2. Class implementation
    if content:
        qa_pairs.append({
            "question": f"Show me the implementation of the `{name}` class.",
            "answer": content
        })
    
    return qa_pairs


async def generate_enhanced_llm_qa_pairs(data: Dict[str, Any], max_pairs_per_item: int = 5) -> List[Dict[str, str]]:
    """Generate enhanced question-answer pairs using LLM-based generation.
    
    Args:
        data: Repository data containing files and their content
        max_pairs_per_item: Maximum number of QA pairs to generate per item
        
    Returns:
        List of question-answer pairs with varied formats including reversed QA pairs
    """
    if not LLM_GENERATOR_AVAILABLE:
        logger.warning("LLM generator not available, falling back to basic QA generation")
        return generate_enhanced_qa_pairs(data) if METHOD_VALIDATOR_AVAILABLE else []
    
    all_qa_pairs = []
    processing_tasks = []
    
    # Collect processing tasks for each file
    for file in data["files"]:
        file_path = file["path"]
        content = file["content"]
        
        # Process Python files
        if file_path.endswith(".py"):
            # Process the whole file
            processing_tasks.append(
                generate_code_qa_pairs(
                    code_content=content, 
                    temperature=None,  # Use random temperature variation
                    max_pairs=max_pairs_per_item
                )
            )
            
            # Extract functions and classes to process individually
            import re
            
            # Find function definitions
            function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            functions = re.findall(function_pattern, content)
            
            # Find class definitions
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            classes = re.findall(class_pattern, content)
            
            # Add tasks for processing each function
            for func_name in functions[:5]:  # Limit to top 5 functions to avoid too many requests
                processing_tasks.append(
                    generate_code_qa_pairs(
                        code_content=content, 
                        function_name=func_name,
                        temperature=None,  # Use random temperature variation
                        max_pairs=max_pairs_per_item // 2  # Fewer pairs for individual functions
                    )
                )
            
            # Add tasks for processing each class
            for class_name in classes[:3]:  # Limit to top 3 classes
                processing_tasks.append(
                    generate_code_qa_pairs(
                        code_content=content, 
                        function_name=class_name,
                        temperature=None,  # Use random temperature variation
                        max_pairs=max_pairs_per_item // 2
                    )
                )
        
        # Process Markdown files
        elif file_path.endswith(".md"):
            # Process the whole document
            processing_tasks.append(
                generate_markdown_qa_pairs(
                    markdown_content=content, 
                    temperature=None,  # Use random temperature variation
                    max_pairs=max_pairs_per_item
                )
            )
            
            # Process sections if available
            if "sections" in file:
                for section in file["sections"][:5]:  # Limit to top 5 sections
                    section_title = section.get("title")
                    section_content = section.get("content")
                    
                    if section_title and section_content and len(section_content) > 100:
                        processing_tasks.append(
                            generate_markdown_qa_pairs(
                                markdown_content=section_content,
                                section_title=section_title,
                                temperature=None,  # Use random temperature variation
                                max_pairs=max_pairs_per_item // 2
                            )
                        )
            
            # Process code blocks if available
            if "code_blocks" in file:
                python_blocks = [
                    block for block in file["code_blocks"]
                    if block.get("language", "").lower() in ["python", "py"] 
                    and len(block.get("content", "")) > 50
                ]
                
                for i, block in enumerate(python_blocks[:3]):  # Limit to top 3 code blocks
                    code = block.get("content", "")
                    
                    processing_tasks.append(
                        generate_code_qa_pairs(
                            code_content=code,
                            temperature=None,  # Use random temperature variation
                            max_pairs=max_pairs_per_item // 3  # Even fewer pairs for code blocks
                        )
                    )
    
    # Process all tasks concurrently for efficiency
    logger.info(f"Processing {len(processing_tasks)} tasks for QA generation")
    qa_batches = await asyncio.gather(*processing_tasks)
    
    # Add context to code block QA pairs
    code_block_counter = 0
    for file in data["files"]:
        if file_path.endswith(".md") and "code_blocks" in file:
            python_blocks = [
                block for block in file["code_blocks"]
                if block.get("language", "").lower() in ["python", "py"]
                and len(block.get("content", "")) > 50
            ]
            
            for i, block in enumerate(python_blocks[:3]):
                if code_block_counter < len(qa_batches):
                    # Add file and block context to each pair
                    for pair in qa_batches[code_block_counter]:
                        pair["question"] = f"[From {file['path']} code block {i+1}] {pair['question']}"
                    
                    code_block_counter += 1
    
    # Combine all QA pairs
    for batch in qa_batches:
        all_qa_pairs.extend(batch)
    
    # Generate reverse QA pairs
    logger.info(f"Generating reverse QA pairs from {len(all_qa_pairs)} original pairs")
    reverse_pairs = await generate_reverse_qa_pairs(
        all_qa_pairs, 
        temperature=None,  # Use random temperature variation
        max_reverse_pairs=max(len(all_qa_pairs) // 4, 5)  # 25% of original pairs, min 5
    )
    
    # Add reverse pairs to the mix
    all_qa_pairs.extend(reverse_pairs)
    
    # Deduplicate and validate all QA pairs
    logger.info(f"Validating and enhancing {len(all_qa_pairs)} QA pairs")
    original_count = len(all_qa_pairs)
    
    # Group QA pairs by file for validation
    file_qa_pairs = {}
    for file in data["files"]:
        file_path = file["path"]
        content = file["content"]
        file_qa_pairs[file_path] = []
        
        # Find QA pairs related to this file
        for pair in all_qa_pairs:
            question = pair.get("question", "")
            if file_path in question or (file_path.endswith(".py") and "function" in question.lower()):
                file_qa_pairs[file_path].append(pair)
    
    # Validate and enhance QA pairs for each file
    validated_qa_pairs = []
    for file_path, pairs in file_qa_pairs.items():
        if pairs:
            file_content = next((f["content"] for f in data["files"] if f["path"] == file_path), "")
            validated = await validate_and_enhance_qa_pairs(
                qa_pairs=pairs,
                original_content=file_content,
                deduplicate=True
            )
            validated_qa_pairs.extend(validated)
    
    # Add remaining QA pairs that weren't associated with specific files
    remaining_pairs = [
        pair for pair in all_qa_pairs 
        if not any(pair in file_pairs for file_pairs in file_qa_pairs.values())
    ]
    validated_qa_pairs.extend(remaining_pairs)
    
    # Final deduplication
    final_qa_pairs = detect_duplicate_pairs(validated_qa_pairs, similarity_threshold=65)
    
    logger.info(f"Generated {len(final_qa_pairs)} validated QA pairs (removed {original_count - len(final_qa_pairs)} duplicates/invalid pairs)")
    return final_qa_pairs


def debug_format_dataset():
    """Simple debug function to test dataset formatting functionality."""
    # Create a temporary directory and files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a test Python file
        test_data = {
            "files": [
                {
                    "path": "test.py",
                    "content": """
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers
        
    Returns:
        The average of the numbers
        
    Raises:
        ValueError: If the list is empty
    \"\"\"
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
"""
                },
                {
                    "path": "README.md",
                    "content": """# Test Project
                    
This is a test project for the DuaLipa formatter.

## Installation

To install the package, run:

```bash
pip install testproject
```

## Usage

Here's how to use the package:

```python
from testproject import calculate_average

result = calculate_average([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```
""",
                    "sections": [
                        {"title": "Test Project", "content": "This is a test project for the DuaLipa formatter.", "level": 1},
                        {"title": "Installation", "content": "To install the package, run:\n\n```bash\npip install testproject\n```", "level": 2},
                        {"title": "Usage", "content": "Here's how to use the package:\n\n```python\nfrom testproject import calculate_average\n\nresult = calculate_average([1, 2, 3, 4, 5])\nprint(result)  # Output: 3.0\n```", "level": 2}
                    ],
                    "code_blocks": [
                        {"language": "bash", "content": "pip install testproject"},
                        {"language": "python", "content": "from testproject import calculate_average\n\nresult = calculate_average([1, 2, 3, 4, 5])\nprint(result)  # Output: 3.0"}
                    ]
                }
            ]
        }
        
        # Write test data to a file
        input_file = os.path.join(temp_dir, "input.json")
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4)
            
        # Create output file path
        output_file = os.path.join(temp_dir, "output.json")
        
        # Test formatting with and without LLM
        print("Testing dataset formatting with standard methods...")
        format_for_lora(input_file, output_file, use_llm=False)
        
        with open(output_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)
            
        print(f"Generated {len(result_data.get('qa_pairs', []))} QA pairs without LLM")
        
        # Test with LLM if available
        if LLM_GENERATOR_AVAILABLE:
            print("\nTesting dataset formatting with LLM...")
            llm_output_file = os.path.join(temp_dir, "output_llm.json")
            format_for_lora(input_file, llm_output_file, use_llm=True, max_pairs_per_item=3)
            
            with open(llm_output_file, "r", encoding="utf-8") as f:
                llm_result_data = json.load(f)
                
            print(f"Generated {len(llm_result_data.get('qa_pairs', []))} QA pairs with LLM")
            
            # Show a sample of QA pairs
            print("\nSample QA pairs:")
            for i, pair in enumerate(llm_result_data.get('qa_pairs', [])[:3]):
                print(f"\nPair {i+1}:")
                print(f"Q: {pair.get('question', '')[:100]}...")
                print(f"A: {pair.get('answer', '')[:100]}...")
        
    except Exception as e:
        print(f"Debug test failed: {e}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nDebug test completed and temporary files cleaned up")


def demo_format_dataset() -> None:
    """Demonstrate the dataset formatting functionality with examples.
    
    This function shows how to use the main components of the dataset formatter:
    1. Creating sample extracted data
    2. Formatting the data into QA pairs with basic generation
    3. Displaying the resulting QA pairs
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("Dataset Formatting Demo")
        logger.info("======================")
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample extracted data
            sample_data = [
                {
                    "type": "code",
                    "language": "python",
                    "path": "sample_repo/utils.py",
                    "content": """
def calculate_average(numbers):
    \"\"\"
    Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers to average
        
    Returns:
        The average value
    \"\"\"
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class DataProcessor:
    \"\"\"A class for processing data collections.\"\"\"
    
    def __init__(self, data):
        \"\"\"Initialize with a data collection.\"\"\"
        self.data = data
    
    def process(self):
        \"\"\"Process the data and return results.\"\"\"
        return [item * 2 for item in self.data]
"""
                },
                {
                    "type": "documentation",
                    "language": "markdown",
                    "path": "sample_repo/README.md",
                    "content": """# Sample Project

This is a sample project for demonstrating dataset formatting.

## Installation

Install the package using pip:

```bash
pip install sample-project
```

## Usage

Basic usage example:

```python
from sample_project import calculate_average

result = calculate_average([1, 2, 3, 4, 5])
print(f"Average: {result}")
```
""",
                    "sections": {
                        "Installation": "Install the package using pip:\n\n```bash\npip install sample-project\n```",
                        "Usage": "Basic usage example:\n\n```python\nfrom sample_project import calculate_average\n\nresult = calculate_average([1, 2, 3, 4, 5])\nprint(f\"Average: {result}\")\n```"
                    }
                }
            ]
            
            # Save sample data to a JSON file
            input_file = temp_path / "sample_data.json"
            with open(input_file, "w") as f:
                json.dump(sample_data, f, indent=2)
            
            # Format the dataset
            logger.info("\n1. Formatting dataset with basic generation:")
            output_file = temp_path / "formatted_data.json"
            
            # Format the dataset without LLM
            stats = format_for_lora(
                str(input_file),
                str(output_file),
                use_llm=False,
                max_pairs_per_item=3
            )
            
            # Display statistics
            logger.info("\nFormatting Statistics:")
            logger.info(f"Total items processed: {stats['total_items_processed']}")
            logger.info(f"Total QA pairs generated: {stats['total_qa_pairs']}")
            logger.info(f"Code items: {stats['code_items']}")
            logger.info(f"Documentation items: {stats['documentation_items']}")
            logger.info(f"Basic generated pairs: {stats['basic_generated_pairs']}")
            
            # Load and display the formatted data
            with open(output_file, "r") as f:
                formatted_data = json.load(f)
            
            logger.info("\n2. Sample QA pairs generated:")
            for i, qa_pair in enumerate(formatted_data[:5], 1):
                logger.info(f"\nPair {i}:")
                logger.info(f"Question: {qa_pair['question']}")
                logger.info(f"Answer: {qa_pair['answer'][:100]}..." if len(qa_pair['answer']) > 100 else f"Answer: {qa_pair['answer']}")
            
            if len(formatted_data) > 5:
                logger.info(f"\n... and {len(formatted_data) - 5} more pairs")
            
            # Test the LLM availability (but don't actually use it)
            llm_available = check_litellm_available()
            logger.info(f"\n3. LLM availability for enhanced generation: {'Available' if llm_available else 'Not available'}")
            
            if llm_available:
                logger.info("LLM-based generation could be used with use_llm=True")
            else:
                logger.info("LLM-based generation requires LiteLLM to be installed and configured")
            
            # Clean up
            logger.info("\nCleaning up temporary files...")
            
        logger.info("\nDataset Formatting Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in dataset formatting demo: {e}")


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_format_dataset()
    
    # Process command line arguments if provided
    if len(sys.argv) > 1:
        try:
            # Check for debug testing
            if sys.argv[1] == "--debug-test":
                logger.info("Running debug test...")
                
                # Create a temporary file with sample data
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                    temp_path = temp_file.name
                    sample_data = [
                        {
                            "type": "code",
                            "language": "python",
                            "path": "test_file.py",
                            "content": """
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
"""
                        }
                    ]
                    json.dump(sample_data, temp_file)
                
                # Create output file path
                output_path = "test_output.json"
                
                # Test both with and without LLM
                logger.info("Testing without LLM:")
                stats_no_llm = format_for_lora(temp_path, output_path, use_llm=False)
                logger.info(f"Generated {stats_no_llm['total_qa_pairs']} QA pairs without LLM")
                
                logger.info("\nTesting with LLM:")
                stats_with_llm = format_for_lora(temp_path, output_path, use_llm=True)
                logger.info(f"Generated {stats_with_llm['total_qa_pairs']} QA pairs with LLM")
                
                # Display sample output
                with open(output_path, "r") as f:
                    formatted_data = json.load(f)
                
                if formatted_data:
                    logger.info("\nSample QA pair:")
                    logger.info(f"Question: {formatted_data[0]['question']}")
                    logger.info(f"Answer: {formatted_data[0]['answer']}")
                
                # Clean up
                os.unlink(temp_path)
                os.unlink(output_path)
                logger.info("Debug test completed")
                
            else:
                # Normal execution with input and output files
                input_file = sys.argv[1]
                output_file = sys.argv[2] if len(sys.argv) > 2 else "formatted_dataset.json"
                
                use_llm = "--use-llm" in sys.argv
                
                if "--max-pairs" in sys.argv and sys.argv.index("--max-pairs") + 1 < len(sys.argv):
                    max_pairs = int(sys.argv[sys.argv.index("--max-pairs") + 1])
                else:
                    max_pairs = DEFAULT_MAX_PAIRS_PER_ITEM
                
                logger.info(f"Processing input file: {input_file}")
                logger.info(f"Output file: {output_file}")
                logger.info(f"Using LLM: {use_llm}")
                logger.info(f"Max pairs per item: {max_pairs}")
                
                start_time = time.time()
                stats = format_for_lora(
                    input_file,
                    output_file,
                    use_llm=use_llm,
                    max_pairs_per_item=max_pairs
                )
                end_time = time.time()
                
                logger.info(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
                logger.info(f"Total items processed: {stats['total_items_processed']}")
                logger.info(f"Total QA pairs generated: {stats['total_qa_pairs']}")
                logger.info(f"Output saved to: {output_file}")
                
        except Exception as e:
            logger.error(f"Error processing command line arguments: {e}")
