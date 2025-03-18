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
import asyncio

# At the import section
try:
    from agent_tools.dualipa.code_extractor import initialize_stats_dict
    STATS_IMPORT_AVAILABLE = True
except ImportError:
    STATS_IMPORT_AVAILABLE = False
    logger.warning("Could not import stats initialization from code_extractor.py")

# Attempt to import method_validator components
try:
    from agent_tools.method_validator.analyzer import MethodAnalyzer, MethodInfo
    from agent_tools.method_validator.cache import AnalysisCache
    METHOD_VALIDATOR_AVAILABLE = True
    logger.info("method_validator is available and will be used for enhanced code analysis")
except ImportError:
    METHOD_VALIDATOR_AVAILABLE = False
    logger.warning("method_validator not available. Using basic function detection.")

# Attempt to import LLM generator components
try:
    from .llm_generator import (
        generate_code_qa_pairs, 
        generate_markdown_qa_pairs, 
        generate_reverse_qa_pairs,
        generate_qa_pairs_from_text
    )
    from .qa_validator import (
        validate_and_enhance_qa_pairs,
        detect_duplicate_pairs,
        validate_function_qa_pair
    )
    LLM_GENERATOR_AVAILABLE = True
    logger.info("LLM generator is available and will be used for enhanced QA pair generation")
except ImportError:
    LLM_GENERATOR_AVAILABLE = False
    logger.warning("LLM generator not available. Using basic QA generation.")

def check_litellm_available() -> bool:
    """Simple function to check if LiteLLM is available."""
    try:
        import litellm
        return True
    except ImportError:
        return False

def format_for_lora(input_file: str, output_file: str, use_llm: bool = True, max_pairs_per_item: int = 5) -> Dict[str, Any]:
    """
    Format extracted data into QA pairs for training data.
    
    Args:
        input_file: Path to input JSON file with extracted data
        output_file: Path to output JSONL file for training data
        use_llm: Whether to use LLM for generating QA pairs
        max_pairs_per_item: Maximum number of QA pairs per item
        
    Returns:
        Dictionary with statistics
    """
    # Initialize stats with standardized format
    stats = initialize_stats_dict() if STATS_IMPORT_AVAILABLE else {
        "total_items": 0,
        "qa_pairs": 0,
        "errors": [],
        "output_file": output_file
    }
    
    try:
        # Validate that the input file exists
        if not os.path.exists(input_file):
            error_msg = f"Input file not found: {input_file}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load JSON data
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Count items for stats
            stats["total_items"] = len(data.get("files", []))
            for file in data.get("files", []):
                if file.get("path", "").endswith(".py"):
                    stats["code_items"] = stats.get("code_items", 0) + 1
                elif file.get("path", "").endswith((".md", ".rst", ".txt")):
                    stats["documentation_items"] = stats.get("documentation_items", 0) + 1

            formatted_data = {"qa_pairs": []}
            
            # Determine QA generation strategy
            if LLM_GENERATOR_AVAILABLE and use_llm:
                logger.info("Using LLM-based QA pair generation.")
                qa_pairs = asyncio.run(generate_enhanced_llm_qa_pairs(data, max_pairs_per_item))
                formatted_data["qa_pairs"] = qa_pairs
            elif METHOD_VALIDATOR_AVAILABLE:
                logger.info("Using advanced method analysis to generate QA pairs.")
                formatted_data["qa_pairs"] = generate_enhanced_qa_pairs(data)
            else:
                logger.info("Using basic function detection for QA pair generation.")
                generate_basic_qa_pairs(data, formatted_data)
            
            # Update stats
            stats["qa_pairs"] = len(formatted_data["qa_pairs"])
            if not (LLM_GENERATOR_AVAILABLE and use_llm):
                stats["basic_generated_pairs"] = len(formatted_data["qa_pairs"])
            
            # Write final results to output file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(formatted_data, f, indent=4)
            
            logger.info(f"Dataset formatted and saved to {output_file}. Generated {len(formatted_data['qa_pairs'])} QA pairs.")
            print(f"Dataset formatted and saved to {output_file}. Generated {len(formatted_data['qa_pairs'])} QA pairs.")
        
        except Exception as e:
            error_msg = f"Error processing input file: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
    except Exception as e:
        logger.error(f"Unexpected error during formatting: {e}")
        raise
    
    return stats

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
        List of question-answer pairs with varied formats.
    """
    qa_pairs = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        module_files = {}
        for file in data["files"]:
            if not file["path"].endswith(".py"):
                continue
            if file["path"].endswith("__init__.py") or not file["content"].strip():
                continue
            
            rel_path = os.path.basename(file["path"])
            module_name = os.path.splitext(rel_path)[0]
            temp_file = os.path.join(temp_dir, f"{module_name}.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(file["content"])
            module_files[module_name] = {
                "path": temp_file,
                "content": file["content"]
            }
        
        # Insert temp dir into sys.path for dynamic import
        sys.path.insert(0, temp_dir)
        analyzer = MethodAnalyzer(include_builtins=False)
        
        for module_name, file_info in module_files.items():
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_info["path"])
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for name, obj in inspect.getmembers(module):
                        if name.startswith("_"):
                            continue
                        if inspect.isfunction(obj) or inspect.ismethod(obj):
                            method_info = analyze_function(name, obj, module_name)
                            if method_info:
                                qa_pairs.extend(generate_function_qa_pairs(method_info))
                        elif inspect.isclass(obj):
                            class_info = {
                                "name": name,
                                "doc": inspect.getdoc(obj) or "",
                                "module": module_name,
                                "content": extract_class_source(obj, file_info["content"])
                            }
                            qa_pairs.extend(generate_class_qa_pairs(class_info))
                            for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                                if not method_name.startswith("_"):
                                    method_info = analyze_function(method_name, method_obj, module_name, class_name=name)
                                    if method_info:
                                        qa_pairs.extend(generate_function_qa_pairs(method_info))
            except Exception as e:
                logger.error(f"Error analyzing module {module_name}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error during enhanced QA pair generation: {e}")
    finally:
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        shutil.rmtree(temp_dir)
    
    return qa_pairs

def analyze_function(name: str, obj: Any, module_name: str, class_name: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a function or method using method_validator if available, otherwise basic inspection.
    
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
            method_info = MethodInfo(obj, full_name)
            return method_info.to_dict()
        else:
            return {
                "name": full_name,
                "doc": inspect.getdoc(obj) or "",
                "signature": str(inspect.signature(obj)),
                "module": module_name,
                "summary": (inspect.getdoc(obj) or "").split("\n")[0] if inspect.getdoc(obj) else "",
                "parameters": {
                    pname: {"description": ""} for pname in inspect.signature(obj).parameters
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
        return inspect.getsource(cls)
    except (IOError, TypeError):
        class_name = cls.__name__
        class_pattern = re.compile(rf"class\s+{class_name}\s*(?:\([^)]*\))?\s*:")
        match = class_pattern.search(file_content)
        if match:
            start_pos = match.start()
            indent = 0
            lines = file_content[start_pos:].split("\n")
            for i, line in enumerate(lines):
                if i == 0:
                    indent = len(line) - len(line.lstrip())
                    continue
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    end_pos = start_pos + file_content[start_pos:].find("\n" + line)
                    return file_content[start_pos:end_pos]
            return file_content[start_pos:]
        return ""

def generate_function_qa_pairs(function_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate diverse QA pairs for a function.
    
    Args:
        function_info: Dictionary with function details
        
    Returns:
        List of QA pairs
    """
    qa_pairs = []
    if not function_info or not function_info.get("name"):
        return []
    
    name = function_info.get("name", "")
    doc = function_info.get("doc", "")
    signature = function_info.get("signature", "()")
    summary = function_info.get("summary", "")
    source = function_info.get("source", "")
    
    if doc:
        qa_pairs.append({
            "question": f"What does the function `{name}{signature}` do?",
            "answer": doc
        })
    
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
    
    # Return info, examples, exceptions can be added if method_validator provided them
    return_info = function_info.get("return_info", {})
    if return_info and return_info.get("description"):
        qa_pairs.append({
            "question": f"What does `{name}` return?",
            "answer": return_info.get("description", "")
        })
    
    examples = function_info.get("examples", [])
    if examples:
        qa_pairs.append({
            "question": f"How do I use `{name}`? Show me an example.",
            "answer": "\n".join(examples)
        })
    
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
    
    if source:
        qa_pairs.append({
            "question": f"Show the implementation of `{name}`.",
            "answer": source
        })
    
    return qa_pairs

def generate_class_qa_pairs(class_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate diverse question-answer pairs for a class.
    
    Args:
        class_info: Dictionary with class details
        
    Returns:
        List of QA pairs
    """
    qa_pairs = []
    if not class_info or not class_info.get("name"):
        return []
    
    name = class_info.get("name", "")
    doc = class_info.get("doc", "")
    content = class_info.get("content", "")
    
    if doc:
        qa_pairs.append({
            "question": f"What is the purpose of the `{name}` class?",
            "answer": doc
        })
    
    if content:
        qa_pairs.append({
            "question": f"Show the implementation of the `{name}` class.",
            "answer": content
        })
    
    return qa_pairs

async def generate_enhanced_llm_qa_pairs(data: Dict[str, Any], max_pairs_per_item: int = 5) -> List[Dict[str, str]]:
    """Generate enhanced question-answer pairs using LLM-based generation.
    
    Args:
        data: Repository data containing files and their content
        max_pairs_per_item: Maximum number of QA pairs to generate per item
        
    Returns:
        List of question-answer pairs (including reverse pairs)
    """
    if not LLM_GENERATOR_AVAILABLE:
        logger.warning("LLM generator not available, falling back to method_validator or basic QA.")
        return generate_enhanced_qa_pairs(data) if METHOD_VALIDATOR_AVAILABLE else []
    
    all_qa_pairs = []
    processing_tasks = []
    
    # Collect tasks for each file
    for file in data["files"]:
        file_path = file["path"]
        content = file["content"]
        
        if file_path.endswith(".py"):
            # Generate QA for the entire file
            processing_tasks.append(
                generate_code_qa_pairs(
                    code_content=content,
                    temperature=None,
                    max_pairs=max_pairs_per_item
                )
            )
            
            # Extract functions and classes
            function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            functions = re.findall(function_pattern, content)
            classes = re.findall(class_pattern, content)
            
            for func_name in functions[:5]:
                processing_tasks.append(
                    generate_code_qa_pairs(
                        code_content=content,
                        function_name=func_name,
                        temperature=None,
                        max_pairs=max_pairs_per_item // 2
                    )
                )
            for class_name in classes[:3]:
                processing_tasks.append(
                    generate_code_qa_pairs(
                        code_content=content,
                        function_name=class_name,
                        temperature=None,
                        max_pairs=max_pairs_per_item // 2
                    )
                )
        
        elif file_path.endswith(".md"):
            # Generate QA for the entire markdown file
            processing_tasks.append(
                generate_markdown_qa_pairs(
                    markdown_content=content,
                    temperature=None,
                    max_pairs=max_pairs_per_item
                )
            )
            
            # If sections exist
            if "sections" in file:
                for section in file["sections"][:5]:
                    section_title = section.get("title")
                    section_content = section.get("content")
                    if section_title and section_content and len(section_content) > 100:
                        processing_tasks.append(
                            generate_markdown_qa_pairs(
                                markdown_content=section_content,
                                section_title=section_title,
                                temperature=None,
                                max_pairs=max_pairs_per_item // 2
                            )
                        )
            
            # If code blocks exist
            if "code_blocks" in file:
                python_blocks = [
                    block for block in file["code_blocks"]
                    if block.get("language", "").lower() in ["python", "py"] 
                       and len(block.get("content", "")) > 50
                ]
                for block in python_blocks[:3]:
                    code = block.get("content", "")
                    processing_tasks.append(
                        generate_code_qa_pairs(
                            code_content=code,
                            temperature=None,
                            max_pairs=max_pairs_per_item // 3
                        )
                    )
    
    # Run all tasks concurrently
    logger.info(f"Processing {len(processing_tasks)} tasks for QA generation.")
    qa_batches = await asyncio.gather(*processing_tasks)
    
    for batch in qa_batches:
        all_qa_pairs.extend(batch)
    
    logger.info(f"Generating reverse QA pairs from {len(all_qa_pairs)} original pairs")
    reverse_pairs = await generate_reverse_qa_pairs(
        all_qa_pairs,
        temperature=None,
        max_reverse_pairs=max(len(all_qa_pairs) // 4, 5)
    )
    all_qa_pairs.extend(reverse_pairs)
    
    # If we have a QA validator, we can do a final pass to deduplicate/validate
    if 'validate_and_enhance_qa_pairs' in globals() and 'detect_duplicate_pairs' in globals():
        logger.info("Validating and deduplicating QA pairs...")
        original_count = len(all_qa_pairs)
        
        # Group by file for validation
        file_qa_pairs = {}
        for file in data["files"]:
            fp = file["path"]
            file_qa_pairs[fp] = []
            for pair in all_qa_pairs:
                if fp in pair.get("question", ""):
                    file_qa_pairs[fp].append(pair)
        
        validated_qa_pairs = []
        for fp, pairs in file_qa_pairs.items():
            if pairs:
                file_content = next((f["content"] for f in data["files"] if f["path"] == fp), "")
                validated = await validate_and_enhance_qa_pairs(
                    qa_pairs=pairs,
                    original_content=file_content,
                    deduplicate=True
                )
                validated_qa_pairs.extend(validated)
        
        # Add pairs that weren't associated with any file path
        remaining_pairs = [
            p for p in all_qa_pairs
            if not any(p in file_pairs for file_pairs in file_qa_pairs.values())
        ]
        validated_qa_pairs.extend(remaining_pairs)
        
        final_qa_pairs = detect_duplicate_pairs(validated_qa_pairs, similarity_threshold=65)
        logger.info(f"Generated {len(final_qa_pairs)} validated QA pairs (removed {original_count - len(final_qa_pairs)} duplicates).")
        return final_qa_pairs
    
    return all_qa_pairs

def debug_format_dataset():
    """Simple debug function to test dataset formatting functionality."""
    temp_dir = tempfile.mkdtemp()
    
    try:
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
        
        input_file = os.path.join(temp_dir, "input.json")
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4)
            
        output_file = os.path.join(temp_dir, "output.json")
        print("Testing dataset formatting with standard methods (no LLM)...")
        format_for_lora(input_file, output_file, use_llm=False)
        
        with open(output_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        print(f"Generated {len(result_data.get('qa_pairs', []))} QA pairs without LLM")
        
        if LLM_GENERATOR_AVAILABLE:
            print("\nTesting dataset formatting with LLM enabled...")
            llm_output_file = os.path.join(temp_dir, "output_llm.json")
            format_for_lora(input_file, llm_output_file, use_llm=True, max_pairs_per_item=3)
            with open(llm_output_file, "r", encoding="utf-8") as f:
                llm_result_data = json.load(f)
            print(f"Generated {len(llm_result_data.get('qa_pairs', []))} QA pairs with LLM")
            
            print("\nSample QA pairs from LLM output:")
            for i, pair in enumerate(llm_result_data.get('qa_pairs', [])[:3]):
                print(f"\nPair {i+1}:")
                print(f"Q: {pair.get('question', '')[:100]}...")
                print(f"A: {pair.get('answer', '')[:100]}...")
        
    except Exception as e:
        print(f"Debug test failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nDebug test completed and temporary files cleaned up")

def demo_format_dataset() -> None:
    """Demonstrate the dataset formatting functionality with examples."""
    try:
        logger.info("Dataset Formatting Demo")
        logger.info("======================")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
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
            
            input_file = temp_path / "sample_data.json"
            with open(input_file, "w") as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info("\n1. Formatting dataset with basic generation:")
            output_file = temp_path / "formatted_data.json"
            
            stats_result = format_for_lora(
                str(input_file),
                str(output_file),
                use_llm=False,
                max_pairs_per_item=3
            )
            
            logger.info("\nFormatting Statistics:")
            logger.info(f"Total items processed: {stats_result.get('total_items_processed', 'N/A')}")
            logger.info(f"Total QA pairs generated: {stats_result.get('total_qa_pairs', 'N/A')}")
            logger.info(f"Code items: {stats_result.get('code_items', 'N/A')}")
            logger.info(f"Documentation items: {stats_result.get('documentation_items', 'N/A')}")
            logger.info(f"Basic generated pairs: {stats_result.get('basic_generated_pairs', 'N/A')}")
            
            with open(output_file, "r") as f:
                formatted_data = json.load(f)
            
            logger.info("\n2. Sample QA pairs generated:")
            for i, qa_pair in enumerate(formatted_data.get("qa_pairs", [])[:5], 1):
                logger.info(f"\nPair {i}:")
                logger.info(f"Question: {qa_pair['question']}")
                if len(qa_pair['answer']) > 100:
                    logger.info(f"Answer: {qa_pair['answer'][:100]}...")
                else:
                    logger.info(f"Answer: {qa_pair['answer']}")
            
            if len(formatted_data.get("qa_pairs", [])) > 5:
                logger.info(f"\n... and {len(formatted_data.get('qa_pairs', [])) - 5} more pairs")
            
            llm_available = check_litellm_available()
            logger.info(f"\n3. LLM availability for enhanced generation: {'Available' if llm_available else 'Not available'}")
            if llm_available:
                logger.info("LLM-based generation could be used with use_llm=True")
            else:
                logger.info("LLM-based generation requires LiteLLM to be installed and configured")
            
            logger.info("\nCleaning up temporary files...")
        
        logger.info("\nDataset Formatting Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in dataset formatting demo: {e}")

def generate_basic_code_qa_pairs(
    code_content: str,
    function_name: Optional[str] = None,
    max_pairs: int = 5
) -> List[Dict[str, str]]:
    """Generate basic QA pairs from code content without using an LLM.
    
    This function uses pattern matching and templates to create simple 
    QA pairs about the code.
    
    Args:
        code_content: The source code content
        function_name: Optional function name to focus on
        max_pairs: Maximum number of QA pairs to generate
        
    Returns:
        List of QA pairs as dictionaries with 'question' and 'answer' keys
    """
    qa_pairs = []
    
    # Extract docstring if available
    docstring_pattern = r'"""(.*?)"""'
    docstring_match = re.search(docstring_pattern, code_content, re.DOTALL)
    docstring = docstring_match.group(1).strip() if docstring_match else ""
    
    # If a specific function is requested, focus on that
    if function_name:
        function_pattern = fr'def\s+{re.escape(function_name)}\s*\((.*?)\):'
        function_match = re.search(function_pattern, code_content)
        if function_match:
            params = function_match.group(1).strip()
            qa_pairs.append({
                "question": f"What is the purpose of the '{function_name}' function?",
                "answer": f"The '{function_name}' function {docstring if docstring else 'performs operations on the provided input.'}"
            })
            if params:
                qa_pairs.append({
                    "question": f"What parameters does the '{function_name}' function accept?",
                    "answer": f"The '{function_name}' function accepts these parameters: {params}"
                })
    
    # General code questions
    if len(qa_pairs) < max_pairs:
        import_matches = re.findall(r'import\s+(\w+)|from\s+(\w+)(?:\.\w+)*\s+import', code_content)
        imports = [m[0] or m[1] for m in import_matches if m[0] or m[1]]
        if imports and len(qa_pairs) < max_pairs:
            imports_str = ", ".join(imports)
            qa_pairs.append({
                "question": "What external libraries or modules does this code use?",
                "answer": f"This code uses the following libraries or modules: {imports_str}"
            })
        if docstring and len(qa_pairs) < max_pairs:
            qa_pairs.append({
                "question": "What does this code do?",
                "answer": docstring
            })
    
    # If we still need more pairs, add generic questions
    while len(qa_pairs) < max_pairs:
        templates = [
            {
                "question": "How would you use this code in a project?",
                "answer": "This code can be integrated into a project by importing it and calling its functions with appropriate parameters."
            },
            {
                "question": "What are the main components of this code?",
                "answer": "The main components include the imported libraries, function definitions, and the implementation logic."
            },
            {
                "question": "Is there error handling in this code?",
                "answer": "Yes, the code implements error handling through try-except blocks." if "except" in code_content else "No explicit error handling was found in this code snippet."
            }
        ]
        for template in templates:
            if len(qa_pairs) < max_pairs and not any(p["question"] == template["question"] for p in qa_pairs):
                qa_pairs.append(template)
    
    return qa_pairs[:max_pairs]

if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_format_dataset()
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "--debug-test":
                logger.info("Running debug test...")
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_file:
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
                
                output_path = "test_output.json"
                logger.info("Testing without LLM:")
                stats_no_llm = format_for_lora(temp_path, output_path, use_llm=False)
                logger.info(f"Generated {stats_no_llm.get('total_qa_pairs', 0)} QA pairs without LLM")
                
                logger.info("Testing with LLM:")
                stats_with_llm = format_for_lora(temp_path, output_path, use_llm=True)
                logger.info(f"Generated {stats_with_llm.get('total_qa_pairs', 0)} QA pairs with LLM")
                
                with open(output_path, "r", encoding="utf-8") as f:
                    formatted_data = json.load(f)
                if formatted_data.get("qa_pairs"):
                    logger.info("\nSample QA pair:")
                    logger.info(f"Question: {formatted_data['qa_pairs'][0]['question']}")
                    logger.info(f"Answer: {formatted_data['qa_pairs'][0]['answer']}")
                
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
                    max_pairs = 5
                
                logger.info(f"Processing input file: {input_file}")
                logger.info(f"Output file: {output_file}")
                logger.info(f"Using LLM: {use_llm}")
                logger.info(f"Max pairs per item: {max_pairs}")
                
                start_time = asyncio.get_event_loop().time()
                stats_result = format_for_lora(input_file, output_file, use_llm=use_llm, max_pairs_per_item=max_pairs)
                end_time = asyncio.get_event_loop().time()
                
                logger.info(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
                logger.info(f"Total items processed: {stats_result.get('total_items_processed', 'N/A')}")
                logger.info(f"Total QA pairs generated: {stats_result.get('total_qa_pairs', 'N/A')}")
                logger.info(f"Output saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error processing command line arguments: {e}")
