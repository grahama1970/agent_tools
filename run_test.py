import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import and run the test module
from src.agent_tools.dualipa.tests.test_smoke_compliance import *

if __name__ == "__main__":
    print("Running smoke tests for DuaLipa...")
    
    # Run tests
    test_version()
    test_module_imports()
    test_modules_have_demo_functions()
    test_modules_have_main_block()
    test_modules_have_docstrings()
    test_modules_have_required_functions()
    test_code_extractor_basic_functionality()
    test_language_detection_basic_functionality()
    test_github_utils_basic_functionality()
    test_markdown_parser_basic_functionality()
    test_validate_and_enhance_qa_pairs()
    test_llm_generator_config()
    test_format_dataset_basic_functionality()
    
    print("All smoke tests passed!") 