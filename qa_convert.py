#!/usr/bin/env python3
"""
Convert AST extraction output to QA-compatible format.

This script takes the output of the AST extractor and converts it to
a format compatible with the QA module, with appropriate sections and metadata.
"""

import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

def generate_uuid():
    """Generate a unique UUID."""
    return str(uuid.uuid4())

def ast_to_qa_format(ast_output_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert AST extraction output to QA-compatible format.
    
    Args:
        ast_output_path: Path to the AST extraction output JSON
        output_path: Optional path to save the QA-compatible output
        
    Returns:
        Dictionary with QA-compatible format
    """
    # Load AST output
    try:
        with open(ast_output_path, 'r', encoding='utf-8') as f:
            ast_data = json.load(f)
    except Exception as e:
        print(f"Error loading AST output: {e}")
        return {"error": str(e)}
    
    # Initialize QA output
    qa_output = {
        "sections": [],
        "extraction_metadata": {
            "model_used": "ast-extraction-model",
            "timestamp": f"{os.path.getmtime(ast_output_path)}",
            "source": ast_output_path,
            "statistics": {}
        }
    }
    
    # Process results
    results = ast_data.get("results", [])
    if not results and "file_path" in ast_data:
        # Direct file result
        results = [ast_data]
        
    # Add statistics
    if "stats" in ast_data:
        qa_output["extraction_metadata"]["statistics"] = ast_data["stats"]
    
    # Process each file result
    for file_result in results:
        file_path = file_result.get("file_path", "unknown")
        language = file_result.get("language", "unknown")
        file_name = os.path.basename(file_path)
        
        # Create file section
        file_section = {
            "uuid": generate_uuid(),
            "type": "code_file",
            "title": file_name,
            "content": f"# {file_name}\n\nFile: {file_path}\nLanguage: {language}\n",
            "extraction_focus": "code structure",
            "summary_instructions": f"Generate QA pairs about the structure of {file_name}",
            "breadcrumb": [file_name]
        }
        
        # Add docstring if available
        if file_result.get("docstring"):
            file_section["content"] += f"\nDocstring:\n{file_result['docstring']}\n"
        
        qa_output["sections"].append(file_section)
        
        # Add classes section if available
        if file_result.get("classes"):
            class_content = "# Classes\n\n"
            for cls in file_result.get("classes", []):
                class_content += f"## {cls['name']}\n\n"
                
                # Inheritance
                if cls.get("inherits_from"):
                    class_content += f"Inherits from: {', '.join(cls['inherits_from'])}\n\n"
                
                # Inner classes
                if cls.get("inner_classes"):
                    inner_names = [ic['name'] for ic in cls.get("inner_classes", [])]
                    class_content += f"Inner classes: {', '.join(inner_names)}\n\n"
                
                # Methods
                if cls.get("methods"):
                    class_content += "### Methods:\n\n"
                    for method in cls.get("methods", []):
                        params = ", ".join(method.get("parameters", []))
                        decorator_str = ""
                        if method.get("decorators"):
                            decorator_str = f" (decorators: {', '.join(method['decorators'])})"
                        class_content += f"- {method['name']}({params}){decorator_str}\n"
                    class_content += "\n"
            
            # Add classes section
            class_section = {
                "uuid": generate_uuid(),
                "type": "code_structure",
                "title": f"Classes in {file_name}",
                "content": class_content,
                "extraction_focus": "class structures",
                "summary_instructions": f"Generate QA pairs about classes in {file_name}",
                "breadcrumb": [file_name, "Classes"]
            }
            qa_output["sections"].append(class_section)
        
        # Add functions section if available
        if file_result.get("functions"):
            func_content = "# Functions\n\n"
            for func in file_result.get("functions", []):
                params = ", ".join(func.get("parameters", []))
                decorator_str = ""
                if func.get("decorators"):
                    decorator_str = f" (decorators: {', '.join(func['decorators'])})"
                func_content += f"## {func['name']}({params}){decorator_str}\n\n"
                
                if func.get("docstring"):
                    func_content += f"{func['docstring']}\n\n"
            
            # Add functions section
            func_section = {
                "uuid": generate_uuid(),
                "type": "code_structure",
                "title": f"Functions in {file_name}",
                "content": func_content,
                "extraction_focus": "function usage",
                "summary_instructions": f"Generate QA pairs about functions in {file_name}",
                "breadcrumb": [file_name, "Functions"]
            }
            qa_output["sections"].append(func_section)
        
        # Add imports section if available
        if file_result.get("imports"):
            import_content = "# Imports\n\n"
            for imp in file_result.get("imports", []):
                import_content += f"- {imp['text']}\n"
            
            # Add imports section
            import_section = {
                "uuid": generate_uuid(),
                "type": "code_structure",
                "title": f"Imports in {file_name}",
                "content": import_content,
                "extraction_focus": "dependencies",
                "summary_instructions": f"Generate QA pairs about dependencies in {file_name}",
                "breadcrumb": [file_name, "Imports"]
            }
            qa_output["sections"].append(import_section)
    
    # Save output if requested
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_output, f, indent=2)
            print(f"Saved QA-compatible output to {output_path}")
        except Exception as e:
            print(f"Error saving QA output: {e}")
    
    return qa_output

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert AST extraction output to QA-compatible format")
    parser.add_argument("ast_output", help="Path to the AST extraction output JSON")
    parser.add_argument("--output", help="Path to save the QA-compatible output")
    
    args = parser.parse_args()
    
    ast_to_qa_format(args.ast_output, args.output or args.ast_output.replace(".json", "_qa.json"))

if __name__ == "__main__":
    main()