#!/usr/bin/env python3
"""
HTML Report Generator for Validation Results.

This script generates detailed HTML reports for validation results, 
including side-by-side comparisons of input and output.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_validation_report")


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def escape_html(text: str) -> str:
    """Escape HTML special characters in a string."""
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apos;",
        ">": "&gt;",
        "<": "&lt;",
    }
    return "".join(html_escape_table.get(c, c) for c in text)


def generate_html_report(
    validation_results: Dict[str, Any],
    extraction_file: Path,
    expected_format_file: Path,
    output_path: Path,
    test_name: str
) -> None:
    """
    Generate an HTML report for validation results.
    
    Args:
        validation_results: The validation results
        extraction_file: Path to the extraction file
        expected_format_file: Path to the expected format file
        output_path: Path to save the HTML report
        test_name: Name of the test
    """
    logger.info(f"Generating HTML report for {test_name}")
    
    # Load extraction data and expected format
    extraction_data = load_json_file(extraction_file)
    expected_format = load_json_file(expected_format_file)
    
    if not extraction_data or not expected_format:
        logger.error(f"Could not load input files for {test_name}")
        return
    
    # Create base HTML template with tailwind CSS
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report - {test_name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide-static@latest/font/lucide.min.css">
    <style>
        pre {{ white-space: pre-wrap; overflow-x: auto; }}
        .content-card {{ max-height: 500px; overflow-y: auto; }}
        .error-card {{ max-height: 300px; overflow-y: auto; }}
    </style>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto px-4 py-8">
        <div class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-bold text-gray-800">{test_name} - Validation Report</h1>
            <div class="text-sm text-gray-500">Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        
        <!-- Overall Results Card -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-2xl font-semibold">Overall Results</h2>
                <div class="{'bg-green-100 text-green-800' if validation_results.get('valid', False) else 'bg-red-100 text-red-800'} px-4 py-2 rounded-full font-bold">
                    {('PASSED' if validation_results.get('valid', False) else 'FAILED')}
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div class="bg-gray-50 p-4 rounded-lg text-center">
                    <div class="text-3xl font-bold mb-1">{validation_results.get('overall_score', 0)}%</div>
                    <div class="text-gray-600">Overall Score</div>
                </div>"""
    
    # Add structure score if available
    if "structure_validation" in validation_results and validation_results["structure_validation"]:
        structure_score = validation_results["structure_validation"].get("score", 0)
        html += f"""
                <div class="bg-gray-50 p-4 rounded-lg text-center">
                    <div class="text-3xl font-bold mb-1">{structure_score}%</div>
                    <div class="text-gray-600">Structure Score</div>
                </div>"""
    
    # Add content score if available
    if "content_validation" in validation_results and validation_results["content_validation"]:
        content_score = validation_results["content_validation"].get("score", 0)
        html += f"""
                <div class="bg-gray-50 p-4 rounded-lg text-center">
                    <div class="text-3xl font-bold mb-1">{content_score}%</div>
                    <div class="text-gray-600">Content Score</div>
                </div>"""
    
    html += """
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">"""
    
    # Validation Components Status
    html += """
                <div>
                    <h3 class="text-lg font-semibold mb-3">Validation Components</h3>
                    <ul class="space-y-2">"""
    
    # Format validation
    format_valid = validation_results.get("format_validation", {}).get("valid", False)
    html += f"""
                        <li class="flex items-center">
                            <span class="{'text-green-500' if format_valid else 'text-red-500'} mr-2">
                                <i class="{"lucide-check-circle" if format_valid else "lucide-x-circle"}"></i>
                            </span>
                            <span>Format Validation</span>
                        </li>"""
    
    # Structure validation
    if "structure_validation" in validation_results:
        structure_valid = validation_results["structure_validation"].get("valid", False)
        html += f"""
                        <li class="flex items-center">
                            <span class="{'text-green-500' if structure_valid else 'text-red-500'} mr-2">
                                <i class="{"lucide-check-circle" if structure_valid else "lucide-x-circle"}"></i>
                            </span>
                            <span>Structure Validation</span>
                        </li>"""
    
    # Content validation
    if "content_validation" in validation_results:
        content_valid = validation_results["content_validation"].get("valid", False)
        html += f"""
                        <li class="flex items-center">
                            <span class="{'text-green-500' if content_valid else 'text-red-500'} mr-2">
                                <i class="{"lucide-check-circle" if content_valid else "lucide-x-circle"}"></i>
                            </span>
                            <span>Content Validation</span>
                        </li>"""
    
    # Structure consistency validation
    if "structure_consistency" in validation_results:
        consistency_valid = validation_results["structure_consistency"].get("valid", False)
        html += f"""
                        <li class="flex items-center">
                            <span class="{'text-green-500' if consistency_valid else 'text-red-500'} mr-2">
                                <i class="{"lucide-check-circle" if consistency_valid else "lucide-x-circle"}"></i>
                            </span>
                            <span>Structure Consistency</span>
                        </li>"""
    
    html += """
                    </ul>
                </div>"""
    
    # Statistics Section
    html += """
                <div>
                    <h3 class="text-lg font-semibold mb-3">Validation Statistics</h3>
                    <div class="space-y-3">"""
    
    # Structure statistics
    if "structure_validation" in validation_results:
        structure = validation_results["structure_validation"]
        passed = structure.get("passed_checks", 0)
        total = structure.get("total_checks", 0)
        percent = (passed / total * 100) if total > 0 else 0
        html += f"""
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm font-medium text-gray-700">Structure Checks</span>
                                <span class="text-sm font-medium text-gray-700">{passed}/{total} ({percent:.1f}%)</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2.5">
                                <div class="bg-blue-600 h-2.5 rounded-full" style="width: {percent}%"></div>
                            </div>
                        </div>"""
    
    # Content statistics
    if "content_validation" in validation_results:
        content = validation_results["content_validation"]
        passed = content.get("passed_checks", 0)
        total = content.get("total_checks", 0)
        percent = (passed / total * 100) if total > 0 else 0
        html += f"""
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm font-medium text-gray-700">Content Checks</span>
                                <span class="text-sm font-medium text-gray-700">{passed}/{total} ({percent:.1f}%)</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2.5">
                                <div class="bg-blue-600 h-2.5 rounded-full" style="width: {percent}%"></div>
                            </div>
                        </div>"""
    
    # Consistency statistics
    if "structure_consistency" in validation_results:
        consistency = validation_results["structure_consistency"]
        passed = consistency.get("passed_checks", 0)
        total = consistency.get("total_checks", 0)
        percent = (passed / total * 100) if total > 0 else 0
        html += f"""
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm font-medium text-gray-700">Consistency Checks</span>
                                <span class="text-sm font-medium text-gray-700">{passed}/{total} ({percent:.1f}%)</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2.5">
                                <div class="bg-blue-600 h-2.5 rounded-full" style="width: {percent}%"></div>
                            </div>
                        </div>"""
    
    html += """
                    </div>
                </div>
            </div>
        </div>"""
    
    # Errors Section (if any)
    has_errors = False
    
    if ("structure_validation" in validation_results and validation_results["structure_validation"].get("errors")) or \
       ("content_validation" in validation_results and validation_results["content_validation"].get("errors")) or \
       ("structure_consistency" in validation_results and validation_results["structure_consistency"].get("errors")):
        has_errors = True
    
    if has_errors:
        html += """
        <!-- Errors Section -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold mb-4">Validation Errors</h2>
            
            <div class="grid grid-cols-1 gap-6">"""
        
        # Structure errors
        if "structure_validation" in validation_results and validation_results["structure_validation"].get("errors"):
            errors = validation_results["structure_validation"]["errors"]
            html += f"""
                <div>
                    <h3 class="text-lg font-semibold mb-2 text-red-700">Structure Errors ({len(errors)})</h3>
                    <div class="bg-red-50 p-4 rounded-lg error-card">
                        <ul class="list-disc pl-5 space-y-1">
                            {"".join(f'<li class="text-red-700">{escape_html(error)}</li>' for error in errors)}
                        </ul>
                    </div>
                </div>"""
        
        # Content errors
        if "content_validation" in validation_results and validation_results["content_validation"].get("errors"):
            errors = validation_results["content_validation"]["errors"]
            html += f"""
                <div>
                    <h3 class="text-lg font-semibold mb-2 text-red-700">Content Errors ({len(errors)})</h3>
                    <div class="bg-red-50 p-4 rounded-lg error-card">
                        <ul class="list-disc pl-5 space-y-1">
                            {"".join(f'<li class="text-red-700">{escape_html(error)}</li>' for error in errors)}
                        </ul>
                    </div>
                </div>"""
        
        # Consistency errors
        if "structure_consistency" in validation_results and validation_results["structure_consistency"].get("errors"):
            errors = validation_results["structure_consistency"]["errors"]
            html += f"""
                <div>
                    <h3 class="text-lg font-semibold mb-2 text-red-700">Consistency Errors ({len(errors)})</h3>
                    <div class="bg-red-50 p-4 rounded-lg error-card">
                        <ul class="list-disc pl-5 space-y-1">
                            {"".join(f'<li class="text-red-700">{escape_html(error)}</li>' for error in errors)}
                        </ul>
                    </div>
                </div>"""
        
        html += """
            </div>
        </div>"""
    
    # Content Comparison Section
    html += """
        <!-- Content Comparison -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold mb-4">Content Comparison</h2>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">"""
    
    # Extraction Content
    html += """
                <div>
                    <h3 class="text-lg font-semibold mb-2">Extraction Content</h3>
                    <div class="bg-gray-50 p-4 rounded-lg content-card">"""
    
    # Format extraction data for display
    if isinstance(extraction_data, list):
        # For list format (deepseek format)
        for i, section in enumerate(extraction_data[:5]):  # Limit to 5 sections for brevity
            title = section.get("title", f"Section {i+1}")
            content = section.get("content", "")
            html += f"""
                        <div class="mb-4">
                            <h4 class="font-semibold text-blue-700">{escape_html(title)}</h4>
                            <pre class="text-sm mt-2 bg-gray-100 p-2 rounded">{escape_html(content[:500])}{'...' if len(content) > 500 else ''}</pre>
                        </div>"""
            
            # Show tables if any
            if section.get("tables"):
                for j, table in enumerate(section["tables"][:2]):  # Limit to 2 tables
                    html += f"""
                        <div class="mb-4">
                            <h5 class="font-semibold text-blue-600">Table {j+1}</h5>
                            <div class="overflow-x-auto">
                                <table class="min-w-full bg-white border border-gray-300 rounded">
                                    <thead>
                                        <tr>"""
                    
                    # Table headers
                    if isinstance(table.get("content"), dict) and "headers" in table["content"]:
                        for header in table["content"]["headers"]:
                            html += f"""
                                            <th class="py-2 px-3 border-b border-gray-300 bg-gray-100 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">{escape_html(header)}</th>"""
                    
                    html += """
                                        </tr>
                                    </thead>
                                    <tbody>"""
                    
                    # Table rows
                    if isinstance(table.get("content"), dict) and "rows" in table["content"]:
                        for row in table["content"]["rows"][:3]:  # Limit to 3 rows
                            html += """
                                        <tr>"""
                            
                            for cell in row:
                                html += f"""
                                            <td class="py-2 px-3 border-b border-gray-300 text-sm">{escape_html(str(cell))}</td>"""
                            
                            html += """
                                        </tr>"""
                    
                    html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>"""
            
            # Show code blocks if any
            if section.get("code"):
                for j, code_block in enumerate(section["code"][:2]):  # Limit to 2 code blocks
                    language = code_block.get("language", "")
                    content = code_block.get("content", "")
                    html += f"""
                        <div class="mb-4">
                            <h5 class="font-semibold text-blue-600">Code Block ({language})</h5>
                            <pre class="text-sm mt-2 bg-gray-800 text-white p-2 rounded">{escape_html(content[:300])}{'...' if len(content) > 300 else ''}</pre>
                        </div>"""
        
        if len(extraction_data) > 5:
            html += f"""
                        <div class="text-gray-500 text-sm mt-4">
                            ... and {len(extraction_data) - 5} more sections not shown
                        </div>"""
    else:
        # For dictionary format
        if "sections" in extraction_data:
            sections = extraction_data["sections"]
            for i, section in enumerate(sections[:5]):  # Limit to 5 sections
                uuid = section.get("uuid", "")
                id = section.get("id", "")
                name = section.get("name", f"Section {i+1}")
                content = section.get("content", "")
                html += f"""
                        <div class="mb-4">
                            <h4 class="font-semibold text-blue-700">{escape_html(name)} ({escape_html(id)})</h4>
                            <div class="text-xs text-gray-500 mb-1">UUID: {uuid}</div>
                            <pre class="text-sm mt-2 bg-gray-100 p-2 rounded">{escape_html(content[:500])}{'...' if len(content) > 500 else ''}</pre>
                        </div>"""
            
            if len(sections) > 5:
                html += f"""
                        <div class="text-gray-500 text-sm mt-4">
                            ... and {len(sections) - 5} more sections not shown
                        </div>"""
        else:
            # Unknown format, just stringify
            html += f"""
                        <pre class="text-sm bg-gray-100 p-2 rounded">{escape_html(json.dumps(extraction_data, indent=2)[:1000])}...</pre>"""
    
    html += """
                    </div>
                </div>"""
    
    # Expected Format
    html += """
                <div>
                    <h3 class="text-lg font-semibold mb-2">Expected Format</h3>
                    <div class="bg-gray-50 p-4 rounded-lg content-card">"""
    
    # Show expected structure
    if "expected_structure" in expected_format:
        html += """
                        <h4 class="font-semibold text-green-700 mb-2">Expected Structure</h4>"""
        
        if "required_block_types" in expected_format["expected_structure"]:
            types = expected_format["expected_structure"]["required_block_types"]
            html += f"""
                        <div class="mb-4">
                            <h5 class="text-sm font-semibold">Required Block Types:</h5>
                            <ul class="list-disc pl-5 space-y-1">
                                {"".join(f'<li class="text-sm">{type}</li>' for type in types)}
                            </ul>
                        </div>"""
        
        if "hierarchy" in expected_format["expected_structure"]:
            hierarchy = expected_format["expected_structure"]["hierarchy"]
            html += """
                        <div class="mb-4">
                            <h5 class="text-sm font-semibold">Hierarchical Relationships:</h5>
                            <ul class="list-disc pl-5 space-y-1">"""
            
            for level in hierarchy:
                parent = level.get("parent_type", "")
                children = level.get("child_types", [])
                html += f"""
                                <li class="text-sm">{parent} &rarr; {", ".join(children)}</li>"""
            
            html += """
                            </ul>
                        </div>"""
        
        if "validation_threshold" in expected_format["expected_structure"]:
            threshold = expected_format["expected_structure"]["validation_threshold"]
            html += f"""
                        <div class="mb-4">
                            <h5 class="text-sm font-semibold">Validation Threshold: {threshold}%</h5>
                        </div>"""
    
    # Show expected content validation
    if "expected_content_validation" in expected_format:
        html += """
                        <h4 class="font-semibold text-green-700 mb-2 mt-4">Expected Content Validation</h4>"""
        
        content_validation = expected_format["expected_content_validation"]
        
        if "function_name" in content_validation:
            html += f"""
                        <div class="mb-2">
                            <h5 class="text-sm font-semibold">Function Name: {content_validation["function_name"]}</h5>
                        </div>"""
        
        if "function_purpose" in content_validation:
            purposes = content_validation["function_purpose"]
            html += f"""
                        <div class="mb-2">
                            <h5 class="text-sm font-semibold">Function Purpose:</h5>
                            <ul class="list-disc pl-5 space-y-1">
                                {"".join(f'<li class="text-sm">{purpose}</li>' for purpose in purposes)}
                            </ul>
                        </div>"""
        
        if "parameters" in content_validation:
            parameters = content_validation["parameters"]
            html += """
                        <div class="mb-2">
                            <h5 class="text-sm font-semibold">Parameters:</h5>
                            <ul class="list-disc pl-5 space-y-1">"""
            
            for param in parameters:
                name = param.get("name", "")
                type = param.get("type", "")
                descriptions = param.get("description", [])
                # Build the descriptions HTML separately to avoid nested f-strings
                desc_html = ""
                if descriptions:
                    desc_items = ""
                    for desc in descriptions:
                        desc_items += f'<li class="text-xs">{desc}</li>'
                    desc_html = f'<ul class="list-disc pl-5 space-y-1">{desc_items}</ul>'
                
                html += f"""
                                <li class="text-sm">
                                    <span class="font-semibold">{name}</span> ({type})
                                    {desc_html}
                                </li>"""
            
            html += """
                            </ul>
                        </div>"""
        
        if "return_type" in content_validation:
            html += f"""
                        <div class="mb-2">
                            <h5 class="text-sm font-semibold">Return Type: {content_validation["return_type"]}</h5>
                        </div>"""
        
        if "examples" in content_validation:
            examples = content_validation["examples"]
            html += """
                        <div class="mb-2">
                            <h5 class="text-sm font-semibold">Examples:</h5>
                            <ul class="list-disc pl-5 space-y-2">"""
            
            for example in examples:
                code = example.get("code", "")
                output = example.get("output", "")
                html += f"""
                                <li class="text-sm">
                                    <div><code class="bg-gray-100 px-1 py-0.5 rounded">{escape_html(code)}</code></div>
                                    {f'<div>Output: <code class="bg-gray-100 px-1 py-0.5 rounded">{escape_html(output)}</code></div>' if output else ''}
                                </li>"""
            
            html += """
                            </ul>
                        </div>"""
        
        if "validation_threshold" in content_validation:
            threshold = content_validation["validation_threshold"]
            html += f"""
                        <div class="mb-4">
                            <h5 class="text-sm font-semibold">Validation Threshold: {threshold}%</h5>
                        </div>"""
    
    # Close html
    html += """
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Add Lucide icons
        document.addEventListener('DOMContentLoaded', function() {
            const lucideIcons = document.querySelectorAll('[class^="lucide-"]');
            lucideIcons.forEach(icon => {
                const iconName = Array.from(icon.classList).find(cls => cls.startsWith('lucide-')).substring(7);
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', '24');
                svg.setAttribute('height', '24');
                svg.setAttribute('viewBox', '0 0 24 24');
                svg.setAttribute('fill', 'none');
                svg.setAttribute('stroke', 'currentColor');
                svg.setAttribute('stroke-width', '2');
                svg.setAttribute('stroke-linecap', 'round');
                svg.setAttribute('stroke-linejoin', 'round');
                
                if (iconName === 'check-circle') {
                    svg.innerHTML = '<circle cx="12" cy="12" r="10"></circle><path d="m9 12 2 2 4-4"></path>';
                } else if (iconName === 'x-circle') {
                    svg.innerHTML = '<circle cx="12" cy="12" r="10"></circle><path d="m15 9-6 6"></path><path d="m9 9 6 6"></path>';
                }
                
                icon.appendChild(svg);
            });
        });
    </script>
</body>
</html>
"""
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"HTML report generated at {output_path}")


def main():
    """Main function for the HTML report generator."""
    parser = argparse.ArgumentParser(description="Generate HTML report for validation results")
    parser.add_argument("--validation", type=str, required=True,
                        help="Path to the validation results JSON file")
    parser.add_argument("--extraction", type=str, required=True,
                        help="Path to the extraction result JSON file")
    parser.add_argument("--expected", type=str, required=True,
                        help="Path to the expected format JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the HTML report")
    parser.add_argument("--name", type=str, default="Validation Results",
                        help="Name for the test")
    args = parser.parse_args()
    
    # Load validation results
    validation_path = Path(args.validation)
    validation_results = load_json_file(validation_path)
    
    if not validation_results:
        logger.error(f"Failed to load validation results from {validation_path}")
        return 1
    
    # Generate HTML report
    generate_html_report(
        validation_results,
        Path(args.extraction),
        Path(args.expected),
        Path(args.output),
        args.name
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())