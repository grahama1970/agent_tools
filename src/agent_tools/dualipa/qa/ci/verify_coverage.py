#!/usr/bin/env python3
"""Test coverage verification for CI.

This script verifies that the test suite meets the required coverage threshold.
It reads coverage data from an XML report and ensures key components have
sufficient test coverage to maintain code quality.

Official documentation:
- argparse: https://docs.python.org/3/library/argparse.html
- json: https://docs.python.org/3/library/json.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- sys: https://docs.python.org/3/library/sys.html
- typing: https://docs.python.org/3/library/typing.html
- xml.etree.ElementTree: https://docs.python.org/3/library/xml.etree.elementtree.html

Expected input/output:
- parse_coverage_xml: Takes coverage XML file path, returns coverage data dictionary
  * Input: Path to coverage XML file
  * Output: Dictionary with coverage metrics by file/class/module
  * Verification: XML file exists and contains valid coverage data

- verify_coverage: Takes coverage data, returns success_flag and error_message
  * Input: Coverage data dictionary and threshold values
  * Output: Boolean success status and error message if applicable
  * Verification: Coverage meets minimum thresholds for all components
"""

import os
import sys
import json
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define coverage thresholds (as percentages)
COVERAGE_THRESHOLDS = {
    "overall": 80.0,           # Overall module coverage
    "processor": 90.0,         # Main processor module
    "utils": 80.0,             # Utility functions
    "models": 90.0,            # Data models
    "llm": 75.0,               # LLM integration (lower due to API dependencies)
    "cli": 85.0,               # Command-line interface
    "monitoring": 80.0,        # Monitoring components
    "performance": 75.0,       # Performance components (some are hard to test)
    "cache": 85.0              # Caching components
}

# Path to default coverage report
DEFAULT_COVERAGE_FILE = Path("coverage.xml")


def parse_coverage_xml(coverage_file: Union[str, Path] = DEFAULT_COVERAGE_FILE) -> Dict[str, Any]:
    """Parse coverage XML data.
    
    Args:
        coverage_file: Path to coverage XML file
        
    Returns:
        Dictionary with parsed coverage metrics
    """
    if not Path(coverage_file).exists():
        logger.error(f"Coverage file not found: {coverage_file}")
        return {}
    
    try:
        # Parse XML file
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Extract overall coverage
        coverage_data = {
            "overall": {
                "line_rate": float(root.attrib.get("line-rate", 0)) * 100,
                "branch_rate": float(root.attrib.get("branch-rate", 0)) * 100,
                "lines_covered": int(root.attrib.get("lines-covered", 0)),
                "lines_valid": int(root.attrib.get("lines-valid", 0)),
                "timestamp": root.attrib.get("timestamp", "")
            },
            "files": {}
        }
        
        # Extract per-file coverage
        for package in root.findall(".//package"):
            package_name = package.attrib.get("name", "")
            
            # Extract coverage for each file in package
            for class_elem in package.findall(".//class"):
                filename = class_elem.attrib.get("filename", "")
                
                # Skip files not in the QA module
                if "dualipa/qa" not in filename:
                    continue
                
                # Get component name
                path_parts = filename.split("/")
                if "dualipa/qa" in filename:
                    idx = path_parts.index("qa")
                    if idx + 1 < len(path_parts):
                        component = path_parts[idx + 1]
                    else:
                        component = "root"
                else:
                    component = "unknown"
                
                # Extract coverage metrics
                line_rate = float(class_elem.attrib.get("line-rate", 0)) * 100
                branch_rate = float(class_elem.attrib.get("branch-rate", 0)) * 100
                
                # Extract line counts
                lines = class_elem.findall(".//line")
                lines_total = len(lines)
                lines_covered = sum(1 for line in lines if line.attrib.get("hits", "0") != "0")
                
                # Add to coverage data
                coverage_data["files"][filename] = {
                    "line_rate": line_rate,
                    "branch_rate": branch_rate,
                    "lines_total": lines_total,
                    "lines_covered": lines_covered,
                    "component": component
                }
        
        # Calculate component-level summaries
        coverage_data["components"] = {}
        
        # Group files by component
        component_files = {}
        for filepath, data in coverage_data["files"].items():
            component = data["component"]
            if component not in component_files:
                component_files[component] = []
            component_files[component].append(filepath)
        
        # Calculate component summaries
        for component, files in component_files.items():
            if not files:
                continue
                
            # Calculate aggregate coverage
            total_lines = sum(coverage_data["files"][f]["lines_total"] for f in files)
            covered_lines = sum(coverage_data["files"][f]["lines_covered"] for f in files)
            
            # Calculate coverage rate
            line_rate = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            # Add component summary
            coverage_data["components"][component] = {
                "line_rate": line_rate,
                "files": len(files),
                "lines_total": total_lines,
                "lines_covered": covered_lines
            }
        
        return coverage_data
    
    except Exception as e:
        logger.error(f"Error parsing coverage XML: {e}")
        return {}


def verify_coverage(
    coverage_data: Dict[str, Any],
    thresholds: Dict[str, float] = COVERAGE_THRESHOLDS
) -> Tuple[bool, Optional[str]]:
    """Verify coverage meets thresholds.
    
    Args:
        coverage_data: Coverage data dictionary
        thresholds: Coverage thresholds by component
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    if not coverage_data:
        return False, "No coverage data available"
    
    failures = []
    
    # Check overall coverage
    overall_rate = coverage_data.get("overall", {}).get("line_rate", 0)
    overall_threshold = thresholds.get("overall", 80.0)
    
    if overall_rate < overall_threshold:
        failures.append(
            f"Overall coverage ({overall_rate:.1f}%) below threshold ({overall_threshold:.1f}%)"
        )
    
    # Check component coverage
    components = coverage_data.get("components", {})
    for component, threshold in thresholds.items():
        if component == "overall":
            continue
            
        # If component exists, check coverage
        if component in components:
            line_rate = components[component].get("line_rate", 0)
            
            if line_rate < threshold:
                failures.append(
                    f"Component '{component}' coverage ({line_rate:.1f}%) below threshold ({threshold:.1f}%)"
                )
    
    # Return success if no failures, otherwise return error message
    if not failures:
        return True, None
    else:
        return False, "\n".join(failures)


def main():
    """Main entry point for coverage verification."""
    parser = argparse.ArgumentParser(
        description="Verify test coverage meets thresholds"
    )
    parser.add_argument(
        "--coverage-file",
        type=str,
        default=str(DEFAULT_COVERAGE_FILE),
        help=f"Path to coverage XML file (default: {DEFAULT_COVERAGE_FILE})"
    )
    parser.add_argument(
        "--thresholds-file",
        type=str,
        help="Path to JSON file with custom coverage thresholds"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a detailed coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom thresholds if specified
    thresholds = COVERAGE_THRESHOLDS
    if args.thresholds_file:
        try:
            with open(args.thresholds_file, 'r') as f:
                custom_thresholds = json.load(f)
                thresholds.update(custom_thresholds)
        except Exception as e:
            logger.error(f"Error loading custom thresholds: {e}")
    
    # Parse coverage data
    logger.info(f"Parsing coverage data from {args.coverage_file}")
    coverage_data = parse_coverage_xml(args.coverage_file)
    
    if not coverage_data:
        logger.error("No coverage data available")
        return 1
    
    # Generate report if requested
    if args.generate_report:
        report_file = "coverage_report.json"
        with open(report_file, 'w') as f:
            json.dump(coverage_data, f, indent=2)
        logger.info(f"Generated coverage report: {report_file}")
    
    # Log overall coverage
    overall_rate = coverage_data.get("overall", {}).get("line_rate", 0)
    logger.info(f"Overall coverage: {overall_rate:.1f}%")
    
    # Log component coverage
    for component, data in coverage_data.get("components", {}).items():
        line_rate = data.get("line_rate", 0)
        threshold = thresholds.get(component, thresholds.get("overall", 80.0))
        status = "✅" if line_rate >= threshold else "❌"
        logger.info(f"{status} {component}: {line_rate:.1f}% (threshold: {threshold:.1f}%)")
    
    # Verify coverage
    success, error_message = verify_coverage(coverage_data, thresholds)
    
    if success:
        logger.info("✅ Coverage verification passed: All components meet thresholds")
        return 0
    else:
        logger.error(f"❌ Coverage verification failed:\n{error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())