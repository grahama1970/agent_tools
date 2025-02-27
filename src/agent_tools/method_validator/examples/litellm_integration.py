"""Example of using Method Validator with LiteLLM package analysis."""

from sparta.tools.method_validator import MethodAnalyzer
import litellm
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def progress(msg):
    """Write progress message with immediate flush."""
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


def categorize_method(method_info: Dict) -> List[str]:
    """Categorize a method based on its name and signature."""
    categories = set()
    name = method_info["name"].lower()
    signature = method_info["signature"].lower()

    # Operation type
    if "completion" in name:
        categories.add("completion")
    if "embedding" in name:
        categories.add("embedding")
    if "stream" in name or "streaming" in name:
        categories.add("streaming")

    # Execution mode
    if name.startswith("a") and ("async" in name or "await" in signature):
        categories.add("async")
    else:
        categories.add("sync")

    # Provider specific
    for provider in ["openai", "anthropic", "vertex", "together", "huggingface"]:
        if provider in name.lower():
            categories.add(f"provider:{provider}")

    # Special features
    if "batch" in name:
        categories.add("batch")
    if "chat" in name:
        categories.add("chat")
    if "text" in name:
        categories.add("text")

    return sorted(list(categories))


def find_related_methods(methods: List[Dict]) -> Dict[str, List[str]]:
    """Find relationships between methods (e.g., sync/async variants)."""
    relationships = {}

    # Group methods by base name (removing a/async prefix)
    base_methods: Dict[str, List[Dict]] = {}
    for method in methods:
        name = method["name"]
        base_name = name
        if name.startswith("a") and len(name) > 1:
            base_name = name[1:]  # Remove 'a' prefix
        base_methods.setdefault(base_name, []).append(method)

    # Establish relationships
    for base_name, related in base_methods.items():
        for method in related:
            name = method["name"]
            relationships[name] = []
            # Find sync/async pairs
            for other in related:
                other_name = other["name"]
                if name != other_name:
                    relationships[name].append(
                        {
                            "name": other_name,
                            "type": (
                                "async_variant"
                                if other_name.startswith("a")
                                else "sync_variant"
                            ),
                        }
                    )

            # Find overloaded variants
            for other in methods:
                if (
                    other["name"] != name
                    and name in other["name"]
                    and other["name"] not in [r["name"] for r in relationships[name]]
                ):
                    relationships[name].append(
                        {"name": other["name"], "type": "specialized_variant"}
                    )

    return relationships


def analyze_litellm_completion():
    """Analyze LiteLLM completion methods and their error handling."""
    analyzer = MethodAnalyzer()

    progress("\n=== Starting LiteLLM Analysis ===")

    # Quick scan for completion-related methods
    progress("\n[1/4] Scanning for completion-related methods...")
    methods = analyzer.quick_scan("litellm")
    completion_methods = [
        name
        for name, summary, _ in methods
        if "completion" in name.lower() or "completion" in summary.lower()
    ]
    progress(f"Found {len(completion_methods)} completion-related methods")

    # Deep analyze each completion method
    progress("\n[2/4] Analyzing methods in detail...")
    results = []
    total = len(completion_methods)
    for i, method_name in enumerate(completion_methods, 1):
        progress(f"Analyzing [{i}/{total}]: {method_name}")
        info = analyzer.deep_analyze("litellm", method_name)
        if info:
            results.append(info)

    # Enhance results with categories and relationships
    progress("\n[3/4] Adding categories and relationships...")
    relationships = find_related_methods(results)
    for method_info in results:
        method_info["categories"] = categorize_method(method_info)
        method_info["related_methods"] = relationships.get(method_info["name"], [])

    # Save results
    progress("\n[4/4] Saving analysis results...")
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "litellm_completion_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    progress("\n=== Analysis Complete ===")
    progress(f"Results saved to: {output_file}")
    return results


def main():
    """Run the example."""
    try:
        results = analyze_litellm_completion()

        # Display a summary
        progress("\n=== Analysis Summary ===")

        # Group by category
        by_category: Dict[str, List[str]] = {}
        for method_info in results:
            for category in method_info["categories"]:
                by_category.setdefault(category, []).append(method_info["name"])

        # Show category summary
        progress("\nMethods by Category:")
        for category, methods in sorted(by_category.items()):
            progress(f"\n• {category.upper()}")
            for method in sorted(methods):
                related = [r["name"] for r in method_info["related_methods"]]
                related_str = f" (Related: {', '.join(related)})" if related else ""
                progress(f"  - {method}{related_str}")

    except Exception as e:
        progress(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
