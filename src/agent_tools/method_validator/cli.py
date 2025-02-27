#!/usr/bin/env python3
"""
Method Validator - An AI Agent's Tool for API Discovery and Validation

This tool helps AI agents quickly discover and validate existing methods in Python packages
before implementing new solutions. It should ONLY be used for analyzing non-standard packages
that are directly relevant to the function being written.

DO NOT USE FOR:
- Standard library packages (os, sys, json, etc.)
- Well-known utility packages (requests, urllib, etc.)
- Packages that aren't directly related to the function being written

Usage workflow for AI agents:
1. When writing a function that might already exist in a relevant package:
   - Use --list-all to check for existing implementations
2. When using a package-specific method:
   - Use --method to verify its exact signature and behavior
3. When handling errors:
   - Use --exceptions-only to see package-specific exceptions to handle
"""

import argparse
import json
import sys
import sqlite3
import math
from pathlib import Path
from loguru import logger
from typing import Optional, List, Dict, Any

from .analyzer import MethodAnalyzer
from .cache import analysis_cache, timing


# Standard library packages that should be skipped
STANDARD_PACKAGES = {
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "binascii",
    "builtins",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "functools",
    "glob",
    "gzip",
    "hashlib",
    "hmac",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "pprint",
    "queue",
    "re",
    "random",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "ssl",
    "statistics",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "timeit",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
    "zlib",
}

# Common utility packages that typically don't need analysis
COMMON_UTILITY_PACKAGES = {
    "requests",
    "urllib3",
    "six",
    "setuptools",
    "pip",
    "wheel",
    "pkg_resources",
    "pytest",
    "nose",
    "mock",
    "coverage",
    "tox",
    "flake8",
    "pylint",
    "mypy",
    "black",
    "isort",
    "yapf",
    "autopep8",
}


def should_analyze_package(package_name: str) -> bool:
    """
    Determine if a package should be analyzed based on its name.

    Returns False for:
    - Standard library packages
    - Common utility packages
    - Package names that suggest they're not directly relevant
    """
    # Skip standard library and common utility packages
    if (
        package_name in STANDARD_PACKAGES
        or package_name in COMMON_UTILITY_PACKAGES
        or any(package_name.startswith(f"{pkg}.") for pkg in STANDARD_PACKAGES)
    ):
        return False

    return True


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes = math.floor(size_bytes / 1024)  # Convert to int
    return f"{size_bytes:.1f}GB"


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    stats = {
        "location": str(analysis_cache.db_path),
        "entries": 0,
        "total_size": 0,
        "age_limit_days": analysis_cache.max_age_days,
        "size_limit": format_size(analysis_cache.max_size_mb * 1024 * 1024),
    }

    try:
        with sqlite3.connect(analysis_cache.db_path) as conn:
            # Get entry count and total size
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(size), 0)
                FROM method_cache
            """
            ).fetchone()
            stats["entries"] = row[0]
            stats["total_size"] = format_size(row[1])

            # Get oldest and newest entry
            times = conn.execute(
                """
                SELECT MIN(timestamp), MAX(timestamp)
                FROM method_cache
            """
            ).fetchone()
            if times[0] and times[1]:
                import time

                now = time.time()
                stats["oldest_entry_days"] = (now - times[0]) / (24 * 3600)
                stats["newest_entry_days"] = (now - times[1]) / (24 * 3600)

            # Get package statistics
            packages = conn.execute(
                """
                SELECT package_name, COUNT(*), SUM(size)
                FROM method_cache
                GROUP BY package_name
            """
            ).fetchall()
            stats["packages"] = {
                pkg: {"entries": count, "size": format_size(size)}
                for pkg, count, size in packages
            }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")

    return stats


def migrate_old_cache():
    """Migrate cache from old location if it exists."""
    old_cache_dir = Path.home() / ".cache" / "sparta" / "method_validator"
    old_cache_path = old_cache_dir / "analysis_cache.db"

    if not old_cache_path.exists():
        return

    try:
        import sqlite3
        import shutil

        # Copy old database to new location
        new_cache_path = analysis_cache.db_path
        if not new_cache_path.parent.exists():
            new_cache_path.parent.mkdir(parents=True)

        # Connect to both databases
        with (
            sqlite3.connect(old_cache_path) as old_conn,
            sqlite3.connect(new_cache_path) as new_conn,
        ):

            # Initialize new database
            analysis_cache._init_db()

            # Copy data
            old_data = old_conn.execute("SELECT * FROM method_cache").fetchall()
            if old_data:
                new_conn.executemany(
                    "INSERT OR REPLACE INTO method_cache VALUES (?, ?, ?, ?, ?, ?)",
                    old_data,
                )
                new_conn.commit()

        # Backup old cache and remove it
        backup_path = old_cache_path.with_suffix(".db.bak")
        shutil.move(old_cache_path, backup_path)
        logger.info(f"Old cache backed up to: {backup_path}")

    except Exception as e:
        logger.error(f"Cache migration failed: {e}")


def main():
    """Main entry point for the method validator."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "package",
        help=(
            "Non-standard package to analyze. DO NOT use for standard library "
            "or common utility packages."
        ),
    )
    parser.add_argument("--method", help="Method name for detailed analysis")
    parser.add_argument(
        "--list-all", action="store_true", help="Quick scan of all available methods"
    )
    parser.add_argument(
        "--by-category", action="store_true", help="Group methods by category"
    )
    parser.add_argument(
        "--show-exceptions",
        action="store_true",
        help="Show detailed exception information",
    )
    parser.add_argument(
        "--exceptions-only",
        action="store_true",
        help="Show only exception information for the package",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format for machine consumption",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show execution timing statistics",
    )

    # New cache management arguments
    cache_group = parser.add_argument_group("Cache Management")
    cache_group.add_argument(
        "--cache-stats", action="store_true", help="Show cache statistics"
    )
    cache_group.add_argument(
        "--cache-clear", action="store_true", help="Clear the entire cache"
    )
    cache_group.add_argument(
        "--cache-migrate", action="store_true", help="Migrate cache from old location"
    )
    cache_group.add_argument(
        "--cache-max-age", type=int, help="Set maximum cache entry age in days"
    )
    cache_group.add_argument(
        "--cache-max-size", type=int, help="Set maximum cache size in MB"
    )
    cache_group.add_argument(
        "--compression-level",
        type=int,
        choices=range(10),
        help="Set compression level (0-9, where 0 is no compression and 9 is maximum)",
    )
    cache_group.add_argument(
        "--recompress",
        action="store_true",
        help="Recompress all cached data with current/new compression level",
    )

    args = parser.parse_args()

    # Enable/disable timing based on argument
    timing.enabled = args.timing

    # Handle cache management commands first
    if args.cache_stats:
        stats = get_cache_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nCache Statistics:")
            print(f"Location: {stats['location']}")
            print(f"Total Entries: {stats['entries']}")
            print(f"Total Size: {stats['total_size']}")
            print(f"Age Limit: {stats['age_limit_days']} days")
            print(f"Size Limit: {stats['size_limit']}")
            print(f"Compression Level: {analysis_cache.compression_level}")
            if "oldest_entry_days" in stats:
                print(f"\nEntry Age Range:")
                print(f"  Oldest: {stats['oldest_entry_days']:.1f} days")
                print(f"  Newest: {stats['newest_entry_days']:.1f} days")
            if stats.get("packages"):
                print("\nPackage Statistics:")
                for pkg, pkg_stats in stats["packages"].items():
                    print(f"  {pkg}:")
                    print(f"    Entries: {pkg_stats['entries']}")
                    print(f"    Size: {pkg_stats['size']}")
        return

    if args.cache_clear:
        analysis_cache.clear()
        print("Cache cleared successfully")
        return

    if args.cache_migrate:
        migrate_old_cache()
        print("Cache migration completed")
        return

    if args.cache_max_age:
        analysis_cache.max_age_days = args.cache_max_age
        print(f"Cache maximum age set to {args.cache_max_age} days")

    if args.cache_max_size:
        analysis_cache.max_size_mb = args.cache_max_size
        print(f"Cache maximum size set to {args.cache_max_size}MB")

    if args.compression_level is not None:
        analysis_cache.compression_level = args.compression_level
        print(f"Cache compression level set to {args.compression_level}")
        if args.recompress:
            print("Recompressing cache with new level...")
            stats = analysis_cache.recompress_all()
            print("\nRecompression Statistics:")
            print(f"Total entries processed: {stats['total']}")
            print(f"Entries compressed: {stats['compressed']}")
            print(f"Size before: {format_size(stats['size_before'])}")
            print(f"Size after: {format_size(stats['size_after'])}")
            print(
                f"Space saved: {format_size(stats['size_before'] - stats['size_after'])}"
            )
            return
    elif args.recompress:
        print("Recompressing cache...")
        stats = analysis_cache.recompress_all()
        print("\nRecompression Statistics:")
        print(f"Total entries processed: {stats['total']}")
        print(f"Entries compressed: {stats['compressed']}")
        print(f"Size before: {format_size(stats['size_before'])}")
        print(f"Size after: {format_size(stats['size_after'])}")
        print(f"Space saved: {format_size(stats['size_before'] - stats['size_after'])}")
        return

    # Validate package name
    if not args.package and not any(
        [args.cache_stats, args.cache_clear, args.cache_migrate]
    ):
        parser.error("package name is required for analysis operations")

    if not should_analyze_package(args.package):
        if args.json:
            print(
                json.dumps(
                    {"error": "Package should not be analyzed", "package": args.package}
                )
            )
            sys.exit(1)
        logger.warning(
            f"Package '{args.package}' is a standard/utility package and should not be analyzed."
        )
        logger.info(
            "This tool is intended for analyzing non-standard packages relevant to your current task."
        )
        sys.exit(1)

    try:
        with timing.measure("total_execution", "Total execution time"):
            analyzer = MethodAnalyzer()
            result = {}

            if args.exceptions_only:
                with timing.measure(
                    "exception_analysis", "Analyzing package exceptions"
                ):
                    # Use deep_analyze to get full method info including exceptions
                    methods = analyzer.quick_scan(args.package)
                    exceptions = []
                    for method_name, _, _ in methods:
                        info = analyzer.deep_analyze(args.package, method_name)
                        if info and info.get("exceptions"):
                            exceptions.extend(info["exceptions"])

                    result = {"package": args.package, "exceptions": exceptions}

            elif args.list_all:
                with timing.measure("quick_scan", "Performing quick scan"):
                    methods = analyzer.quick_scan(args.package)
                    if args.json:
                        result["methods"] = []
                        if args.by_category:
                            by_category = {}
                            for name, summary, categories in methods:
                                for category in categories:
                                    by_category.setdefault(category, []).append(
                                        {"name": name, "summary": summary}
                                    )
                            result["methods_by_category"] = by_category
                        else:
                            result["methods"] = [
                                {"name": name, "summary": summary}
                                for name, summary, _ in methods
                            ]
                    else:
                        if args.by_category:
                            by_category = {}
                            for name, summary, categories in methods:
                                for category in categories:
                                    by_category.setdefault(category, []).append(
                                        (name, summary)
                                    )

                            for category, methods in by_category.items():
                                logger.info(f"\n[{category.upper()}]")
                                for name, summary in methods:
                                    logger.info(f"\n{name}:")
                                    if summary:
                                        logger.info(f"  Summary: {summary}")
                        else:
                            for name, summary, _ in methods:
                                logger.info(f"\n{name}:")
                                if summary:
                                    logger.info(f"  Summary: {summary}")

            elif args.method:
                with timing.measure("deep_analysis", f"Analyzing method {args.method}"):
                    method_info = analyzer.deep_analyze(args.package, args.method)
                    if method_info:
                        if args.json:
                            result["method_info"] = method_info
                        else:
                            logger.info(f"\nDetailed analysis of '{args.method}':")
                            logger.info(f"Signature: {method_info['signature']}")
                            logger.info(
                                f"Categories: {', '.join(method_info['categories'])}"
                            )

                            if method_info["doc"]:
                                logger.info(f"\nDocumentation:\n{method_info['doc']}")

                            logger.info("\nParameters:")
                            for name, details in method_info["parameters"].items():
                                required = (
                                    "required" if details["required"] else "optional"
                                )
                                logger.info(f"  {name} ({required}):")
                                if details["type"]:
                                    logger.info(f"    Type: {details['type']}")
                                if details["description"]:
                                    logger.info(
                                        f"    Description: {details['description']}"
                                    )
                                if details["default"]:
                                    logger.info(f"    Default: {details['default']}")

                            if args.show_exceptions and method_info["exceptions"]:
                                logger.info("\nExceptions:")
                                for exc in method_info["exceptions"]:
                                    logger.info(f"  {exc['type']}:")
                                    logger.info(
                                        f"    Description: {exc['description']}"
                                    )
                                    if exc["hierarchy"]:
                                        logger.info(
                                            f"    Hierarchy: {' -> '.join(exc['hierarchy'])}"
                                        )

                            if method_info["return_info"]["type"]:
                                logger.info(
                                    f"\nReturns: {method_info['return_info']['type']}"
                                )
                                if method_info["return_info"]["description"]:
                                    logger.info(
                                        f"  {method_info['return_info']['description']}"
                                    )

                            if method_info["examples"]:
                                logger.info("\nExamples:")
                                for example in method_info["examples"]:
                                    logger.info(f"\n{example}")
                    else:
                        similar = [
                            name
                            for name, _, _ in analyzer.quick_scan(args.package)
                            if args.method.lower() in name.lower()
                        ]
                        if args.json:
                            result["error"] = f"Method '{args.method}' not found"
                            if similar:
                                result["similar_methods"] = similar
                        else:
                            logger.warning(f"\nMethod '{args.method}' not found.")
                            if similar:
                                logger.info("Similar methods found:")
                                for method in similar:
                                    logger.info(f"  - {method}")

            else:
                if args.json:
                    result["error"] = "No action specified"
                else:
                    parser.print_help()

            if args.timing:
                timing_stats = timing.get_summary()
                if args.json:
                    result["timing_stats"] = timing_stats
                else:
                    logger.debug("\nTiming Statistics:")
                    for op, stats in timing_stats.items():
                        logger.debug(f"\n{op}:")
                        logger.debug(f"  Total time: {stats['total_time']:.3f}s")
                        logger.debug(f"  Calls: {stats['calls']}")
                        logger.debug(f"  Average time: {stats['average_time']:.3f}s")

            if args.json:
                print(json.dumps(result))

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
