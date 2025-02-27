#!/usr/bin/env python3
"""
Test module for web table extraction functionality using Playwright and pandas.
This module tests the ability to extract tables from various web pages using
the method validator tool.
"""

import asyncio
import pandas as pd
from typing import List, Optional, Dict, Any
from playwright.async_api import async_playwright, Page
from loguru import logger
import pytest
from sparta.tools.method_validator import MethodAnalyzer


async def extract_tables_from_webpage(url: str) -> Optional[List[pd.DataFrame]]:
    """
    Extract tables from a webpage using Playwright and pandas.

    Args:
        url (str): The URL of the webpage to extract tables from.

    Returns:
        Optional[List[pd.DataFrame]]: List of pandas DataFrames containing the tables,
                                    or None if no tables were found.
    """
    try:
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to the URL with a timeout
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for any tables to load
            await page.wait_for_selector("table", timeout=5000)

            # Extract all tables from the page
            tables = []
            table_elements = await page.query_selector_all("table")

            for table in table_elements:
                # Get table HTML
                table_html = await table.inner_html()

                # Convert HTML table to pandas DataFrame
                dfs = pd.read_html(f"<table>{table_html}</table>")
                if dfs:
                    tables.extend(dfs)

            await browser.close()
            return tables if tables else None

    except Exception as e:
        logger.error(f"Error extracting tables from {url}: {e}")
        return None


async def analyze_table_data(tables: List[pd.DataFrame]) -> Dict[str, Any]:
    """
    Analyze the extracted table data.

    Args:
        tables (List[pd.DataFrame]): List of pandas DataFrames containing table data.

    Returns:
        Dict[str, Any]: Analysis results including number of tables, row counts, etc.
    """
    analysis = {
        "num_tables": len(tables),
        "total_rows": sum(len(df) for df in tables),
        "tables_info": [],
    }

    for i, df in enumerate(tables):
        table_info = {
            "table_index": i,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "has_missing_values": df.isnull().any().any(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024,  # KB
        }
        analysis["tables_info"].append(table_info)

    return analysis


async def main():
    """
    Main function to test table extraction from various websites.
    """
    # Example URLs to try
    urls = [
        "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue",
        "https://en.wikipedia.org/wiki/List_of_programming_languages",
        "https://socialblade.com/youtube/top/category/news/mostviewed",  # Keep as fallback
    ]

    # Try each URL until we get results
    for url in urls:
        logger.info(f"Attempting to extract tables from {url}")
        tables = await extract_tables_from_webpage(url)
        if tables:
            break

    if not tables:
        logger.error("Failed to extract tables from any of the URLs")
        return

    # Analyze the extracted tables
    analysis = await analyze_table_data(tables)
    logger.info(f"Table analysis results: {analysis}")

    # Example: Save the first table to CSV
    if tables:
        tables[0].to_csv("extracted_table.csv", index=False)
        logger.info("First table saved to extracted_table.csv")


@pytest.mark.asyncio
async def test_table_extraction():
    """
    Test the table extraction functionality.
    """
    # Test with Wikipedia page that has known tables
    url = "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue"
    tables = await extract_tables_from_webpage(url)

    assert tables is not None, "Failed to extract tables"
    assert len(tables) > 0, "No tables found in the webpage"

    # Test table analysis
    analysis = await analyze_table_data(tables)
    assert analysis["num_tables"] > 0, "Analysis shows no tables"
    assert analysis["total_rows"] > 0, "Analysis shows no rows"

    # Test first table structure
    first_table = tables[0]
    assert isinstance(first_table, pd.DataFrame), "First table is not a DataFrame"
    assert len(first_table) > 0, "First table has no rows"
    assert len(first_table.columns) > 0, "First table has no columns"


if __name__ == "__main__":
    asyncio.run(main())
