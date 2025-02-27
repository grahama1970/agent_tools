import asyncio
from playwright.async_api import async_playwright
from loguru import logger
import json
from pathlib import Path
from typing import List, Dict, Any


async def extract_tables_from_webpage(url: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a webpage using Playwright.

    Args:
        url: The URL of the webpage to extract tables from

    Returns:
        List of dictionaries containing table data and metadata
    """
    logger.info(f"Extracting tables from {url}")

    async with async_playwright() as p:
        # Launch browser with stealth mode
        browser = await p.chromium.launch(
            headless=True,
        )

        # Create context with realistic user agent
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )

        # Create page with longer timeout
        page = await context.new_page()
        page.set_default_timeout(60000)  # 60 second timeout

        try:
            # Navigate to page and wait for content to load
            response = await page.goto(url, wait_until="networkidle")

            # Check if response exists and is successful
            if response is None:
                logger.error("Failed to get response from page")
                return []
            elif not response.ok:
                logger.error(
                    f"Failed to load page: {response.status} {response.status_text}"
                )
                return []

            # Wait for any dynamic content to load
            await page.wait_for_load_state("networkidle")

            # Check if page was blocked
            if await page.title() == "Access Denied":
                logger.error("Access was denied by the website")
                return []

            # Try to find tables
            tables = await page.query_selector_all("table")
            if not tables:
                logger.warning("No tables found on page")
                return []

            logger.info(f"Found {len(tables)} potential tables")

            # Extract table data using JavaScript
            tables_data = await page.evaluate(
                """() => {
                const tables = document.querySelectorAll('table');
                return Array.from(tables).map(table => {
                    // Get table headers
                    const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                    
                    // If no headers found, try first row
                    if (headers.length === 0) {
                        const firstRow = Array.from(table.querySelector('tr').querySelectorAll('td')).map(td => td.textContent.trim());
                        if (firstRow.length > 0) {
                            headers.push(...firstRow);
                        }
                    }
                    
                    // Get table rows
                    const rows = Array.from(table.querySelectorAll('tr')).slice(headers.length ? 1 : 0);
                    const data = rows.map(row => {
                        const cells = Array.from(row.querySelectorAll('td')).map(td => {
                            // Get both text content and any links
                            const links = Array.from(td.querySelectorAll('a')).map(a => ({
                                text: a.textContent.trim(),
                                href: a.href,
                                title: a.title || null
                            }));
                            return {
                                text: td.textContent.trim(),
                                links: links.length > 0 ? links : null
                            };
                        });
                        return cells;
                    });
                    
                    // Get table caption or nearest heading
                    let title = '';
                    const caption = table.querySelector('caption');
                    if (caption) {
                        title = caption.textContent.trim();
                    } else {
                        // Look for nearest heading
                        let prevElem = table.previousElementSibling;
                        while (prevElem && !title) {
                            if (prevElem.tagName.match(/^H[1-6]$/)) {
                                title = prevElem.textContent.trim();
                                break;
                            }
                            prevElem = prevElem.previousElementSibling;
                        }
                    }
                    
                    return {
                        headers,
                        data,
                        title
                    };
                });
            }"""
            )

            # Process extracted tables
            processed_tables = []
            for idx, table in enumerate(tables_data):
                # Calculate table statistics
                row_count = len(table["data"])
                col_count = len(table["headers"])

                # Skip empty tables
                if row_count == 0 or col_count == 0:
                    logger.warning(f"Skipping empty table {idx + 1}")
                    continue

                # Process rows into structured format
                processed_rows = []
                for row in table["data"]:
                    row_dict = {}
                    for col_idx, cell in enumerate(row):
                        header = (
                            table["headers"][col_idx]
                            if col_idx < len(table["headers"])
                            else f"Column {col_idx + 1}"
                        )
                        row_dict[header] = {
                            "value": cell["text"],
                            "links": cell["links"],
                        }
                    processed_rows.append(row_dict)

                processed_tables.append(
                    {
                        "table_index": idx,
                        "title": table["title"] or f"Table {idx + 1}",
                        "headers": table["headers"],
                        "data": processed_rows,
                        "stats": {"row_count": row_count, "column_count": col_count},
                    }
                )

            logger.info(f"Successfully processed {len(processed_tables)} tables")
            return processed_tables

        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            # Take screenshot for debugging
            try:
                await page.screenshot(path="error_screenshot.png")
                logger.info("Saved error screenshot to error_screenshot.png")
            except:
                pass
            return []

        finally:
            await context.close()
            await browser.close()


async def main():
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

    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "extracted_tables.json"
    with open(output_file, "w") as f:
        json.dump(tables, f, indent=2)

    logger.info(f"Saved {len(tables)} tables to {output_file}")

    # Print summary of each table
    for table in tables:
        logger.info(f"\nTable: {table['title']}")
        logger.info(f"Headers: {table['headers']}")
        logger.info(f"Rows: {table['stats']['row_count']}")
        logger.info(f"Columns: {table['stats']['column_count']}")


if __name__ == "__main__":
    asyncio.run(main())
