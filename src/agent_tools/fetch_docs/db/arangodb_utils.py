"""
ArangoDB utilities for storing and retrieving web page content.
Uses python-arango's native functionality for database operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json
import asyncio
import os
from arango.client import ArangoClient
from arango.database import StandardDatabase
from loguru import logger
from ..models import PageDocument, ContentDocument, ExtractionResult, PageMetadata
from arango.exceptions import ArangoError, DocumentInsertError, DocumentGetError
from fetch_page.embedding.embedding_utils import create_embedding_sync


def get_db(db_url: str, db_name: str) -> StandardDatabase:
    """Get database connection and ensure collections exist.

    Args:
        db_url: ArangoDB server URL.
        db_name: Database name.

    Returns:
        Database handle with initialized collections.
    """
    client = ArangoClient(hosts=db_url)
    root_password = os.getenv("ARANGO_ROOT_PASSWORD", "rootpassword")
    
    # First connect to _system database to create our database
    sys_db = client.db('_system', username='root', password=root_password)
    
    # Create target database if it doesn't exist
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    
    # Connect to target database
    db = client.db(db_name, username='root', password=root_password)
    
    # Create collections if they don't exist
    if not db.has_collection('pages'):
        db.create_collection('pages')
    if not db.has_collection('content_items'):
        db.create_collection('content_items')

    return db


def generate_page_key(url: str) -> str:
    """Generate a unique key for a page based on its URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def process_table_metadata(table_json: str) -> Dict[str, Any]:
    """Process table metadata from JSON string."""
    try:
        return json.loads(table_json) if table_json else {}
    except json.JSONDecodeError:
        return {}


async def store_page_content(db: StandardDatabase, result: ExtractionResult) -> Dict[str, Any]:
    """Store or update page content using native ArangoDB operations.

    Args:
        db: Database handle.
        result: Extraction result containing page and content.

    Returns:
        Dict containing page_key and list of content_keys.
    """
    try:
        pages_coll = db.collection("pages")
        content_coll = db.collection("content_items")

        # Generate a unique key for the page
        page_key = generate_page_key(result.page.url)
        
        # Prepare page document
        page_doc = {
            '_key': page_key,
            'url': result.page.url,
            'title': result.page.title,
            'summary': result.page.summary,
            'token_count': result.page.token_count,
            'LLM_model': result.page.LLM_model,
            'fetch_timestamp': datetime.utcnow().isoformat()
        }

        # Add embedding to the document
        page_doc["embedding"] = create_embedding_for_page(result)

        # Store page document
        pages_coll.insert(page_doc, overwrite=True)
        
        # Store content items
        content_keys = []
        for section in result.content_list:
            for item in section.contents:
                content_key = f"{page_key}_{section.position}_{item.position}"
                content_keys.append(content_key)
                
                content_doc = {
                    '_key': content_key,
                    'page_key': page_key,
                    'section_name': section.section_name,
                    'section_position': section.position,
                    'content_position': item.position,
                    'type': item.type,
                    'content': item.content,
                    'src': item.src,
                    'description': item.description,
                    'table_metadata': process_table_metadata(item.content) if item.type == 'table' and item.content else None,
                    'fetch_timestamp': datetime.utcnow().isoformat()
                }
                
                content_coll.insert(content_doc, overwrite=True)

        return {"page_key": page_key, "content_keys": content_keys}

    except ArangoError as e:
        logger.error(f"Failed to store page content: {e}")
        raise


async def get_page_content(db: StandardDatabase, url: str) -> Dict[str, Any]:
    """Retrieve page content and its sections using native ArangoDB queries.

    Args:
        db: Database handle.
        url: Page URL to retrieve.

    Returns:
        Dict containing the page document and organized sections.
    """
    try:
        page_key = generate_page_key(url)
        
        # Get page document
        page = db.collection('pages').get(page_key)
        if not page:
            return {"page": None, "sections": {}}
        
        # Get content items
        aql = """
        FOR doc IN content_items
            FILTER doc.page_key == @page_key
            SORT doc.section_position, doc.content_position
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"page_key": page_key})
        content_items = [doc for doc in cursor]
        
        # Organize into sections
        sections = {}
        for item in content_items:
            section_pos = item["section_position"]
            if section_pos not in sections:
                sections[section_pos] = {
                    "section_name": item["section_name"],
                    "contents": []
                }
            sections[section_pos]["contents"].append(item)
        
        return {"page": page, "sections": dict(sorted(sections.items()))}

    except ArangoError as e:
        logger.error(f"Failed to retrieve page content: {e}")
        raise


def create_embedding_for_page(extraction_result: ExtractionResult) -> Dict[str, Any]:
    """Create an embedding for a page object."""
    # Get the title from the page metadata
    title = extraction_result.page.title
    
    # Extract content from the content_list
    content_texts = []
    for section in extraction_result.content_list:
        for item in section.contents:
            if item.type == "text" and item.content:
                content_texts.append(item.content)
    
    # Join the content texts
    content = "\n\n".join(content_texts)
    
    # Combine the title and content for embedding
    text_to_embed = f"search_document: {title}\n\n{content}"
    
    # Create the embedding
    embedding_result = create_embedding_sync(text_to_embed)
    
    return embedding_result


async def main():
    """Main function for testing and debugging database operations."""
    try:
        # Initialize database connection
        db = get_db("http://localhost:8529", "mydb")

        # Example: Create test page metadata
        test_page = PageMetadata(
            url="http://example.com",
            title="Test Page",
            summary="Test summary",
            token_count=100,
            LLM_model="gpt-3.5-turbo",
        )

        # Create test extraction result (add test content as needed)
        result = ExtractionResult(
            page=test_page,
            content_list=[],
        )

        # Test store operation (upsert)
        stored = await store_page_content(db, result)
        logger.info(f"Stored page with key: {stored['page_key']}")

        # Test retrieve operation
        content = await get_page_content(db, "http://example.com")
        if content["page"]:
            logger.info(f"Retrieved page: {content['page']['title']}")
            logger.info(f"Number of sections: {len(content['sections'])}")
        else:
            logger.warning("Page not found")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
