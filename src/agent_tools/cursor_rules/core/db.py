"""
Database utilities for Cursor Rules.

!!! TECHNICAL DEBT WARNING !!!
This module currently coexists with enhanced_db.py, which violates our architectural principles
from LESSONS_LEARNED.md. This situation needs to be addressed in a dedicated refactoring effort.

Current Status:
- This module (db.py) provides basic database connection functionality
- enhanced_db.py provides advanced graph features and search capabilities
- Both are actively used in the codebase

Future Plan:
1. This technical debt should be addressed in a dedicated refactoring sprint
2. The functionality should be consolidated into a single implementation
3. Until then, use enhanced_db.py for new development requiring graph or advanced search features
4. Use this module (db.py) only for basic database connections in existing code

For new development:
- Prefer enhanced_db.py as it contains the more complete implementation
- See enhanced_db.py for graph database features and advanced search
- Reference LESSONS_LEARNED.md for database integration patterns

Related Documentation:
- ArangoDB Python Driver: https://python-arango.readthedocs.io/
- Project LESSONS_LEARNED.md: Directory Structure and Database Integration sections
"""

import os
import asyncio
from typing import Optional

from arango import ArangoClient
from arango.database import StandardDatabase
from loguru import logger

DEFAULT_HOST = "http://localhost:8529"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "openSesame"
DEFAULT_DB_NAME = "cursor_rules"


def get_db(
    host: str = DEFAULT_HOST,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    db_name: str = DEFAULT_DB_NAME,
) -> StandardDatabase:
    """
    Get a connection to the ArangoDB database.

    Args:
        host: ArangoDB host URL
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name

    Returns:
        StandardDatabase: ArangoDB database connection
    """
    client = ArangoClient(hosts=host)
    return client.db(db_name, username=username, password=password)


def create_database(
    host: str = DEFAULT_HOST,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    db_name: str = DEFAULT_DB_NAME,
) -> StandardDatabase:
    """
    Create a database if it doesn't exist and return a connection to it.

    Args:
        host: ArangoDB host URL
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name

    Returns:
        StandardDatabase: ArangoDB database connection
    """
    client = ArangoClient(hosts=host)
    sys_db = client.db("_system", username=username, password=password)
    
    # Check if database exists
    if not db_name in sys_db.databases():
        logger.info(f"Creating database {db_name}")
        sys_db.create_database(db_name)
    else:
        logger.info(f"Database {db_name} already exists")
    
    # Connect to the database
    return client.db(db_name, username=username, password=password)


async def get_db_async(
    host: str = DEFAULT_HOST,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    db_name: str = DEFAULT_DB_NAME,
) -> StandardDatabase:
    """
    Get a connection to the ArangoDB database asynchronously.

    Args:
        host: ArangoDB host URL
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name

    Returns:
        StandardDatabase: ArangoDB database connection
    """
    return await asyncio.to_thread(
        get_db, host=host, username=username, password=password, db_name=db_name
    )


async def create_database_async(
    host: str = DEFAULT_HOST,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    db_name: str = DEFAULT_DB_NAME,
) -> StandardDatabase:
    """
    Create a database if it doesn't exist and return a connection to it asynchronously.

    Args:
        host: ArangoDB host URL
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name

    Returns:
        StandardDatabase: ArangoDB database connection
    """
    return await asyncio.to_thread(
        create_database, host=host, username=username, password=password, db_name=db_name
    ) 