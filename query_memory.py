#!/usr/bin/env python3
"""
Query the memory database and display results.
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path

def query_db(db_path, query, limit=None):
    """Run an SQL query against the database."""
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        if limit is not None:
            query = f"{query} LIMIT {limit}"
            
        cursor.execute(query)
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Fetch results
        results = cursor.fetchall()
        
        # Print headers
        print("=" * 100)
        print(" | ".join(columns))
        print("=" * 100)
        
        # Print results
        for row in results:
            print(" | ".join(str(item) for item in row))
            
        print(f"\nTotal results: {len(results)}")
        
    except sqlite3.Error as e:
        print(f"SQL Error: {e}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Query the memory database")
    parser.add_argument("--db", default="extraction_memory.db", help="Path to the database")
    parser.add_argument("--query", required=True, help="SQL query to execute")
    parser.add_argument("--limit", type=int, help="Optional limit on results")
    
    args = parser.parse_args()
    
    query_db(args.db, args.query, args.limit)

if __name__ == "__main__":
    main()