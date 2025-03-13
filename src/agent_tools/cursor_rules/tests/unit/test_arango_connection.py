#!/usr/bin/env python3
"""Simple script to test ArangoDB connection."""

from arango import ArangoClient
import sys

def test_connection():
    print("Creating client...")
    client = ArangoClient(hosts='http://localhost:8529')
    
    print("Attempting to connect...")
    try:
        db = client.db('cursor_rules_test_scenarios', username='root', password='openSesame')
        print("Connected successfully!")
        print(f"Database version: {db.version()}")
        return True
    except Exception as e:
        print(f"Connection failed: {str(e)}", file=sys.stderr)
        return False

if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1) 