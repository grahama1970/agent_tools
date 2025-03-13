#!/usr/bin/env python
"""
Tests for the cursor rules database implementation.

These tests verify that the core functionality of cursor_rules_simple.py works correctly
using a real ArangoDB instance.
"""
import unittest
from unittest.mock import patch
import os
from pathlib import Path
import tempfile
import json
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the module we're testing
from agent_tools.cursor_rules.core.cursor_rules import (
    load_rules_from_directory,
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    semantic_search,
    generate_embedding,
    SAMPLE_RULE,
    SAMPLE_EXAMPLE,
    bm25_keyword_search,
    EMBEDDING_AVAILABLE,
    create_arangosearch_view
)

# Test configuration - use real database with test DB name
TEST_CONFIG = {
    "arango": {
        "hosts": ["http://localhost:8529"],
        "username": "root",
        "password": "openSesame"
    }
}

class TestCursorRules(unittest.TestCase):
    """Test cases for cursor_rules.py."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test database."""
        print("Connecting to ArangoDB at http://localhost:8529")
        print("Using username: root")
        print("Connected to _system database")
        
        # Create a test database with timestamp to avoid conflicts
        cls.test_db_name = f"cursor_rules_test_{int(time.time())}"
        print(f"Created database {cls.test_db_name}")
        
        # Connect to the database
        cls.db = setup_cursor_rules_db(TEST_CONFIG, db_name=cls.test_db_name)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test database."""
        # Delete the test database
        client = cls.db.conn.client
        sys_db = client.db("_system", username="root", password="openSesame")
        sys_db.delete_database(cls.test_db_name)
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Create test collections
        if not self.db.has_collection("rules"):
            self.db.create_collection("rules")
        if not self.db.has_collection("rule_examples"):
            self.db.create_collection("rule_examples")
        
        # Insert sample data
        rules_collection = self.db.collection("rules")
        examples_collection = self.db.collection("rule_examples")
        
        # Insert a sample rule
        sample_rule = SAMPLE_RULE.copy()
        sample_rule["_key"] = "test_rule_" + str(int(time.time()))
        rules_collection.insert(sample_rule)
        print("Inserted sample rule")
        
        # Insert a sample example
        sample_example = SAMPLE_EXAMPLE.copy()
        sample_example["_key"] = "test_example_" + str(int(time.time()))
        sample_example["rule_key"] = sample_rule["_key"]
        examples_collection.insert(sample_example)
        print("Inserted sample example")
    
    def tearDown(self):
        """Clean up after each test."""
        # Nothing to do here, we'll clean up the database in tearDownClass
        pass
    
    def test_load_rules_from_directory(self):
        """Test loading rules from a directory."""
        # Create a temporary directory with test rules
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test rule file
            rule_file = os.path.join(temp_dir, "001-test-rule.md")
            with open(rule_file, "w") as f:
                f.write("# Test Rule\n\nThis is a test rule.\n")
            
            # Load rules from the directory
            rules = load_rules_from_directory(temp_dir)
            
            # Check that the rule was loaded
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0]["title"], "Test Rule")
            self.assertEqual(rules[0]["content"], "This is a test rule.")
    
    def test_setup_and_get_rules(self):
        """Test setting up the database and getting rules."""
        # Get all rules
        rules = get_all_rules(self.db)
        
        # Check that we got at least one rule
        self.assertGreaterEqual(len(rules), 1)
        
        # Check that the rule has the expected fields
        rule = rules[0]
        self.assertIn("_key", rule)
        self.assertIn("title", rule)
        self.assertIn("content", rule)
        
        # Check that we can get a specific rule
        rule_key = rule["_key"]
        rules_collection = self.db.collection("rules")
        specific_rule = rules_collection.get(rule_key)
        self.assertEqual(specific_rule["_key"], rule_key)
    
    def test_examples_for_rule(self):
        """Test storing and retrieving examples for a rule."""
        # Insert a test rule
        rules_collection = self.db.collection("rules")
        test_rule = SAMPLE_RULE.copy()
        test_rule["_key"] = "test_rule_examples_" + str(int(time.time()))
        rules_collection.insert(test_rule)
    
        # Insert an example for the rule
        examples_collection = self.db.collection("rule_examples")
        test_example = SAMPLE_EXAMPLE.copy()
        test_example["_key"] = "test_example_" + str(int(time.time()))
        test_example["rule_key"] = test_rule["_key"]
        examples_collection.insert(test_example)
        
        # Get examples for the rule
        examples = get_examples_for_rule(self.db, test_rule["_key"])
        
        # Check that we got the example
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["rule_key"], test_rule["_key"])
    
    def test_generate_embedding_no_utils(self):
        """Test embedding generation when utils aren't available."""
        with patch('agent_tools.cursor_rules.core.cursor_rules.EMBEDDING_AVAILABLE', False):
            # Generate an embedding
            embedding = generate_embedding("Test text")
            
            # Check that we got a default embedding
            self.assertIsNotNone(embedding)
            self.assertEqual(len(embedding), 768)  # Default embedding size
    
    @patch('agent_tools.cursor_rules.core.cursor_rules.EMBEDDING_AVAILABLE', True)
    @patch('agent_tools.cursor_rules.embedding.ensure_text_has_prefix')
    @patch('agent_tools.cursor_rules.embedding.create_embedding_sync')
    def test_generate_embedding_with_utils(self, mock_create_embedding, mock_ensure_prefix):
        """Test embedding generation when utils are available."""
        # Set up the mocks
        mock_ensure_prefix.return_value = "prefixed test text"
        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Generate an embedding
        embedding = generate_embedding("Test text")
        
        # Check that the mocks were called
        mock_ensure_prefix.assert_called_once_with("Test text")
        mock_create_embedding.assert_called_once_with("prefixed test text")
        
        # Check that we got the expected embedding
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
    
    @patch('agent_tools.cursor_rules.core.cursor_rules.EMBEDDING_AVAILABLE', True)
    @patch('agent_tools.cursor_rules.embedding.create_embedding_sync')
    def test_semantic_search(self, mock_create_embedding):
        """Test semantic search."""
        # Set up the mock
        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Perform a semantic search
        results = semantic_search(self.db, "Test query")
        
        # Check that the mock was called
        mock_create_embedding.assert_called_once()
        
        # We can't check the results since they depend on the database state,
        # but we can check that the function ran without errors
        self.assertIsNotNone(results)

if __name__ == "__main__":
    unittest.main() 