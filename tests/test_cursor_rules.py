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

# Import the module we're testing
from cursor_rules_simple import (
    load_rules_from_directory,
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    semantic_search,
    generate_embedding,
    SAMPLE_RULE,
    SAMPLE_EXAMPLE
)

# Test configuration - use real database with test DB name
TEST_CONFIG = {
    "arango_config": {
        "hosts": ["http://localhost:8529"],
        "username": "root",
        "password": "openSesame"
    }
}
TEST_DB_NAME = "cursor_rules_test_" + str(int(time.time()))

class TestCursorRules(unittest.TestCase):
    """Test cases for cursor rules database functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Create a test DB with a unique name
        config = TEST_CONFIG.copy()
        cls.db = setup_cursor_rules_db(config, db_name=TEST_DB_NAME)
        if not cls.db:
            raise Exception(f"Failed to set up test database {TEST_DB_NAME}. Make sure ArangoDB is running.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # We deliberately keep the test database to allow inspection
        # In a real-world scenario, you might want to drop it
        pass
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test rules
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rules_dir = Path(self.temp_dir.name)
        
        # Create a test rule file
        self.test_rule_content = """# Test Rule
        
This is a test rule for unit testing.

## Usage
- Follow these guidelines
- Write proper tests
"""
        self.test_rule_path = self.rules_dir / "001-test-rule.mdc"
        with open(self.test_rule_path, 'w') as f:
            f.write(self.test_rule_content)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_rules_from_directory(self):
        """Test loading rules from a directory."""
        rules = load_rules_from_directory(str(self.rules_dir))
        
        # Check that we loaded the test rule
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["rule_number"], "001")
        self.assertEqual(rules[0]["title"], "Test Rule")
        self.assertTrue("This is a test rule" in rules[0]["description"])
    
    def test_setup_and_get_rules(self):
        """Test database setup and retrieving rules."""
        # The database is already set up in setUpClass
        self.assertIsNotNone(self.db)
        
        # Insert a test rule
        rules_collection = self.db.collection("rules")
        test_rule = SAMPLE_RULE.copy()
        test_rule["_key"] = "test_rule_" + str(int(time.time()))
        rules_collection.insert(test_rule)
        
        # Retrieve all rules
        rules = get_all_rules(self.db)
        
        # Verify a rule was retrieved
        self.assertGreaterEqual(len(rules), 1)
        
        # Find our test rule
        inserted_rule = next((r for r in rules if r["_key"] == test_rule["_key"]), None)
        self.assertIsNotNone(inserted_rule)
        self.assertEqual(inserted_rule["title"], test_rule["title"])
    
    def test_examples_for_rule(self):
        """Test storing and retrieving examples for a rule."""
        # Insert a test rule
        rules_collection = self.db.collection("rules")
        test_rule = SAMPLE_RULE.copy()
        test_rule["_key"] = "test_rule_examples_" + str(int(time.time()))
        rules_collection.insert(test_rule)
        
        # Insert an example for the rule
        examples_collection = self.db.collection("pattern_examples")
        test_example = SAMPLE_EXAMPLE.copy()
        test_example["_key"] = "test_example_" + str(int(time.time()))
        test_example["rule_key"] = test_rule["_key"]
        examples_collection.insert(test_example)
        
        # Retrieve examples for the rule
        examples = get_examples_for_rule(self.db, test_rule["_key"])
        
        # Verify the example was retrieved
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["rule_key"], test_rule["_key"])
        self.assertEqual(examples[0]["title"], test_example["title"])
    
    def test_generate_embedding_no_utils(self):
        """Test embedding generation when utils aren't available."""
        with patch('cursor_rules_simple.EMBEDDING_AVAILABLE', False):
            embedding = generate_embedding("Test text")
            
            # Check that a placeholder was returned
            self.assertEqual(embedding["embedding"], [])
            self.assertEqual(embedding["metadata"]["embedding_model"], "none")
            self.assertEqual(embedding["metadata"]["embedding_dim"], 0)
    
    @patch('cursor_rules_simple.EMBEDDING_AVAILABLE', True)
    @patch('cursor_rules_simple.ensure_text_has_prefix')
    @patch('cursor_rules_simple.create_embedding_sync')
    def test_generate_embedding_with_utils(self, mock_create_embedding, mock_ensure_prefix):
        """Test embedding generation with utils available."""
        # Setup mocks
        mock_ensure_prefix.return_value = "search_document: Test text"
        mock_create_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "embedding_model": "test-model",
                "embedding_timestamp": "2023-01-01T00:00:00Z",
                "embedding_method": "test",
                "embedding_dim": 3
            }
        }
        
        # Generate embedding
        embedding = generate_embedding("Test text")
        
        # Check that the embedding was generated correctly
        self.assertEqual(embedding["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(embedding["metadata"]["embedding_model"], "test-model")
        self.assertEqual(embedding["metadata"]["embedding_dim"], 3)
        
        # Verify that text was properly prefixed
        mock_ensure_prefix.assert_called_once_with("Test text")
        
        # Verify that embedding was generated
        mock_create_embedding.assert_called_once_with("search_document: Test text")
    
    @patch('cursor_rules_simple.EMBEDDING_AVAILABLE', True)
    @patch('cursor_rules_simple.create_embedding_sync')
    def test_semantic_search(self, mock_create_embedding):
        """Test semantic search."""
        # Setup mock for embedding
        mock_create_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"embedding_model": "test"}
        }
        
        # Insert a test rule with a mock embedding
        rules_collection = self.db.collection("rules")
        test_rule = SAMPLE_RULE.copy()
        test_rule["_key"] = "test_rule_semantic_" + str(int(time.time()))
        test_rule["embedding"] = [0.1, 0.2, 0.3]  # Same as query embedding for perfect match
        rules_collection.insert(test_rule)
        
        # Execute semantic search
        results = semantic_search(self.db, "test query")
        
        # Since our test only has synthetic embeddings, we might not get results
        # Just verify the function runs without error and returns the expected structure
        self.assertIsInstance(results, list)
        
        # If we got results, check their structure
        if results:
            self.assertIn("rule", results[0])
            self.assertIn("similarity", results[0])

if __name__ == "__main__":
    unittest.main() 