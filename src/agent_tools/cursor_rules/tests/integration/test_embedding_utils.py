"""Tests for embedding utilities."""

import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F

from agent_tools.cursor_rules.embedding import ensure_text_has_prefix, create_embedding_sync


class DummyModelOutput:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class DummyModel:
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        # Create a dummy last_hidden_state with shape (1, sequence_length, hidden_dim) = (1, 3, 2)
        last_hidden_state = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
        return DummyModelOutput(last_hidden_state)


def dummy_tokenizer(text, padding=True, truncation=True, return_tensors='pt'):
    # Produce a dummy encoded output mimicking the tokenizer
    # Make sure to create tensors on CPU to avoid device mismatch with the model's output
    return {
        'input_ids': torch.tensor([[1, 2, 3]], device='cpu'),
        'attention_mask': torch.tensor([[1, 1, 1]], device='cpu')
    }


class TestEmbeddingUtils(unittest.TestCase):
    def test_ensure_text_has_prefix(self):
        # Test when text does not have a prefix
        input_text = "Hello, world!"
        result = ensure_text_has_prefix(input_text)
        self.assertTrue(result.startswith("search_document:"), "Text should be prefixed with 'search_document:'")
        
        # Test when text already has a prefix
        prefixed_text = "search_query: Hello, world!"
        result2 = ensure_text_has_prefix(prefixed_text)
        self.assertEqual(result2, prefixed_text, "Text with existing prefix should remain unchanged")

    @patch("agent_tools.cursor_rules.embedding.embedding_utils.get_model_and_tokenizer")
    @patch("torch.cuda.is_available", return_value=False)  # Force CPU usage during test
    def test_create_embedding_sync(self, mock_cuda_available, mock_get_model_and_tokenizer):
        # Setup dummy model and tokenizer via patch
        dummy_model = DummyModel()
        mock_get_model_and_tokenizer.return_value = (dummy_model, dummy_tokenizer)
        
        # Call create_embedding_sync with a sample input
        result = create_embedding_sync("Test input")
        
        # Check that result is a dict with keys 'embedding' and 'metadata'
        self.assertIn("embedding", result)
        self.assertIn("metadata", result)
        
        # Check that embedding is a list of floats
        self.assertIsInstance(result["embedding"], list)
        self.assertTrue(all(isinstance(x, float) for x in result["embedding"]))
        
        # Validate metadata contains required keys
        metadata = result["metadata"]
        self.assertIn("embedding_model", metadata)
        self.assertIn("embedding_timestamp", metadata)
        self.assertIn("embedding_method", metadata)
        self.assertIn("embedding_dim", metadata)
        
        # The embedding_method should be 'local'
        self.assertEqual(metadata["embedding_method"], "local")
        
        # Check that embedding_dim matches the length of embedding
        self.assertEqual(metadata["embedding_dim"], len(result["embedding"]))


if __name__ == "__main__":
    unittest.main() 