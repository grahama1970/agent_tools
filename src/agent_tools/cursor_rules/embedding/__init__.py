"""
Embedding functionality for cursor_rules package.

This module provides functionality for generating vector embeddings
from text using various embedding models.
"""

from agent_tools.cursor_rules.embedding.embedding_utils import (
    create_embedding,
    create_embedding_sync,
    create_embedding_with_sentence_transformer,
    ensure_text_has_prefix,
    get_model_and_tokenizer,
    get_sentence_transformer
)

__all__ = [
    'create_embedding',
    'create_embedding_sync',
    'create_embedding_with_sentence_transformer',
    'ensure_text_has_prefix',
    'get_model_and_tokenizer',
    'get_sentence_transformer'
]
