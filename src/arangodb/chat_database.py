import time
import spacy
from arango import ArangoClient
import os
from loguru import logger

from smolagent.utils.file_utils import get_project_root, load_env_file, load_text_file
from smolagent.database.initialize_database import initialize_database

project_root = get_project_root()
load_env_file('.env')


class ChatDatabase:
    def __init__(self, config: dict, db_name="smol_chat", collection_name="chat_history"):
        self.config = config
        self._connect_to_db()  # ← Before this line add:
        if not config.get("arango_config"):
            raise ValueError("Missing arango_config in database configuration")
        self.db_name = db_name
        self.collection_name = collection_name
        self.db = None
        self.collection = None
        self._connect_to_db()
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for tokenization

    def _connect_to_db(self):
        """Simplified connection using initialize_database"""
        logger.info("Connecting to ArangoDB...")
        self.db = initialize_database(self.config)

        if not self.db.has_collection(self.collection_name):
            self.db.create_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' created.")

        self.collection = self.db.collection(self.collection_name)
        logger.success("Database connection established")

    def _token_count(self, text):
        """
        Returns the token count for a given text using spaCy.
        """
        doc = self.nlp(text)
        return len(doc)

    def store_chat(self, user_id, user_query, smolagent_response, response_embedding, embedding_metadata):
        """
        Stores a single chat interaction in the database with a last_updated timestamp and token count.
        """
        query_tokens = self._token_count(user_query)
        response_tokens = self._token_count(smolagent_response)

        self.collection.insert(
            {
                "user_id": user_id,
                "user_query": user_query,
                "smolagent_response": smolagent_response,
                "query_tokens": query_tokens,
                "response_tokens": response_tokens,
                "total_tokens": query_tokens + response_tokens,
                "last_updated": int(time.time()),  # Unix timestamp
                "response_embedding": response_embedding,
                "embedding_metadata": embedding_metadata
            }
        )
        logger.info("Chat stored in ArangoDB with token count and last_updated field.")

    def get_chat_history(self, user_id, max_tokens=6000):
        """
        Retrieves the chat history for a user, limited by a total token count.
        """
        cursor = self.collection.find(
            {"user_id": user_id}, sort_field="last_updated", sort_order="asc"
        )
        history = []
        total_tokens = 0

        for chat in cursor:
            total_tokens += chat["total_tokens"]
            if total_tokens > max_tokens:
                break
            history.append(chat)

        return history
