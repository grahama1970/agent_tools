from smolagents.tools import Tool
from smolagents import LiteLLMModel
from pathlib import Path
from loguru import logger
import regex as re
from typing import Dict
from arango import ArangoClient
import arango

from aql_rag.utils.get_project_root import get_project_root
from aql_rag.embedding.embedding_utils import create_embedding_sync

project_dir = get_project_root()


class GetBM25Search(Tool):
    name = "bm25_search"
    description = """
    A tool that that fetch the most relevant documents (by BM25) from a ArangoDB view based on a query.
    Args:
        query: A string representing a valid timezone (e.g., 'America/New_York').
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "object"

    @staticmethod
    def load_aql_query(file_path: str | Path) -> str:
        """Load AQL query from file."""
        try:
            path = Path(file_path) if isinstance(file_path, str) else file_path
            with open(path, "r") as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Failed to load AQL query from {file_path}: {e}")
            raise

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize the input text by removing HTML, special characters, and extra spaces."""

        regex_patterns = [
            (r"<[^>]+>", ""),  # Remove HTML tags
            (r"[^a-zA-Z0-9 ]", ""),  # Remove special characters
            (r"\s+", " "),  # Normalize whitespace
        ]

        normalized_text = text
        for pattern, replacement in regex_patterns:
            normalized_text = re.sub(pattern, replacement, normalized_text)

        return normalized_text.strip()
    
    

    def forward(self, query: str) -> Dict[str, str]:
        try:
            path = project_dir / "aql_rag" / "arango" / "aql" / "bm25_embedding_keyword_combined.aql"
            aql = self.load_aql_query(path)
            return {"aql": aql, "query": query}
            # query = self.normalize_text(query)

            # return f"The current local time in {timezone} is: {local_time}"
        except Exception as e:
            return {"error": str(e), "path": str(path)}


if __name__ == "__main__":
    tool = GetBM25Search()
    tool.forward("What is the capital of France?")
