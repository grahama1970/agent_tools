"""Integration modules for the extraction system."""

from .fetch_docs_adapter import DocumentationDownloader, HTMLProcessor
from .qa_adapter import QAIntegration, QuestionGenerator
from .validation_adapter import ExtractionValidator, QualityChecker

__all__ = [
    "DocumentationDownloader", "HTMLProcessor",
    "QAIntegration", "QuestionGenerator",
    "ExtractionValidator", "QualityChecker",
]
