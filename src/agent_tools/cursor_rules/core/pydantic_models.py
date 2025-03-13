"""
Pydantic models for the Cursor Rules package.

This module contains the data models used throughout the Cursor Rules system,
using Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Optional, Callable
from arango import ArangoClient


class CursorRulesDatabase(BaseModel):
    """Database configuration and connection for Cursor Rules."""
    
    # Configuration
    hosts: Union[str, List[str]] = Field(
        default=["http://localhost:8529"], 
        description="ArangoDB host(s)"
    )
    username: str = Field(default="root", description="ArangoDB username")
    password: str = Field(default="openSesame", description="ArangoDB password")
    database_name: str = Field(default="cursor_rules", description="Database name")
    
    # Properties
    db: Any = Field(default=None, exclude=True, description="Database connection")
    
    def connect(self) -> Any:
        """Connect to the database and return the connection."""
        client = ArangoClient(hosts=self.hosts[0] if isinstance(self.hosts, list) else self.hosts)
        self.db = client.db(
            self.database_name,
            username=self.username,
            password=self.password
        )
        return self.db

    class Config:
        arbitrary_types_allowed = True


class RuleExample(BaseModel):
    """Example illustrating a coding rule or pattern."""
    
    title: str = Field(..., description="Title of the example")
    description: str = Field(..., description="Description of what the example demonstrates")
    language: str = Field(default="python", description="Programming language of the example")
    good_example: str = Field(..., description="Code example showing the correct pattern")
    bad_example: Optional[str] = Field(None, description="Code example showing the incorrect pattern")
    rule_key: str = Field(..., description="Key of the rule this example belongs to")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule_key": "001-code-advice-rules",
                "title": "Type Hints Example",
                "description": "Demonstrates proper use of type hints",
                "language": "python",
                "good_example": "def add(a: int, b: int) -> int:\n    return a + b",
                "bad_example": "def add(a, b):\n    return a + b"
            }
        }


class Rule(BaseModel):
    """Coding rule or pattern definition."""
    
    rule_number: str = Field(..., description="Numeric identifier (e.g., '001')")
    title: str = Field(..., description="Rule title")
    description: str = Field(..., description="Brief description of the rule")
    content: str = Field(..., description="Full markdown content of the rule")
    glob_pattern: Optional[str] = Field(None, description="File pattern where rule applies")
    priority: Optional[int] = Field(None, description="Priority order (lower = more important)")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic search")
    embedding_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the embedding")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule_number": "001",
                "title": "Use Type Hints",
                "description": "Always use type hints for function parameters and return values",
                "content": "# Type Hints\n\nAlways use type hints to improve code readability and enable static type checking.",
                "glob_pattern": "*.py",
                "priority": 1
            }
        }


class CursorRules(BaseModel):
    """Main class for interacting with cursor rules."""
    
    database: CursorRulesDatabase = Field(
        default_factory=CursorRulesDatabase,
        description="Database configuration"
    )
    rules_dir: Optional[str] = Field(
        default=None, 
        description="Directory containing rule files"
    )
    
    # These methods will be implemented functionally outside the class
    # but we define them here for documentation and typing purposes
    setup: Callable = Field(default=None, exclude=True)
    search: Callable = Field(default=None, exclude=True)
    get_rule: Callable = Field(default=None, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True


class SearchResult(BaseModel):
    """Result from a search query."""
    
    rule: Rule = Field(..., description="The matched rule")
    score: float = Field(..., description="Match score (higher is better)")
    matches: Optional[List[str]] = Field(None, description="Highlighted matched text segments")


class QueryOptions(BaseModel):
    """Options for querying the database."""
    
    search_type: str = Field(default="hybrid", description="Search type: 'bm25', 'semantic', or 'hybrid'")
    limit: int = Field(default=5, description="Maximum number of results to return")
    verbose: bool = Field(default=False, description="Whether to print detailed information during search")


class RuleFilter(BaseModel):
    """Filter criteria for rules."""
    
    rule_number: Optional[str] = Field(None, description="Filter by rule number")
    glob_pattern: Optional[str] = Field(None, description="Filter by file pattern")
    priority: Optional[int] = Field(None, description="Filter by priority")
    keyword: Optional[str] = Field(None, description="Filter by keyword in title or description") 