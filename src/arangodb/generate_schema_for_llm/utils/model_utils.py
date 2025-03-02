from typing import Literal, List, Dict
from pydantic import BaseModel, Field

###
# Pydantic Models for Structured Outputs
###
class FieldDescription(BaseModel):
    """Describes a field in an ArangoDB collection"""
    name: str = Field(..., description="Name of the field")
    description: str = Field(..., description="Concise 1 sentence that describes the primary purpose of the field")
    type: Literal["string", "number", "boolean", "array", "object", "date", "unknown"] = Field(
        ...,
        description="Type of the field, e.g., string, number, boolean, array, object, date, etc."
    )
    
class CollectionDescription(BaseModel):
    """Describes an ArangoDB collection and its properties"""
    name: str = Field(..., description="Name of the collection")
    type: Literal["document", "edge"] = Field(..., description="Type of collection (document or edge)")
    fields: List[FieldDescription] = Field(..., description="List of fields in the collection")
    description: str = Field(..., description="Concise 1-2 sentence description of the collection's primarypurpose")

class ViewDescription(BaseModel):
    """Describes an ArangoDB view configuration"""
    name: str = Field(..., description="Name of the view")
    type: str = Field(..., description="Type of view (e.g., arangosearch)")
    linked_collections: List[str] = Field(..., description="Collections linked to this view")
    analyzers: List[str] = Field(..., description="Text analyzers used in the view")
    description: str = Field(..., description="Concise 1-2 sentence description of the view's primary purpose")

class AnalyzerDescription(BaseModel):
    """Describes an ArangoDB analyzer configuration"""
    name: str = Field(..., description="Name of the analyzer")
    description: str = Field(..., description="Concise 1-2 sentence description of the analyzer's primary purpose")

class SchemaDescription(BaseModel):
    """Complete description of an ArangoDB database schema"""
    collections: List[CollectionDescription] = Field(..., description="List of collections in the database")
    views: List[ViewDescription] = Field(..., description="List of views in the database")
    analyzers: List[AnalyzerDescription] = Field(..., description="List of custom analyzers")
    example_queries: List[Dict] = Field(..., description="Example AQL queries for common operations")
    relationships: List[Dict] = Field(default_factory=list, description="Relationships between collections")


if __name__ == "__main__":
    print('loaded')