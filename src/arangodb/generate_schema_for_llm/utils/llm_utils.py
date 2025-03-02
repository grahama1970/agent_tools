import asyncio
from loguru import logger
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
import uuid

from src.generate_schema_for_llm.shared.litellm_call import litellm_call
from src.generate_schema_for_llm.utils.model_utils import CollectionDescription, FieldDescription, ViewDescription, AnalyzerDescription



async def describe_fields_and_collections(
        db, 
        collection_name: str, 
        collection_type: str, 
        sample_rows: List[Dict], 
        request_id: str, 
        config: Dict
) -> Dict:
    """Describe fields and collections using the LLM."""
    llm_config = config.get("llm_config", {})
    if not sample_rows:
        return {"collection_description": "No sample rows available.", "field_descriptions": {}}

    # Generate collection description
    system_message = (
    "You are a DBA expert that concisely explains the purpose and structure of ArangoDB collections. "
    "Always output in well-formatted JSON with no additional text."
    )
    user_message = (
        f"Describe the purpose of the collection '{collection_name}' in 1-2 sentences, focusing on its primary function and the type of data it stores. "
        f"Use the following sample rows as context:\n"
        f"{sample_rows}\n\n"
        "Do not include technical details or field names. Keep the description concise and clear."
    )
    messages = [ {"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    llm_config["messages"] = messages
    llm_config["response_format"] = CollectionDescription
    llm_config["request_id"] = request_id

    try:
        collection_description = await litellm_call(config)

    except Exception as e:
        logger.error(f"Error generating collection description: {e}")
        collection_description = CollectionDescription(
            name=collection_name, 
            description="Description unavailable",
            type= collection_type,
            fields=[]
        )

    # Generate field descriptions
    field_descriptions = {}
    if sample_rows:
        sample_row = sample_rows[0]  # Use the first row to infer fields
        for field in sample_row.keys():
            
            # Standardize descriptions for common fields
            if field == "embedding":
                field_descriptions[field] = FieldDescription(
                    name=field, 
                    type="array",
                    description="A numerical vector representing the content of a document or record, used for similarity searches and machine learning tasks."
                )
            elif field == 'embedding_metadata':
                field_descriptions[field] = FieldDescription(
                    name=field, 
                    type="object", 
                    description="Metadata for the embedding process, including model and timestamp."
                )
            else:
                # If not a special field, ask the LLM to describe the field
                try:
                    system_message = (
                        "You are a DBA expert that concisely explains the purpose of fields in ArangoDB collections. "
                        "Always output in well-formatted JSON with no additional text."
                    )

                    user_message = (
                        f"Describe the field '{field}' in the collection '{collection_name}' based on the following sample value:\n"
                        f"{sample_row[field]}\n\n"
                        "Provide a concise 1-sentence description of the field's primary purpose. "
                        "Do not include technical details or examples."
                    )

                    messages = [ {"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
                    llm_config["messages"] = messages
                    llm_config["response_format"] = FieldDescription
                    llm_config["request_id"] = f"{collection_name}_{field}_{uuid.uuid4()}"

                    field_description = await litellm_call(config)
                except Exception as e:
                    logger.error(f"Error generating description for field '{field}': {e}")
                    field_description = FieldDescription(
                        name=field, 
                        type="object",
                        description="Description unavailable"
                    )

                field_descriptions[field] = field_description

    results = {
        "collection_description": collection_description,
        "field_descriptions": field_descriptions,
    }
    return results

# Async function to describe analyzers
async def describe_analyzer(analyzer: Dict, request_id: str, config: Dict) -> AnalyzerDescription:
    """Describe an ArangoDB analyzer."""
    try:
        llm_config = config.get("llm_config", {})
        system_message = (
        "You are a DBA expert that concisely explains the purpose of ArangoDB analyzers. "
        "Always output in well-formatted JSON with no additional text."
        )
        user_message = (
            f"Describe the purpose of the analyzer '{analyzer['name']}' with the following properties:\n"
            f"Type: {analyzer['type']}\n"
            f"Properties: {analyzer['properties']}\n\n"
            "Provide a concise 1-2 sentence description of its key features and use cases. "
            "Avoid technical jargon and focus on its primary purpose."
        )

        messages = [ {"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        llm_config["messages"] = messages
        llm_config["response_format"] = AnalyzerDescription
        llm_config["request_id"] = f"analyzer_{analyzer['name']}_{request_id}_{uuid.uuid4()}"

        response = await litellm_call(config)
        return response
    except Exception as e:
        logger.error(f"Error generating analyzer description: {e}")
        return AnalyzerDescription(
            name=analyzer['name'], 
            description="Description unavailable"
        )

    # return await generate_description(prompt, AnalyzerDescription, request_id, llm_config)

# Async function to describe views
async def describe_view(view: Dict, request_id: str, config: Dict) -> ViewDescription:
    llm_config = config.get("llm_config", {})
    view_type = view.get("type", None)
    view_links = view.get("links", {})
    view_analyzers = view.get("analyzers", [])
    view_fields = view.get("fields", []) #indexed

    system_message = (
        "You are a DBA expert that concisely explains the purpose of ArangoDB views. "
        "Always output in well-formatted JSON with no additional text."
    )
    user_message = (
        f"Describe the purpose of the view '{view['name']}' with the following properties:\n"
        f"Type: {view_type}\n"
        f"Linked Collections: {view_links}\n"
        f"Analyzers: {view_analyzers}\n\n"
        "Provide a concise 1-2 sentence description of how this view is meant to be used. "
        "Focus on its primary function and avoid technical jargon."
    )

    messages = [ {"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    llm_config["messages"] = messages
    llm_config["response_format"] = ViewDescription
    llm_config["request_id"] = request_id
    response = await litellm_call(config)
    return response

if __name__ == "__main__":
    print('loaded')