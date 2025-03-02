import re
import json
from loguru import logger
from pathlib import Path
from arango import ArangoClient
from arango.database import StandardDatabase
from typing import Dict, Any

from search_tool.arangodb.generate_schema_for_llm.utils.arango_utils import initialize_database
from search_tool.shared_utils.json_utils import save_json_to_file
from search_tool.shared_utils.file_utils import get_project_root


# ----------------------------
# Precompile regex patterns:
# ----------------------------
# CATEGORY_PATTERN matches lines like: "# General Errors #"
CATEGORY_PATTERN = re.compile(r"^#+\s*([^#].*?)\s*#+$")
# ERROR_PATTERN matches error definitions like: "ERROR_NAME = 1234"
ERROR_PATTERN = re.compile(r"^([A-Z0-9_]+)\s*=\s*(\d+)$")

# ----------------------------
# Parse error file:
# ----------------------------

def parse_error_file(file_name: Path, config: dict) -> list[dict]:
    data_dir: Path = config["directories"]["data_dir"]
    file_path: Path = data_dir / file_name
    errors: list[dict] = []
    current_category = None
    last_comment = None  # Holds the description for the next error definition

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            logger.info(f"Starting to parse error file: {file_path}")
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Check for a category header using the precompiled regex
                category_match = CATEGORY_PATTERN.match(line)
                if category_match:
                    current_category = category_match.group(1)
                    logger.debug(f"Found new category: {current_category}")
                    last_comment = None  # Reset description when category changes
                    continue

                # Process comment lines (lines starting with '#' but not matching the header pattern)
                if line.startswith("#"):
                    # Skip lines made solely of '#' characters
                    if set(line) == {"#"}:
                        continue
                    # Otherwise, treat the line as a comment/description
                    last_comment = line.lstrip("#").strip()
                    logger.trace(f"Found comment: {last_comment}")
                    continue

                # Process error definition lines (e.g., "ERROR_NAME = 1234")
                error_match = ERROR_PATTERN.match(line)
                if error_match:
                    error_name = error_match.group(1)
                    error_code = int(error_match.group(2))
                    logger.debug(f"Found error definition: {error_name} = {error_code}")

                    if current_category is None:
                        logger.warning(f"Error {error_name} found before any category definition!")

                    errors.append({
                        "name": error_name,
                        "category": current_category,
                        "code": error_code,
                        "description": last_comment  # May be None if no comment was set
                    })

                    # Reset last_comment after assigning it to an error definition
                    last_comment = None

        logger.success(f"Successfully parsed {len(errors)} errors from file")
        return errors

    except FileNotFoundError:
        logger.error(f"Error file not found: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while parsing error file: {e}")
        raise

def insert_errors_into_database(db: StandardDatabase, errors: list[dict]):
    try:
        # Create or get the 'arango_errors' collection
        if not db.has_collection('arango_errors'):
            collection = db.create_collection('arango_errors')
            collection.add_persistent_index(fields=['code'], unique=True)
            logger.info("Created collection 'arango_errors' with index on 'code'")
        else:
            collection = db.collection('arango_errors')

        result = collection.import_bulk(
            errors,
            on_duplicate='update',
            overwrite=True
        )
        logger.success(f"Documents processed - created: {result['created']}, updated: {result['updated']}, ignored: {result['ignored']}")
   
    except Exception as e:
        logger.exception(f"Failed to insert errors into database: {e}")
        raise

if __name__ == "__main__":
    project_root: Path = get_project_root()
    config = {
            "directories": {
                "data_dir": project_root / "src/search_tool/data",
                "results_dir": project_root / "src/search_tool/data"

            },
            "arango_config": {
                "hosts": "http://localhost:8529",
                "db_name": "verifaix",
                "username": "root",
                "password": "openSesame"
            }
        }

    try:
        
        # Setting up directories and ArangoDB database
        data_dir: Path = config["directories"]["data_dir"]
        results_dir: Path = config["directories"]["results_dir"]
        arango_config: Dict[str, Any] = config.get("arango_config", {})
        db: StandardDatabase = initialize_database(arango_config)
        
        # Parse error file into a list of dictionaries
        file_name: Path = "error_definitions.py"
        logger.info(f"Starting error parser with file: {file_name}")
        errors: list[dict] = parse_error_file(file_name, config)

        # Print or save as JSON
        logger.info(json.dumps(errors, indent=4))
        logger.info("Successfully printed errors in JSON format")
        
        # Save errors to JSON file
        json_file_path: Path = results_dir / "errors.json"
        save_json_to_file(errors, json_file_path)
        logger.info(f"Successfully saved errors to {json_file_path}")

        # Insert errors into ArangoDB database
        insert_errors_into_database(db, errors)
    
        
    except Exception as e:
        logger.exception("Fatal error in main execution")
        raise
