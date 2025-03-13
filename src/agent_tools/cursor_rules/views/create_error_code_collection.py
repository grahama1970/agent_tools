"""
This script handles the creation and population of an ArangoDB collection containing error codes and messages.
It downloads an error code dataset from a specified URL, parses it into JSON format, and stores it in a collection
for easy reference and querying. The script is designed to be run as a standalone utility to initialize or update
the error code collection in an ArangoDB database.
"""

import requests
import csv
from arango import ArangoClient
from loguru import logger
from pathlib import Path


def download_file(url: str, local_file: str) -> bool:
    """
    Downloads a file from the specified URL and saves it locally.
    """
    logger.info("Downloading ArangoDB errors.dat file...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_file, "w", encoding="utf-8") as file:
            file.write(response.text)
        logger.success("File downloaded successfully.")
        return True
    except requests.RequestException as e:
        logger.exception(f"Failed to download file: {e}")
        return False


def parse_errors(local_file: str):
    """
    Parses the downloaded errors.dat file into a list of JSON objects.
    """
    logger.info("Processing errors.dat into JSON format...")
    errors = []
    try:
        with open(local_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)  # Handles quoted fields properly
            for row in reader:
                if len(row) == 4 and not row[0].startswith("#"):  # Ignore comments
                    try:
                        error_entry = {
                            "_key": str(
                                row[1].strip()
                            ),  # Use error code as _key for upsert
                            "code": int(
                                row[1].strip()
                            ),  # Second field is the numeric error code
                            "name": row[0].strip(),  # First field is the error name
                            "message": row[3]
                            .strip()
                            .strip('"'),  # Fourth field is full error message
                        }
                        errors.append(error_entry)
                    except ValueError:
                        logger.warning(f"Skipping invalid row: {row}")
        logger.success(f"Processed {len(errors)} error codes.")
        return errors
    except Exception as e:
        logger.exception(f"Error processing file: {e}")
        return None


def upsert_errors(db, errors, collection_name: str):
    """
    Upserts error records into the specified ArangoDB collection using AQL.
    """
    logger.info("Upserting error records into ArangoDB...")
    aql_upsert = f"""
    FOR error IN @errors
        UPSERT {{ _key: error._key }}
        INSERT error
        UPDATE error
        IN {collection_name}
    """
    try:
        db.aql.execute(aql_upsert, bind_vars={"errors": errors})
        logger.success(f"Upserted {len(errors)} error records.")
    except Exception as e:
        logger.exception(f"Error during upsert: {e}")
        raise


def download_and_store_arango_errors(
    host: str, username: str, password: str, db_name: str = "arangodb"
):
    """
    Downloads the ArangoDB errors.dat file, processes it into JSON,
    and stores/upserts it into an ArangoDB collection.

    :param host: ArangoDB host URL (e.g., "http://localhost:8529")
    :param username: ArangoDB username
    :param password: ArangoDB password
    :param db_name: Name of the database (default: "arangodb")
    """
    url = "https://raw.githubusercontent.com/arangodb/arangodb/devel/lib/Basics/errors.dat"
    local_file = "errors.dat"
    collection_name = "arango_errors"

    if not download_file(url, local_file):
        return

    errors = parse_errors(local_file)
    if errors is None:
        return

    logger.info("Connecting to ArangoDB...")
    try:
        client = ArangoClient(hosts=host)
        sys_db = client.db("_system", username=username, password=password)
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            logger.success(f"Database '{db_name}' created.")
        db = client.db(db_name, username=username, password=password)
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            logger.success(f"Collection '{collection_name}' created.")
        upsert_errors(db, errors, collection_name)
    except Exception as e:
        logger.exception(f"ArangoDB connection or insertion error: {e}")
        return

    # Cleanup: Remove the downloaded file using pathlib and wrap in try/except block
    file_path = Path(local_file)
    try:
        file_path.unlink()
        logger.info("Temporary file removed. Process completed successfully.")
    except Exception as e:
        logger.exception(f"Error removing temporary file {local_file}: {e}")


if __name__ == "__main__":
    download_and_store_arango_errors(
        host="http://localhost:8529",
        username="root",
        password="openSesame",
        db_name="verifaix",
    )
