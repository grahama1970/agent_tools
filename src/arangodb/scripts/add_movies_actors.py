import docker
from pathlib import Path
import io
import tarfile
from docker.errors import NotFound, APIError # wierd import


# Variables
CONTAINER_NAME = "gallant_thompson"
DATABASE_NAME = "test"
ARANGO_URL = f"http://localhost:8529/_db/{DATABASE_NAME}"
USERNAME = "root"
PASSWORD = "openSesame"
JAVASCRIPT_FILE = Path("/Users/robert/Desktop/dev/projects/experiments/aql_rag/scripts/populate_movie_db.js")
SERVER_ENDPOINT = "tcp://127.0.0.1:8529"

def create_tar_stream(js_file_path: Path) -> bytes:
    try:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            if not js_file_path.exists():
                raise FileNotFoundError(f"JavaScript file not found: {js_file_path}")
            tar.add(js_file_path, arcname=js_file_path.name)
        
        tar_stream.seek(0)
        return tar_stream.read()
    except (tarfile.TarError, IOError) as e:
        raise Exception(f"Failed to create tar archive: {str(e)}")


def main():
    # Initialize Docker client
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
    except APIError as e:
        print(f"APIError: {e}")
        return
    
    # Check if container is running
    try:
        container = client.containers.get(CONTAINER_NAME)
        if container.status != "running":
            raise Exception(f"Container '{CONTAINER_NAME}' is not running")
    except NotFound as e:
        raise Exception(f"Container '{CONTAINER_NAME}' does not exist")

    # Copy JavaScript file to container
    try:
        tar_data = create_tar_stream(JAVASCRIPT_FILE)
        success = container.put_archive('/tmp', tar_data)
        if not success:
            raise Exception("Failed to copy file to container")
    except APIError as e:
        raise Exception(f"Docker API error while copying file: {str(e)}")

    # Execute JavaScript file
    exec_command = [
        "arangosh",
        "--server.endpoint", SERVER_ENDPOINT,
        "--server.database", DATABASE_NAME,
        "--server.username", USERNAME,
        "--server.password", PASSWORD,
        "--javascript.execute", "/tmp/populate_movie_db.js"
    ]
    
    result = container.exec_run(
        exec_command,
        tty=True,
        stream=True
    )
    
    # Stream the output
    for output in result.output:
        print(output.decode().strip())

    print(f"JavaScript file executed successfully on {ARANGO_URL}")

if __name__ == "__main__":
    main()