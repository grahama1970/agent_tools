"""
Docker-based report server

This module provides utilities for serving HTML reports using Docker.
It's designed to work with both regular Docker and Docker Compose.
"""

import os
import shutil
import socket
import subprocess
from pathlib import Path
import logging
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8765
CONTAINER_NAME = "dualipa-test-reports"


def get_ip_address() -> str:
    """Get the primary IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
    except Exception:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except:
            ip = '127.0.0.1'  # Last resort fallback
    finally:
        s.close()
    return ip


def is_running_in_wsl() -> bool:
    """Check if we're running in a WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def serve_with_docker(results_dir: str or Path, port: int = DEFAULT_PORT, *, use_compose: bool = None) -> Tuple[bool, str]:
    """
    Serve the results directory using a Docker container with docker-compose if available.
    
    Args:
        results_dir: Directory to serve
        port: Port to use
        use_compose: Force use of Docker Compose (True) or Docker (False). If None, auto-detect.
        
    Returns:
        Tuple[bool, str]: (success, url) - success status and URL if successful
    """
    # Convert Path to string if needed
    results_dir = str(results_dir) if isinstance(results_dir, Path) else results_dir
    
    # Get absolute path
    results_dir = os.path.abspath(results_dir)
    
    # Check if directory exists
    if not os.path.isdir(results_dir):
        logger.error(f"Error: Directory {results_dir} does not exist")
        return False, ""
    
    # Get the module directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine whether to use Docker Compose or Docker
    if use_compose is None:
        # Auto-detect: check if docker-compose is available
        use_compose = shutil.which('docker-compose') is not None
    
    # Prepare environment variables for Docker Compose
    env = os.environ.copy()
    env['PORT'] = str(port)
    env['RESULTS_DIR'] = results_dir
    
    try:
        # Stop any existing container with the same name
        stop_docker_server()
        
        if use_compose:
            # Use Docker Compose
            logger.info(f"Starting Docker container with Docker Compose on port {port}")
            
            # Path to docker-compose.yml
            compose_file = os.path.join(module_dir, 'docker-compose.yml')
            
            # Start the container
            subprocess.run(
                ['docker-compose', '-f', compose_file, 'up', '-d'],
                check=True,
                text=True,
                env=env
            )
            
            logger.info(f"Docker Compose service started on port {port}")
        else:
            # Use regular Docker
            logger.info(f"Starting Docker container with docker on port {port}")
            
            # Start the container
            subprocess.run(
                ['docker', 'run', '--name', CONTAINER_NAME, '-d',
                 '-p', f"{port}:80",
                 '-v', f"{results_dir}:/usr/share/nginx/html:ro",
                 '-e', 'NGINX_PORT=80',
                 'nginx:alpine'],
                check=True,
                text=True
            )
            
            logger.info(f"Docker container started on port {port}")
        
        # Get the URL
        local_ip = get_ip_address()
        url = f"http://{local_ip}:{port}/"
        
        # Print success message
        print(f"\n==========================================================", flush=True)
        print(f"✅ REPORT SERVER STARTED SUCCESSFULLY", flush=True)
        print(f"📁 Serving directory: {results_dir}", flush=True)
        print(f"\n🌐 ACCESS THE REPORTS USING THESE URLS:", flush=True)
        print(f"  • Local access:  http://localhost:{port}/", flush=True)
        print(f"  • Network:       {url}", flush=True)
        
        # WSL2-specific message
        if is_running_in_wsl():
            print(f"\n🪟 WINDOWS WSL ACCESS:", flush=True)
            print(f"  You can access the reports from your Windows browser using the URLs above.", flush=True)
            print(f"  Docker Desktop automatically forwards ports to your Windows host.", flush=True)
        
        # Command examples
        print(f"\n📋 COMMANDS:", flush=True)
        if use_compose:
            print(f"  • View logs:    docker-compose -f {compose_file} logs", flush=True)
            print(f"  • Stop server:  docker-compose -f {compose_file} down", flush=True)
        else:
            print(f"  • View logs:    docker logs {CONTAINER_NAME}", flush=True)
            print(f"  • Stop server:  docker stop {CONTAINER_NAME}", flush=True)
        print(f"==========================================================", flush=True)
        
        return True, url
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Docker container: {e}")
        return False, ""
    except Exception as e:
        logger.error(f"Unexpected error starting Docker container: {e}")
        return False, ""


def stop_docker_server() -> bool:
    """
    Stop the running Docker server if it exists.
    
    Returns:
        bool: True if the server was stopped or didn't exist, False if an error occurred
    """
    try:
        # First try to stop with Docker Compose
        module_dir = os.path.dirname(os.path.abspath(__file__))
        compose_file = os.path.join(module_dir, 'docker-compose.yml')
        
        if os.path.exists(compose_file) and shutil.which('docker-compose'):
            try:
                logger.info("Attempting to stop Docker container with docker-compose")
                subprocess.run(
                    ['docker-compose', '-f', compose_file, 'down'],
                    check=False,  # Don't fail if containers don't exist
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                logger.warning(f"Could not stop with Docker Compose: {e}")
        
        # Also try to stop using regular Docker (as a fallback)
        try:
            # Check if the container exists
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name={CONTAINER_NAME}', '--format', '{{.Names}}'],
                check=True,
                text=True,
                stdout=subprocess.PIPE
            )
            
            if CONTAINER_NAME in result.stdout:
                logger.info(f"Stopping Docker container: {CONTAINER_NAME}")
                subprocess.run(['docker', 'stop', CONTAINER_NAME], check=True, stdout=subprocess.PIPE)
                subprocess.run(['docker', 'rm', CONTAINER_NAME], check=True, stdout=subprocess.PIPE)
        except Exception as e:
            logger.warning(f"Could not stop with regular Docker: {e}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error stopping Docker server: {e}")
        return False


def is_docker_server_running() -> bool:
    """
    Check if the Docker report server is running.
    
    Returns:
        bool: True if running, False otherwise
    """
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={CONTAINER_NAME}', '--format', '{{.Names}}'],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return CONTAINER_NAME in result.stdout
    except:
        return False