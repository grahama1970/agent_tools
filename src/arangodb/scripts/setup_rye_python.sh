#!/bin/bash

# Define the Python version to use
PYTHON_VERSION="cpython@3.10.16"

# Function to install curl if not present
install_curl() {
    echo "Installing curl..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y curl
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install curl
    else
        echo "Unsupported OS for curl installation."
        exit 1
    fi
}

# Function to install Rust if not present
install_rust() {
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env  # Update PATH for the current session
}

# Check for curl and install if not found
if ! command -v curl &> /dev/null; then
    install_curl
fi

# Check for Rust and install if not found
if ! command -v rustc &> /dev/null; then
    install_rust
fi

# Install Rye if not already installed
if ! command -v rye &> /dev/null; then
    echo "Installing Rye..."
    curl -sSf https://rye.astral.sh/get | bash
fi

# Add Rye to PATH (if not already added)
export PATH="$HOME/.rye/shims:$PATH"

# Verify Rye installation
echo "Rye version: $(rye --version)"

# Fetch and pin the specified Python version
echo "Fetching and pinning $PYTHON_VERSION..."
rye toolchain fetch $PYTHON_VERSION
rye pin $PYTHON_VERSION

# Get the full path to the pinned Python version
PYTHON_PATH=$(rye toolchain list | grep "$PYTHON_VERSION" | awk '{print $2}')
if [ -z "$PYTHON_PATH" ]; then
    echo "Failed to find Python path for $PYTHON_VERSION."
    exit 1
fi
echo "Using Python path: $PYTHON_PATH"

# Create a virtual environment using uv with the pinned Python version
echo "Creating a virtual environment with $PYTHON_VERSION..."
uv venv --python "$PYTHON_PATH" .venv

# Activate the virtual environment
source .venv/bin/activate

# Install requirements from requirements.txt if it exists
if [ -f requirements.txt ]; then
    echo "Installing requirements from requirements.txt..."
    uv pip install -r requirements.txt
else
    echo "No requirements.txt file found."
fi

echo "Setup complete: Using $PYTHON_VERSION in the virtual environment."