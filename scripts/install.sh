#!/bin/bash

# Configuration
REPO_URL="https://github.com/yourusername/cursor-patterns"
RULES_DIR=".cursor/rules"
PATTERNS_DIR="$RULES_DIR/design_patterns"

# Create necessary directories
mkdir -p "$RULES_DIR"
mkdir -p "$PATTERNS_DIR"

# Clone or update the repository
if [ -d "cursor-patterns" ]; then
    echo "Updating cursor-patterns repository..."
    cd cursor-patterns
    git pull
    cd ..
else
    echo "Cloning cursor-patterns repository..."
    git clone "$REPO_URL"
fi

# Copy core rules
echo "Installing core rules..."
cp cursor-patterns/rules/core/*.mdc "$RULES_DIR/"

# Copy design patterns
echo "Installing design patterns..."
cp cursor-patterns/rules/design_patterns/*.mdc "$PATTERNS_DIR/"

# Copy project-specific patterns based on language
if [ -f "pyproject.toml" ]; then
    echo "Installing Python-specific patterns..."
    cp cursor-patterns/rules/project_specific/python/*.mdc "$RULES_DIR/"
elif [ -f "package.json" ]; then
    echo "Installing TypeScript/JavaScript patterns..."
    cp cursor-patterns/rules/project_specific/typescript/*.mdc "$RULES_DIR/"
elif [ -f "Cargo.toml" ]; then
    echo "Installing Rust patterns..."
    cp cursor-patterns/rules/project_specific/rust/*.mdc "$RULES_DIR/"
fi

echo "Installation complete!" 