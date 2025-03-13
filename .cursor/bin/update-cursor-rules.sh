#!/bin/bash

# Exit on any error
set -e

# Configuration
REPO_URL="https://github.com/grahama1970/snippets"
TEMP_DIR="/tmp/cursor_rules_temp_$$"  # Using $$ (PID) to make unique
CURSOR_DIR=".cursor"
DRY_RUN=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN: No changes will be made"
fi

# Check if .cursor already exists
if [ -d "$CURSOR_DIR" ]; then
    echo "🔄 Existing .cursor directory found"
    BACKUP_DIR="${CURSOR_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "📦 Would backup to: $BACKUP_DIR"
    if [ "$DRY_RUN" = false ]; then
        mv "$CURSOR_DIR" "$BACKUP_DIR"
    fi
fi

# Create and enter temp directory
echo "📁 Creating temporary directory"
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
fi

# Clone repository
echo "⬇️  Would clone: $REPO_URL"
if [ "$DRY_RUN" = false ]; then
    git clone "$REPO_URL" .

    # Copy .cursor directory
    echo "📋 Copying .cursor directory"
    if [ ! -d ".cursor" ]; then
        echo "❌ Error: .cursor directory not found in cloned repository"
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    # Return to original directory and copy
    cd - > /dev/null
    cp -r "$TEMP_DIR/.cursor" .

    # Cleanup
    echo "🧹 Cleaning up temporary files"
    rm -rf "$TEMP_DIR"

    echo "✅ Cursor rules updated successfully!"
    if [ -d "$BACKUP_DIR" ]; then
        echo "ℹ️  Previous rules backed up to: $BACKUP_DIR"
    fi
else
    echo "✨ DRY RUN Summary:"
    echo "  • Would create temp directory: $TEMP_DIR"
    echo "  • Would clone repository"
    echo "  • Would copy .cursor directory to current location"
    echo "  • Would clean up temp directory"
fi 