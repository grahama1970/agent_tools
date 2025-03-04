#!/bin/bash

# Configuration
RULES_DIR=".cursor/rules"
BACKUP_DIR="$RULES_DIR/backup_$(date +%Y%m%d_%H%M%S)"

# Create backup of current rules
echo "Creating backup of current rules..."
mkdir -p "$BACKUP_DIR"
cp -r "$RULES_DIR"/*.mdc "$BACKUP_DIR/"
cp -r "$RULES_DIR/design_patterns" "$BACKUP_DIR/"

# Update repository
echo "Updating cursor-patterns repository..."
cd cursor-patterns
git pull
cd ..

# Run installation script
echo "Running installation script..."
./scripts/install.sh

echo "Update complete! Backup stored in $BACKUP_DIR" 