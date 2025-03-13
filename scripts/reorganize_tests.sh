#!/bin/bash

# Script to reorganize tests directory to mirror main code structure
# This should be run from the project root directory

set -e

echo "Starting tests directory reorganization..."
cd $(dirname $0)/..

# Ensure we're in the project root
if [ ! -d "src/agent_tools/cursor_rules/tests" ]; then
  echo "Error: src/agent_tools/cursor_rules/tests directory not found. Please run this script from the project root."
  exit 1
fi

cd src/agent_tools/cursor_rules/tests

# Create backup
echo "Creating backup..."
BACKUP_DIR="../../../../cursor_rules_tests_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r . $BACKUP_DIR

# Ensure all test directories exist to mirror main code structure
mkdir -p unit/core unit/schemas unit/cli unit/utils unit/views unit/scenarios
mkdir -p integration/core integration/schemas integration/cli integration/utils integration/views integration/scenarios
mkdir -p end_to_end

# Move core-related tests
echo "Organizing core-related tests..."
find . -name "test_*cursor_rules*.py" -not -path "*/deprecated/*" | while read file; do
  if [[ $file != ./unit/core/* && $file != ./integration/core/* ]]; then
    if [[ $file == *"/test_"* ]]; then
      # Determine destination based on test type
      if [[ $file == *"integration"* ]]; then
        cp "$file" "integration/core/"
      else
        cp "$file" "unit/core/"
      fi
    fi
  fi
done

# Move database-related tests
echo "Organizing database-related tests..."
find . -name "test_*db*.py" -name "test_*arango*.py" -not -path "*/deprecated/*" | while read file; do
  if [[ $file != ./unit/core/* && $file != ./integration/core/* ]]; then
    if [[ $file == *"/test_"* ]]; then
      # Determine destination based on test type
      if [[ $file == *"integration"* ]]; then
        cp "$file" "integration/core/"
      else
        cp "$file" "unit/core/"
      fi
    fi
  fi
done

# Move CLI-related tests
echo "Organizing CLI-related tests..."
find . -name "test_*cli*.py" -not -path "*/deprecated/*" | while read file; do
  if [[ $file != ./unit/cli/* && $file != ./integration/cli/* ]]; then
    if [[ $file == *"/test_"* ]]; then
      # Determine destination based on test type
      if [[ $file == *"integration"* ]]; then
        cp "$file" "integration/cli/"
      else
        cp "$file" "unit/cli/"
      fi
    fi
  fi
done

# Move scenario-related tests
echo "Organizing scenario-related tests..."
find . -name "test_*scenario*.py" -name "test_*queries*.py" -not -path "*/deprecated/*" | while read file; do
  if [[ $file != ./unit/scenarios/* && $file != ./integration/scenarios/* ]]; then
    if [[ $file == *"/test_"* ]]; then
      # Determine destination based on test type
      if [[ $file == *"integration"* ]]; then
        cp "$file" "integration/scenarios/"
      else
        cp "$file" "unit/scenarios/"
      fi
    fi
  fi
done

# Move view-related tests
echo "Organizing view-related tests..."
find . -name "test_*view*.py" -not -path "*/deprecated/*" | while read file; do
  if [[ $file != ./unit/views/* && $file != ./integration/views/* ]]; then
    if [[ $file == *"/test_"* ]]; then
      # Determine destination based on test type
      if [[ $file == *"integration"* ]]; then
        cp "$file" "integration/views/"
      else
        cp "$file" "unit/views/"
      fi
    fi
  fi
done

# Move end-to-end tests
echo "Organizing end-to-end tests..."
find . -name "test_*.py" -path "*/end_to_end/*" | while read file; do
  if [[ $file != ./end_to_end/* ]]; then
    cp "$file" "end_to_end/"
  fi
done

# Create init files
echo "Creating __init__.py files..."
find . -type d -not -path "*/__pycache__*" | while read dir; do
  if [ ! -f "$dir/__init__.py" ]; then
    echo "Creating $dir/__init__.py"
    touch "$dir/__init__.py"
  fi
done

echo "Tests directory reorganization completed!"
echo "Backup created at $BACKUP_DIR"
echo ""
echo "IMPORTANT: This script has copied files to their new locations but hasn't deleted the originals."
echo "After verifying everything works correctly, you'll need to manually remove duplicate files."
echo ""
echo "Next steps:"
echo "1. Update import statements in all test files to use the new directory structure"
echo "2. Run the tests to ensure everything works correctly"
echo "3. Remove duplicate files once verified" 