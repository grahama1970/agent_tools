#!/bin/bash

# Script to organize main files in cursor_rules directory
# This should be run from the project root directory

set -e

echo "Starting main files reorganization..."
cd $(dirname $0)/..

# Ensure we're in the project root
if [ ! -d "src/agent_tools/cursor_rules" ]; then
  echo "Error: src/agent_tools/cursor_rules directory not found. Please run this script from the project root."
  exit 1
fi

cd src/agent_tools/cursor_rules

# Create backup
echo "Creating backup..."
BACKUP_DIR="../../../cursor_rules_files_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r . $BACKUP_DIR

# Ensure directories exist
mkdir -p core schemas cli/commands utils/helpers utils/ai utils/text views scenarios docs scripts

# 1. Move schema files to schemas directory
echo "Moving schema files..."
if [ -f ai_knowledge_schema.json ]; then
  echo "Moving ai_knowledge_schema.json to schemas/"
  cp ai_knowledge_schema.json schemas/
fi
if [ -f db_schema.json ]; then
  echo "Moving db_schema.json to schemas/"
  cp db_schema.json schemas/
fi

# 2. Move core database and AI knowledge files
echo "Moving core files..."
if [ -f ai_knowledge_db.py ]; then
  echo "Moving ai_knowledge_db.py to core/"
  cp ai_knowledge_db.py core/
fi
if [ -f cursor_rules.py ]; then
  echo "Moving cursor_rules.py to core/"
  cp cursor_rules.py core/
fi
if [ -f db.py ]; then
  echo "Moving db.py to core/"
  cp db.py core/
fi
if [ -f enhanced_db.py ]; then
  echo "Moving enhanced_db.py to core/"
  cp enhanced_db.py core/
fi

# 3. Move scenario-related files
echo "Moving scenario files..."
if [ -f common_queries.py ]; then
  echo "Moving common_queries.py to scenarios/"
  cp common_queries.py scenarios/
fi
if [ -f scenario_management.py ]; then
  echo "Moving scenario_management.py to scenarios/"
  cp scenario_management.py scenarios/
fi
if [ -f sample_scenarios.json ]; then
  echo "Moving sample_scenarios.json to scenarios/"
  cp sample_scenarios.json scenarios/
fi

# 4. Move documentation
echo "Moving documentation..."
if [ -f retrieval_scenarios.md ]; then
  echo "Moving retrieval_scenarios.md to docs/"
  cp retrieval_scenarios.md docs/
fi
if [ -f task.md ]; then
  echo "Moving task.md to docs/"
  cp task.md docs/
fi

# 5. Move script files
echo "Moving script files..."
if [ -f demo.py ]; then
  echo "Moving demo.py to scripts/"
  cp demo.py scripts/
fi
if [ -f cleanup_databases.py ]; then
  echo "Moving cleanup_databases.py to scripts/"
  cp cleanup_databases.py scripts/
fi

# 6. Move view helper files
echo "Moving view helper files..."
if [ -d view_helpers ]; then
  echo "Moving files from view_helpers/ to views/"
  cp view_helpers/* views/ 2>/dev/null || true
fi

# 7. Create init files
for dir in core schemas cli cli/commands utils utils/helpers utils/ai utils/text views scenarios; do
  if [ ! -f "$dir/__init__.py" ]; then
    echo "Creating $dir/__init__.py"
    touch "$dir/__init__.py"
  fi
done

# 8. Update main __init__.py to expose core modules
cat > __init__.py << EOL
"""
Cursor Rules Package - AI-powered rule management and enforcement system.

This package provides tools for managing, searching, and enforcing coding
rules and patterns stored in ArangoDB with vector search capabilities.
"""

# Core functionality
from agent_tools.cursor_rules.core.cursor_rules import (
    CursorRules,
    CursorRulesDatabase,
    get_cursor_rules,
    setup_cursor_rules,
)
from agent_tools.cursor_rules.core.db import get_db

__all__ = [
    "CursorRules",
    "CursorRulesDatabase",
    "get_cursor_rules",
    "setup_cursor_rules",
    "get_db",
]
EOL

echo "Main files reorganization completed!"
echo "Backup created at $BACKUP_DIR"
echo ""
echo "IMPORTANT: This script has copied files to their new locations but hasn't deleted the originals."
echo "After verifying everything works correctly, you'll need to manually remove duplicate files."
echo ""
echo "Next steps:"
echo "1. Update import statements in all files to use the new directory structure"
echo "2. Test the application to ensure everything works correctly"
echo "3. Remove duplicate files once verified" 