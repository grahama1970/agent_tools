#!/bin/bash

# Script to reorganize cursor_rules directory structure
# This should be run from the project root directory

set -e

echo "Starting cursor_rules directory reorganization..."
cd $(dirname $0)/..

# Ensure we're in the project root
if [ ! -d "src/agent_tools/cursor_rules" ]; then
  echo "Error: src/agent_tools/cursor_rules directory not found. Please run this script from the project root."
  exit 1
fi

cd src/agent_tools/cursor_rules

# Create backup
echo "Creating backup..."
BACKUP_DIR="../../../cursor_rules_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r . $BACKUP_DIR

# Ensure all directories exist
mkdir -p core schemas cli/commands utils/helpers utils/ai utils/text views scenarios docs scripts

# Move schema files to schemas directory
echo "Moving schema files..."
if [ -f ai_knowledge_schema.json ]; then
  mv ai_knowledge_schema.json schemas/
fi
if [ -f db_schema.json ]; then
  mv db_schema.json schemas/
fi

# Move core files
echo "Moving core files..."
if [ -f cursor_rules.py ] && [ ! -f core/cursor_rules.py ]; then
  cp cursor_rules.py core/
fi
if [ -f db.py ] && [ ! -f core/db.py ]; then
  cp db.py core/
fi

# Move view helper files
echo "Moving view helper files..."
if [ -d view_helpers ]; then
  cp view_helpers/* views/ 2>/dev/null || true
fi

# Move scenario files
echo "Moving scenario files..."
if [ -f sample_scenarios.json ] && [ ! -f scenarios/sample_scenarios.json ]; then
  cp sample_scenarios.json scenarios/
fi
if [ -f scenario_management.py ] && [ ! -f scenarios/scenario_management.py ]; then
  cp scenario_management.py scenarios/
fi
if [ -f common_queries.py ] && [ ! -f scenarios/common_queries.py ]; then
  cp common_queries.py scenarios/
fi

# Move documentation
echo "Moving documentation..."
if [ -f retrieval_scenarios.md ]; then
  mv retrieval_scenarios.md docs/
fi
if [ -f task.md ]; then
  mv task.md docs/
fi

# Move scripts
echo "Moving scripts..."
if [ -f demo.py ] && [ ! -f scripts/demo.py ]; then
  cp demo.py scripts/
fi
if [ -f cleanup_databases.py ] && [ ! -f scripts/cleanup_databases.py ]; then
  cp cleanup_databases.py scripts/
fi

# Create init files
for dir in core schemas cli cli/commands utils utils/helpers utils/ai utils/text views scenarios; do
  if [ ! -f "$dir/__init__.py" ]; then
    echo "Creating $dir/__init__.py"
    touch "$dir/__init__.py"
  fi
done

echo "Directory reorganization completed!"
echo "Backup created at $BACKUP_DIR"
echo ""
echo "IMPORTANT: This script has copied files to their new locations but hasn't deleted the originals."
echo "After verifying everything works correctly, you'll need to manually remove duplicate files."
echo ""
echo "Next steps:"
echo "1. Update import statements in all files to use the new directory structure"
echo "2. Test the application to ensure everything works correctly"
echo "3. Remove duplicate files once verified" 