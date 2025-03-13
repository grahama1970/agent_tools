#!/bin/bash

# Script to update import statements in Python files
# This should be run from the project root directory

set -e

echo "Starting import statement updates..."
cd $(dirname $0)/..

# Ensure we're in the project root
if [ ! -d "src/agent_tools/cursor_rules" ]; then
  echo "Error: src/agent_tools/cursor_rules directory not found. Please run this script from the project root."
  exit 1
fi

# Create a log file for changes
LOGFILE="cursor_rules_import_updates_$(date +%Y%m%d_%H%M%S).log"
touch $LOGFILE

# Function to update imports in a file
update_imports() {
  local file=$1
  echo "Updating imports in $file" | tee -a $LOGFILE
  
  # Create backup of original file
  cp "$file" "${file}.bak"
  
  # Direct local imports to use absolute imports with proper module paths
  
  # ai_knowledge_db.py -> core/ai_knowledge_db.py
  sed -i 's/from ai_knowledge_db import/from agent_tools.cursor_rules.core.ai_knowledge_db import/g' "$file"
  sed -i 's/import ai_knowledge_db/import agent_tools.cursor_rules.core.ai_knowledge_db/g' "$file"
  
  # cursor_rules.py -> core/cursor_rules.py
  sed -i 's/from cursor_rules import/from agent_tools.cursor_rules.core.cursor_rules import/g' "$file"
  sed -i 's/import cursor_rules/import agent_tools.cursor_rules.core.cursor_rules/g' "$file"
  
  # db.py -> core/db.py
  sed -i 's/from db import/from agent_tools.cursor_rules.core.db import/g' "$file"
  sed -i 's/import db/import agent_tools.cursor_rules.core.db/g' "$file"
  
  # enhanced_db.py -> core/enhanced_db.py
  sed -i 's/from enhanced_db import/from agent_tools.cursor_rules.core.enhanced_db import/g' "$file"
  sed -i 's/import enhanced_db/import agent_tools.cursor_rules.core.enhanced_db/g' "$file"
  
  # common_queries.py -> scenarios/common_queries.py
  sed -i 's/from common_queries import/from agent_tools.cursor_rules.scenarios.common_queries import/g' "$file"
  sed -i 's/import common_queries/import agent_tools.cursor_rules.scenarios.common_queries/g' "$file"
  
  # scenario_management.py -> scenarios/scenario_management.py
  sed -i 's/from scenario_management import/from agent_tools.cursor_rules.scenarios.scenario_management import/g' "$file"
  sed -i 's/import scenario_management/import agent_tools.cursor_rules.scenarios.scenario_management/g' "$file"
  
  # schema paths
  sed -i 's/"ai_knowledge_schema.json"/"schemas\/ai_knowledge_schema.json"/g' "$file"
  sed -i 's/"db_schema.json"/"schemas\/db_schema.json"/g' "$file"
  
  # Update references to util modules
  sed -i 's/from utils\./from agent_tools.cursor_rules.utils./g' "$file"
  sed -i 's/import utils\./import agent_tools.cursor_rules.utils./g' "$file"
  
  # Update references to views modules
  sed -i 's/from views\./from agent_tools.cursor_rules.views./g' "$file"
  sed -i 's/import views\./import agent_tools.cursor_rules.views./g' "$file"
  
  # Update references to CLI modules
  sed -i 's/from cli\./from agent_tools.cursor_rules.cli./g' "$file"
  sed -i 's/import cli\./import agent_tools.cursor_rules.cli./g' "$file"
  
  # Log differences
  echo "Changes made to $file:" >> $LOGFILE
  diff "${file}.bak" "$file" >> $LOGFILE 2>&1 || true
  echo "-----------------------------------" >> $LOGFILE
}

# Find all Python files in the cursor_rules directory and update imports
find src/agent_tools/cursor_rules -name "*.py" -not -path "*/\.*" | while read file; do
  update_imports "$file"
done

echo "Import statement updates completed!"
echo "Check $LOGFILE for details of changes made."
echo ""
echo "Next steps:"
echo "1. Test the application to ensure imports are working correctly"
echo "2. Review changes in the log file to verify they're accurate"
echo "3. Remove backup files (*.bak) once everything is verified" 