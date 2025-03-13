# Cursor Rules Update Script

This script helps manage and update Cursor AI rules in your projects by syncing with the central rules repository.

## Quick Start

```bash
# Download the script
curl -o update-cursor-rules.sh https://raw.githubusercontent.com/grahama1970/snippets/master/bin/update-cursor-rules.sh

# Make it executable
chmod +x update-cursor-rules.sh

# Run it (this will download and install the cursor rules)
./update-cursor-rules.sh
```

## Features

- 🔄 Automatic backup of existing `.cursor` directory
- 🔍 Dry-run option to preview changes
- 🧹 Automatic cleanup of temporary files
- ⚡ Simple one-command update process

## Usage

### Basic Update

```bash
./update-cursor-rules.sh
```

This will:
1. Backup any existing `.cursor` directory (if present)
2. Download the latest rules
3. Install them in your project root

### Preview Changes (Dry Run)

```bash
./update-cursor-rules.sh --dry-run
```

This will show what would happen without making any changes.

## Output Directory Structure

After running the script, you'll have:

```
your-project/
├── .cursor/          # The installed cursor rules
│   └── rules/       # Rule definitions
└── .cursor_backup_* # Backup of previous rules (if existed)
```

## Error Handling

The script will:
- Create backups automatically
- Clean up temporary files
- Exit with helpful error messages if something goes wrong

## Updating the Script

To get the latest version of this script:

```bash
curl -o update-cursor-rules.sh https://raw.githubusercontent.com/grahama1970/snippets/master/bin/update-cursor-rules.sh
chmod +x update-cursor-rules.sh
```

## Contributing

The script and cursor rules are maintained in the [snippets repository](https://github.com/grahama1970/snippets).
To contribute:
1. Fork the repository
2. Make your changes
3. Submit a pull request 