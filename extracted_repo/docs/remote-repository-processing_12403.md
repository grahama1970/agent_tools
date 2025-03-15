# Remote Repository Processing

## Basic Usage

Process public repositories:
```bash
# Using full URL
repomix --remote https://github.com/user/repo

# Using GitHub shorthand
repomix --remote user/repo
```

## Branch and Commit Selection

```bash
# Specific branch
repomix --remote user/repo --remote-branch main

# Tag
repomix --remote user/repo --remote-branch v1.0.0

# Commit hash
repomix --remote user/repo --remote-branch 935b695
```

## Requirements

- Git must be installed
- Internet connection
- Read access to repository

## Output Control

```bash
# Custom output location
repomix --remote user/repo -o custom-output.xml

# With XML format
repomix --remote user/repo --style xml

# Remove comments
repomix --remote user/repo --remove-comments
```

## Docker Usage

```bash
# Process and output to current directory
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix \
  --remote user/repo

# Output to specific directory
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix \
  --remote user/repo
```

## Common Issues

### Access Issues
- Ensure repository is public
- Check Git installation
- Verify internet connection

### Large Repositories
- Use `--include` to select specific paths
- Enable `--remove-comments`
- Process branches separately
