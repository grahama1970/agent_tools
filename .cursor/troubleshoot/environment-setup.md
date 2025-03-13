# Environment Setup for Testing

## Required Packages
- pytest
- pytest-asyncio
- pytest-order
- pytest-depends
- pytest-mock
- pytest-xdist

## Required Environment Variables
- ARANGO_ROOT_PASSWORD
- LITELLM_LOG
- HF_HUB_ENABLE_HF_TRANSFER (set to 0 if hf_transfer not installed)

## Model Dependencies
- hf_transfer (for Hugging Face model downloads)
- einops (for Nomic models)
- torch (for embedding models)

## Database Setup
- ArangoDB must be running
- Test database must be created
- Collections must be initialized

## Quick Setup Commands

```bash
# Install all dependencies
uv pip install -e ".[dev]"

# Install additional model dependencies
uv pip install hf_transfer einops

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0  # Disable if hf_transfer not installed
```

## Dependency Troubleshooting

### Missing Dependencies
If tests fail with import errors:

```bash
# Check if package is installed
uv pip list | grep package_name

# Install missing package
uv pip install package_name

# Update pyproject.toml to include the dependency
# Add to [project.dependencies] or [project.optional-dependencies.dev]
```

### Environment Variable Issues
If tests fail due to missing environment variables:

```bash
# Check current environment variables
env | grep VARIABLE_NAME

# Set environment variable for current session
export VARIABLE_NAME=value

# Add to .env file for persistence
echo "VARIABLE_NAME=value" >> .env
```

### Virtual Environment Issues
If the virtual environment is not working correctly:

```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Reinstall dependencies
uv pip install -e ".[dev]"
```

## Common Error Messages and Solutions

### "No module named 'hf_transfer'"
```bash
# Install hf_transfer
uv pip install hf_transfer

# Or disable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### "No module named 'einops'"
```bash
# Install einops
uv pip install einops
```

### "This modeling file requires the following packages that were not found in your environment"
```bash
# Install the missing package
uv pip install package_name
```

### "fixture 'test_db' not found"
```bash
# Check if ArangoDB is running
docker ps | grep arango

# Start ArangoDB if needed
docker start arango
``` 