# Development Setup

## Prerequisites

- Node.js ≥ 18.0.0
- Git
- npm

## Local Development

```bash
# Clone repository
git clone https://github.com/yamadashy/repomix.git
cd repomix

# Install dependencies
npm install

# Run CLI
npm run repomix
```

## Docker Development

```bash
# Build image
docker build -t repomix .

# Run container
docker run -v ./:/app -it --rm repomix
```

## Project Structure

```
src/
├── cli/          # CLI implementation
├── config/       # Configuration handling
├── core/         # Core functionality
└── shared/       # Shared utilities
```

## Testing

```bash
# Run tests
npm run test

# Test coverage
npm run test-coverage

# Linting
npm run lint-biome
npm run lint-ts
npm run lint-secretlint
```

## Release Process

1. Update version
```bash
npm version patch  # or minor/major
```

2. Run tests and build
```bash
npm run test-coverage
npm run build
```

3. Publish
```bash
npm publish
```
