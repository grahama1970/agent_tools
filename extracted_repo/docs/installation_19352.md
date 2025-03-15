# Installation

## Using npx (No Installation Required)

```bash
npx repomix
```

## Global Installation

### npm
```bash
npm install -g repomix
```

### Yarn
```bash
yarn global add repomix
```

### Homebrew (macOS/Linux)
```bash
brew install repomix
```

## Docker Installation

Pull and run the Docker image:

```bash
# Current directory
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix

# Specific directory
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory

# Remote repository
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote yamadashy/repomix
```

## VSCode Extension

Run Repomix directly in VSCode with the community-maintained [Repomix Runner](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner) extension.

Features:
- Pack any folder with just a few clicks
- Choose between file or content mode for copying
- Automatic cleanup of output files
- Works with repomix.config.json

Install it from the [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner).

## System Requirements

- Node.js: ≥ 18.0.0
- Git: Required for remote repository processing

## Verification

After installation, verify that Repomix is working:

```bash
repomix --version
repomix --help
```
