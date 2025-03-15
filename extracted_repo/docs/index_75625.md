---
layout: home
title: Repomix
titleTemplate: Pack your codebase into AI-friendly formats
aside: false
editLink: false

features:
  - icon: 🤖
    title: AI-Optimized
    details: Formats your codebase in a way that's easy for AI to understand and process.

  - icon: ⚙️
    title: Git-Aware
    details: Automatically respects your .gitignore files.

  - icon: 🛡️
    title: Security-Focused
    details: Incorporates Secretlint for robust security checks to detect and prevent inclusion of sensitive information.

  - icon: 📊
    title: Token Counting
    details: Provides token counts for each file and the entire repository, useful for LLM context limits.

---

<div class="cli-section">


## Quick Start

Once you've generated a packed file (`repomix-output.txt`) using Repomix, you can send it to an AI assistant with a prompt like:

```
This file contains all the files in the repository combined into one.
I want to refactor the code, so please review it first.
```

The AI will analyze your entire codebase and provide comprehensive insights:

![Repomix File Usage 1](/images/docs/repomix-file-usage-1.png)

When discussing specific changes, the AI can help generate code. With features like Claude's Artifacts, you can even receive multiple interdependent files:

![Repomix File Usage 2](/images/docs/repomix-file-usage-2.png)

Happy coding! 🚀



## Power User Guide

For advanced users who need more control, Repomix offers extensive customization options through its CLI interface.

### Quick Start

You can try Repomix instantly in your project directory without installation:

```bash
npx repomix
```

Or install globally for repeated use:

```bash
# Install using npm
npm install -g repomix

# Alternatively using yarn
yarn global add repomix

# Alternatively using Homebrew (macOS/Linux)
brew install repomix

# Then run in any project directory
repomix
```

That's it! Repomix will generate a `repomix-output.txt` file in your current directory, containing your entire repository in an AI-friendly format.



### Usage

To pack your entire repository:

```bash
repomix
```

To pack a specific directory:

```bash
repomix path/to/directory
```

To pack specific files or directories using [glob patterns](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax):

```bash
repomix --include "src/**/*.ts,**/*.md"
```

To exclude specific files or directories:

```bash
repomix --ignore "**/*.log,tmp/"
```

To pack a remote repository:
```bash
# Using shorthand format
npx repomix --remote yamadashy/repomix

# Using full URL (supports branches and specific paths)
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# Using commit's URL
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

To initialize a new configuration file (`repomix.config.json`):

```bash
repomix --init
```

Once you have generated the packed file, you can use it with Generative AI tools like Claude, ChatGPT, and Gemini.

#### Docker Usage

You can also run Repomix using Docker 🐳  
This is useful if you want to run Repomix in an isolated environment or prefer using containers.

Basic usage (current directory):

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

To pack a specific directory:
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

Process a remote repository and output to a `output` directory:

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### Output Formats

Choose your preferred output format:

```bash
# XML format (default)
repomix --style xml

# Markdown format
repomix --style markdown

# Plain text format
repomix --style plain
```

### Customization

Create a `repomix.config.json` for persistent settings:

```json
{
  "output": {
    "style": "markdown",
    "filePath": "custom-output.md",
    "removeComments": true,
    "showLineNumbers": true,
    "topFilesLength": 10
  },
  "ignore": {
    "customPatterns": ["*.test.ts", "docs/**"]
  }
}
```

### More Examples
::: tip
💡 Check out our [GitHub repository](https://github.com/yamadashy/repomix) for complete documentation and more examples!
:::

</div>
