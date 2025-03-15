# Repomix Website

This directory contains the source code for the Repomix website, built with [VitePress](https://vitepress.dev/) and [Vue.js](https://vuejs.org/)

## Prerequisites

- Docker must be installed on your system

## Development

To start the development server:

```bash
# Start the website development server
npm run website

# Access the website at http://localhost:5173/
```

## Documentation

When updating documentation, you only need to update the English version (`client/src/en/`).
The maintainers will handle translations to other languages.

## Building for Production

To build the website for production:

```bash
npm run website:build
```

The built files will be available in the `client/dist` directory. 
