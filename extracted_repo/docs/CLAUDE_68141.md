# Repomix Development Guide

## Build & Development Commands
- `npm run build` - Build the project
- `npm run lint` - Run all linters (Biome, TypeScript, secretlint)
- `npm run lint-biome` - Run Biome linter with auto-fix
- `npm run lint-ts` - Run TypeScript type checking
- `npm test` - Run all tests
- `npm test -- /path/to/file.test.ts` - Run a specific test file
- `npm test -- -t "test name"` - Run tests matching description
- `npm run test-coverage` - Run tests with coverage report
- `npm run repomix` - Build and run the CLI tool

## Code Style Guidelines
- **Formatting**: 2 spaces indentation, 120 char line width, single quotes, trailing commas
- **Imports**: Use `node:` prefix for built-ins, `.js` extension for relative imports, `import type` for types
- **TypeScript**: Strict type checking, explicit return types, prefer interfaces for object types
- **Error Handling**: Custom errors extend `RepomixError`, descriptive messages, proper try/catch
- **Dependencies**: Inject dependencies through deps object parameter for testability
- **File Structure**: Keep files under 250 lines, organize by feature, use workers for concurrent operations
- **Testing**: Use Vitest, mock dependencies, descriptive test names, arrange/act/assert pattern

## Naming Conventions
- PascalCase: Classes, interfaces, types
- camelCase: Variables, functions, methods, file names
- Test files: `[filename].test.ts`

## Git Workflow
- Write detailed commit messages focusing on the "why" rather than the "what"
- Run `npm run lint && npm test` before committing changes