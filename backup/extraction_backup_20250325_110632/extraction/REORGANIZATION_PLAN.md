# Directory Reorganization Plan

This document outlines a plan to reduce fragmentation in the extraction module structure and improve organization.

## Current Issues

1. **Fragmented Documentation**: Documentation is scattered across multiple directories
2. **Inconsistent Directory Structure**: Different patterns for organizing files
3. **Duplicate Files**: Same concepts documented in multiple places
4. **Unclear Integration Points**: The relationship between fetch_docs and extraction is not clearly defined
5. **Test Organization**: Test files are spread across many directories with redundant tests

## Reorganization Plan

### Phase 1: Consolidate Documentation

1. Create a central docs directory with clear categories:
   ```
   src/agent_tools/dualipa/extraction/docs/
   ├── api/                    # API reference documentation
   ├── concepts/               # Concept explanations
   ├── examples/               # Example usage scenarios
   ├── integration/            # Integration points with other modules
   ├── validation/             # Validation framework documentation
   ├── guides/                 # How-to guides
   └── testing/                # Testing approach and strategies
   ```

2. Move all documentation files to appropriate categories:
   - Move doc files from `examples/end_to_end/` to proper location
   - Move concept files from fetch_docs to integration directory
   - Consolidate duplicate documents

### Phase 2: Reorganize Examples

1. Create a standard examples structure:
   ```
   src/agent_tools/dualipa/extraction/examples/
   ├── basic/                 # Basic extraction examples
   ├── end_to_end/            # Complete workflow examples
   │   ├── code/              # Code extraction examples
   │   ├── markdown/          # Markdown extraction examples
   │   └── html/              # HTML extraction examples
   ├── validation/            # Validation examples
   └── integration/           # Integration examples with other systems
   ```

2. Organize existing examples into the new structure:
   - Clean up the end_to_end directory
   - Move example files to appropriate directories
   - Verify all examples still work after reorganization

### Phase 3: Clarify Test Structure

1. Create a clearer test organization:
   ```
   tests/dualipa/extraction/
   ├── unit/                  # Unit tests for individual components
   │   ├── extractors/        # Tests for extractors
   │   ├── utils/             # Tests for utilities
   │   └── validation/        # Tests for validation components
   ├── integration/           # Tests that verify component integration
   ├── end_to_end/            # Full workflow tests
   └── fixtures/              # Test fixtures and data
   ```

2. Move all tests to the appropriate directories:
   - Consolidate duplicate tests
   - Ensure test naming is consistent
   - Update imports in test files

### Phase 4: Improve Integration Points

1. Create clear integration module between fetch_docs and extraction:
   ```
   src/agent_tools/dualipa/extraction/integration/
   ├── __init__.py
   ├── fetch_docs.py           # Integration with fetch_docs module
   ├── qa_system.py            # Integration with QA system
   └── validators.py           # Cross-module validators
   ```

2. Document integration points:
   - Create clear API boundaries
   - Document expected data formats
   - Provide example workflows

### Phase 5: Clean Up Unnecessary Files

1. Identify and remove:
   - Duplicate test and example files
   - Old/obsolete documentation 
   - Temporary files and backups
   - Debug files not needed in production

## Implementation Approach

1. **Create Script**: Write a reorganization script to move files
2. **Update References**: Update import statements and file references
3. **Test Verification**: Run tests after each phase to ensure functionality
4. **Documentation**: Update documentation to reflect new structure
5. **Pull Request**: Create PR with detailed description of changes

## Success Criteria

1. All tests continue to pass after reorganization
2. Documentation is consolidated and easier to find
3. Clear integration points between modules
4. Reduced duplication across directories
5. Consistent directory structure throughout the project