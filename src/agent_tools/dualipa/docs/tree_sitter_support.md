# Tree-sitter Language Support

This document summarizes the tree-sitter language support implementation for multi-language code extraction in the DuaLipa pipeline.

## Fully Supported Languages

The following languages have been fully tested and are supported with tree-sitter for syntax-aware code extraction:

| Language    | Package                | Status             | Node Types Supported                                  |
|-------------|------------------------|--------------------|----------------------------------------------------|
| JavaScript  | tree-sitter-javascript | ✅ Fully functional | Functions, Classes, Methods                         |
| TypeScript  | tree-sitter-typescript | ✅ Fully functional | Functions, Classes, Interfaces, Methods, Types      |
| Python      | tree-sitter-python     | ✅ Fully functional | Functions, Classes, Methods                         |
| Go          | tree-sitter-go         | ✅ Fully functional | Functions, Structs, Methods                         |
| Rust        | tree-sitter-rust       | ✅ Fully functional | Functions, Structs, Impl blocks                     |
| C++         | tree-sitter-cpp        | ✅ Fully functional | Functions, Classes, Methods                         |
| Java        | tree-sitter-java       | ✅ Fully functional | Classes, Methods                                    |
| Ruby        | tree-sitter-ruby       | ✅ Fully functional | Methods, Classes                                    |
| Bash        | tree-sitter-bash       | ✅ Fully functional | Functions                                           |

## Languages with Issues

The following languages have been implemented but have known issues:

| Language    | Package                | Status                | Issue                                             | Fallback                    |
|-------------|------------------------|----------------------|---------------------------------------------------|----------------------------|
| C           | tree-sitter-c          | ⚠️ Version incompatible | Grammar version 15, tree-sitter expects 13-14     | Regex-based extraction      |
| PHP         | tree-sitter-php        | ⚠️ Missing attribute    | Language attribute not found                       | None implemented yet        |

## Implementation Details

### Node Types by Language

Each language extracts different types of syntax nodes based on the language's structure:

- **JavaScript/TypeScript**:
  - Functions: `function_declaration`, `function`, `arrow_function`
  - Classes: `class_declaration`
  - Interfaces (TS): `interface_declaration`
  - Type Aliases (TS): `type_alias_declaration`
  - Variable Declarations: `variable_declaration` (with function assignments)
  - Method Handling:
    - Regular methods: Counted in stats
    - Constructors: Counted in class stats, excluded from interface stats
    - Private/Protected methods: Detected via modifiers
    - Static methods: Detected via modifiers
    - Async methods: Detected via modifiers

- **Python**:
  - Functions: `function_definition`
  - Classes: `class_definition`
  - Decorators: `decorator`

- **Go**:
  - Functions: `function_declaration`
  - Structs: `type_declaration` > `type_spec`
  - Methods: Found through function declarations with receivers

- **Rust**:
  - Functions: `function_item`
  - Structs: `struct_item`
  - Impls: `impl_item`

- **C++**:
  - Functions: `function_definition`
  - Classes: `class_specifier`

- **Java**:
  - Classes: `class_declaration`
  - Methods: `method_declaration`

- **Ruby**:
  - Methods: `method`
  - Classes: `class`

- **Bash**:
  - Functions: `function_definition`

### Integration with Code Extraction

To use tree-sitter for code extraction:

1. Import the appropriate tree-sitter language module
2. Create a Parser instance
3. Load the language using `get_language` from `tree_sitter_languages`
4. Parse the source code
5. Traverse the syntax tree to find declarations
6. Extract each declaration with metadata (name, type, line range)

### Method Counting Rules

When counting methods in TypeScript:

1. For interfaces:
   - Count all method declarations
   - Exclude constructors from method count
   - Include getter/setter methods
   - Include async methods

2. For classes:
   - Count all methods including constructors
   - Count private and protected methods
   - Count static methods
   - Count async methods
   - Include getter/setter methods

### Fallback Mechanisms

For languages with issues or when tree-sitter is not available:

1. **C Language**: Uses regex-based extraction with patterns for functions and structs
2. **Unsupported Languages**: Falls back to general newline-based chunking
3. **PHP**: Currently not supported (needs implementation)

## Future Work

- Implement proper support for PHP by fixing the language attribute issue
- Resolve C grammar version incompatibility
- Add support for more languages (Kotlin, Swift, etc.)
- Enhance extraction with more detailed node traversal for nested structures
- Create a more robust fallback system for all languages 