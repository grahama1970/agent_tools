# AI Agent Retrieval Scenarios for Cursor Rules

This document outlines common data retrieval scenarios that an AI agent needs when writing code. These scenarios will guide the database design and implementation to ensure the system meets the agent's actual needs.

## Method Usage Patterns

### 1. Find Correct Usage of `asyncio.to_thread`
- **Query Need**: When implementing async code using `asyncio.to_thread`, need to see verified patterns
- **Expected Result**: Example code snippets showing correct usage, common mistakes to avoid
- **AQL Pattern**: 
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.method_name == "asyncio.to_thread"
    SORT pattern.confidence_score DESC
    LIMIT 1
    LET docs = (FOR doc IN OUTBOUND pattern documented_at RETURN doc)
    RETURN { pattern: pattern, documentation_links: docs }
  ```

### 2. Verify Method Signature for `requests.post`
- **Query Need**: Need to confirm parameters for `requests.post` method
- **Expected Result**: Method signature, parameter types, required vs optional params
- **AQL Pattern**:
  ```aql
  FOR method IN method_signatures
    FILTER method.package == "requests" AND method.name == "post"
    LET params = (FOR param IN OUTBOUND method has_parameter RETURN param)
    RETURN { method: method, parameters: params }
  ```

### 3. Find Alternative Pattern to `time.sleep` in Async Code
- **Query Need**: Need async alternative to blocking `time.sleep`
- **Expected Result**: Alternative method suggestions with example usage
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.method_name == "time.sleep"
    LET alternatives = (FOR alt IN OUTBOUND pattern alternative_to
                         FILTER alt.is_async == true
                         RETURN alt)
    RETURN { pattern: pattern, alternatives: alternatives }
  ```

### 4. Identify Correct Import Statement
- **Query Need**: Need correct import statement for a specific functionality
- **Expected Result**: Import statements, package installation command if needed
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.method_name == @method_name
    LET imports = (FOR imp IN INBOUND pattern requires_import RETURN imp)
    RETURN { pattern: pattern, import_statements: imports }
  ```

### 5. Validate PyTorch Tensor Operations
- **Query Need**: Confirm correct way to perform specific tensor operation
- **Expected Result**: Verified code pattern with documentation link
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.library == "pytorch" AND pattern.operation_type == "tensor_manipulation"
    FILTER pattern.tags ANY == @specific_operation
    SORT pattern.verification_status DESC
    LIMIT 1
    RETURN pattern
  ```

## Error Resolution

### 6. Fix "Coroutine Was Never Awaited" Error
- **Query Need**: Find solution for asyncio error message
- **Expected Result**: Error explanation, resolution steps, code example
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    SEARCH ANALYZER(error.message IN TOKENS(@error_text, "text_en"), "text_en")
    LET solutions = (FOR sol IN OUTBOUND error resolves_error
                      SORT sol.confidence_score DESC
                      LIMIT 1
                      RETURN sol)
    RETURN { error: error, solution: solutions[0] }
  ```

### 7. Resolve Package Conflict
- **Query Need**: Fix dependency conflict between packages
- **Expected Result**: Resolution strategies, compatible versions
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.error_type == "package_conflict"
    FILTER error.packages ANY == @package1 AND error.packages ANY == @package2
    LET solutions = (FOR sol IN OUTBOUND error resolves_error RETURN sol)
    RETURN { error: error, solutions: solutions }
  ```

### 8. Debug Database Connection Error
- **Query Need**: Resolve ArangoDB connection issues
- **Expected Result**: Troubleshooting steps, configuration checks
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.category == "database" AND error.component == "arango"
    FILTER error.symptoms ANY == @error_symptom
    LET steps = (FOR step IN OUTBOUND error resolves_error
                  SORT step.step_order ASC
                  RETURN step)
    RETURN { error: error, troubleshooting_steps: steps }
  ```

### 9. Fix Incorrect AQL Query Pattern
- **Query Need**: Identify error in AQL query syntax
- **Expected Result**: Corrected query pattern, explanation
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.language == "aql" AND pattern.error_prone == true
    FILTER pattern.description LIKE CONCAT("%", @query_fragment, "%")
    LET corrected = (FOR cor IN OUTBOUND pattern alternative_to
                      FILTER cor.is_correct == true
                      RETURN cor)
    RETURN { incorrect_pattern: pattern, corrected_pattern: corrected[0] }
  ```

### 10. Resolve "Module Not Found" Error
- **Query Need**: Fix import errors with missing modules
- **Expected Result**: Installation commands, import fixes
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.error_type == "import_error" AND error.message LIKE "%ModuleNotFoundError%"
    FILTER error.module == @module_name
    LET solutions = (FOR sol IN OUTBOUND error resolves_error RETURN sol)
    RETURN { error: error, solutions: solutions }
  ```

## Documentation References

### 11. Find Official Documentation for Package
- **Query Need**: Get authoritative documentation link
- **Expected Result**: Documentation URL, version information
- **AQL Pattern**:
  ```aql
  FOR doc IN documentation_references
    FILTER doc.package == @package_name
    SORT doc.version DESC
    LIMIT 1
    RETURN doc
  ```

### 12. Get Example from Documentation
- **Query Need**: Find official example for specific functionality
- **Expected Result**: Code example from documentation, URL to section
- **AQL Pattern**:
  ```aql
  FOR doc IN documentation_references
    FILTER doc.package == @package_name AND doc.feature == @feature_name
    LET examples = (FOR ex IN OUTBOUND doc has_example RETURN ex)
    RETURN { documentation: doc, examples: examples }
  ```

### 13. Find Documentation for API Endpoint
- **Query Need**: Get details on API endpoint usage
- **Expected Result**: Endpoint specification, parameters, authentication
- **AQL Pattern**:
  ```aql
  FOR doc IN documentation_references
    FILTER doc.type == "api" AND doc.endpoint LIKE CONCAT("%", @endpoint_fragment, "%")
    RETURN doc
  ```

### 14. Compare Package Versions
- **Query Need**: Compare features across package versions
- **Expected Result**: Feature differences, breaking changes
- **AQL Pattern**:
  ```aql
  FOR doc1 IN documentation_references
    FILTER doc1.package == @package_name AND doc1.version == @version1
    FOR doc2 IN documentation_references
      FILTER doc2.package == @package_name AND doc2.version == @version2
      RETURN {
        version1: doc1.version,
        version2: doc2.version,
        changes: doc2.breaking_changes,
        new_features: doc2.new_features
      }
  ```

### 15. Find Latest Stable Version Documentation
- **Query Need**: Get documentation for latest stable version
- **Expected Result**: Documentation URL, version number
- **AQL Pattern**:
  ```aql
  FOR doc IN documentation_references
    FILTER doc.package == @package_name AND doc.is_stable == true
    SORT doc.version_date DESC
    LIMIT 1
    RETURN doc
  ```

## Critical Rules Retrieval

### 16. Get Rules for File Type
- **Query Need**: Find rules applicable to specific file type
- **Expected Result**: Prioritized list of rules
- **AQL Pattern**:
  ```aql
  FOR rule IN critical_rules
    FILTER rule.glob_pattern == @file_pattern OR rule.glob_pattern == "*"
    SORT rule.priority ASC
    RETURN rule
  ```

### 17. Check Rules for Current Package
- **Query Need**: Find rules for working with specific package
- **Expected Result**: Package-specific rules and patterns
- **AQL Pattern**:
  ```aql
  FOR rule IN critical_rules
    FILTER rule.packages ANY == @package_name
    SORT rule.priority ASC
    RETURN rule
  ```

### 18. Get Implementation Rule Requirements
- **Query Need**: Find what must be included in implementation
- **Expected Result**: Required components, patterns to follow
- **AQL Pattern**:
  ```aql
  FOR rule IN critical_rules
    FILTER rule.implementation_type == @implementation_type
    LET requirements = (FOR req IN OUTBOUND rule has_requirement RETURN req)
    RETURN { rule: rule, requirements: requirements }
  ```

### 19. Find Anti-Patterns to Avoid
- **Query Need**: Identify patterns that should not be used
- **Expected Result**: Anti-patterns with reasons to avoid
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.is_anti_pattern == true
    FILTER pattern.language == @language
    FILTER pattern.context == @context
    RETURN pattern
  ```

### 20. Get Critical Error Prevention Rules
- **Query Need**: Find rules to prevent common errors
- **Expected Result**: Prevention strategies, rules to follow
- **AQL Pattern**:
  ```aql
  FOR rule IN critical_rules
    FILTER rule.purpose == "error_prevention"
    FILTER rule.error_type == @error_type
    RETURN rule
  ```

## Multi-Hop Knowledge Traversal

### 21. Find Related Concepts
- **Query Need**: Discover concepts related to a topic
- **Expected Result**: Related concepts with relationship types
- **AQL Pattern**:
  ```aql
  FOR concept IN code_patterns
    FILTER concept.name == @concept_name
    LET related = (FOR rel IN 1..2 ANY concept GRAPH "knowledge_graph"
                   RETURN DISTINCT {
                     concept: rel,
                     relationship: rel.relationship_type,
                     distance: LENGTH(PATH)
                   })
    RETURN { concept: concept, related_concepts: related }
  ```

### 22. Discover Implementation Dependencies
- **Query Need**: Find what else is needed to implement a pattern
- **Expected Result**: Required dependencies, related patterns
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.name == @pattern_name
    LET dependencies = (FOR dep IN 1..1 OUTBOUND pattern requires_pattern
                        RETURN dep)
    LET imports = (FOR imp IN INBOUND pattern requires_import RETURN imp)
    RETURN { pattern: pattern, dependencies: dependencies, imports: imports }
  ```

### 23. Find Alternative Implementation Paths
- **Query Need**: Discover different ways to implement functionality
- **Expected Result**: Implementation alternatives with tradeoffs
- **AQL Pattern**:
  ```aql
  FOR start IN code_patterns
    FILTER start.functionality == @functionality
    LET alternatives = (FOR alt IN OUTBOUND start alternative_to
                         RETURN {
                           pattern: alt,
                           pros: alt.advantages,
                           cons: alt.disadvantages
                         })
    RETURN { pattern: start, alternatives: alternatives }
  ```

### 24. Connect Error to Root Cause
- **Query Need**: Find underlying cause of an error
- **Expected Result**: Error chain from symptom to root cause
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.message LIKE @error_message
    LET causes = (FOR cause IN 1..3 INBOUND error causes_error
                  RETURN { cause: cause, level: LENGTH(PATH) })
    RETURN { error: error, root_causes: causes }
  ```

### 25. Build Implementation Path
- **Query Need**: Get step-by-step guide to implement feature
- **Expected Result**: Ordered steps with code examples
- **AQL Pattern**:
  ```aql
  FOR goal IN code_patterns
    FILTER goal.name == @implementation_goal
    LET steps = (FOR step IN 1..5 INBOUND goal implementation_step
                 SORT step.order ASC
                 RETURN step)
    RETURN { goal: goal, implementation_steps: steps }
  ```

## Language-Specific Patterns

### 26. Find Python Async Best Practices
- **Query Need**: Get best practices for async Python
- **Expected Result**: Prioritized async patterns, anti-patterns
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.language == "python" AND pattern.category == "async"
    FILTER pattern.is_best_practice == true
    SORT pattern.confidence_score DESC
    RETURN pattern
  ```

### 27. JavaScript Promises vs Async/Await
- **Query Need**: Compare JavaScript async approaches
- **Expected Result**: Comparison with example code
- **AQL Pattern**:
  ```aql
  FOR p1 IN code_patterns
    FILTER p1.language == "javascript" AND p1.name == "promises"
    FOR p2 IN code_patterns
      FILTER p2.language == "javascript" AND p2.name == "async_await"
      RETURN { promises: p1, async_await: p2, comparison: p1.comparison_notes }
  ```

### 28. Find TypeScript Type Definition Patterns
- **Query Need**: Get patterns for TypeScript interfaces
- **Expected Result**: Type definition examples, best practices
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.language == "typescript" AND pattern.category == "types"
    FILTER pattern.sub_category == @type_category
    SORT pattern.priority ASC
    RETURN pattern
  ```

### 29. Get SQL Query Optimization Patterns
- **Query Need**: Find patterns to optimize SQL queries
- **Expected Result**: Optimization strategies with examples
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.language == "sql" AND pattern.purpose == "optimization"
    FILTER pattern.database_type == @database_type
    SORT pattern.impact_score DESC
    RETURN pattern
  ```

### 30. Find React Component Design Patterns
- **Query Need**: Get patterns for React components
- **Expected Result**: Component patterns with examples
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.framework == "react" AND pattern.component_type == @component_type
    SORT pattern.modern_practice DESC
    RETURN pattern
  ```

## Context-Specific Recommendations

### 31. Find Patterns for File Type
- **Query Need**: Get patterns relevant to current file type
- **Expected Result**: File-specific patterns and rules
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.file_types ANY == @file_extension
    SORT pattern.frequency DESC
    LIMIT 5
    RETURN pattern
  ```

### 32. Get Patterns Based on Import Statements
- **Query Need**: Find patterns based on imports in file
- **Expected Result**: Relevant patterns for imported packages
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.packages ANY IN @imported_packages
    SORT pattern.relevance DESC
    LIMIT 10
    RETURN pattern
  ```

### 33. Find Testing Patterns for Current Code
- **Query Need**: Get testing approaches for implementation
- **Expected Result**: Testing patterns with examples
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.purpose == "testing" AND pattern.applies_to == @implementation_type
    SORT pattern.reliability DESC
    RETURN pattern
  ```

### 34. Get Error Handling for Current Task
- **Query Need**: Find appropriate error handling
- **Expected Result**: Error handling patterns for context
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.category == "error_handling" AND pattern.context == @current_context
    SORT pattern.robustness DESC
    RETURN pattern
  ```

### 35. Find Optimization for Current Algorithm
- **Query Need**: Get optimization techniques
- **Expected Result**: Optimization approaches with tradeoffs
- **AQL Pattern**:
  ```aql
  FOR pattern IN code_patterns
    FILTER pattern.category == "optimization" AND pattern.algorithm_type == @algorithm_type
    SORT pattern.performance_impact DESC
    RETURN { 
      pattern: pattern, 
      tradeoffs: { 
        speed: pattern.speed_impact, 
        memory: pattern.memory_impact, 
        readability: pattern.readability_impact 
      }
    }
  ```

## Error Message Analysis

### 36. Parse Stack Trace for Solution
- **Query Need**: Analyze stack trace to find solution
- **Expected Result**: Error identification, solution steps
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.stack_trace_pattern LIKE @stack_trace_fragment
    LET solutions = (FOR sol IN OUTBOUND error resolves_error
                      SORT sol.effectiveness DESC
                      LIMIT 1
                      RETURN sol)
    RETURN { identified_error: error, solution: solutions[0] }
  ```

### 37. Analyze Compiler Error Message
- **Query Need**: Understand compiler error and fix
- **Expected Result**: Error explanation, fix example
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.compiler == @compiler_name
    FILTER error.code == @error_code OR error.message LIKE @error_message
    LET fixes = (FOR fix IN OUTBOUND error resolves_error RETURN fix)
    RETURN { error: error, fixes: fixes }
  ```

### 38. Find Solution by Error Keywords
- **Query Need**: Match error solution by keywords
- **Expected Result**: Potential solutions ranked by relevance
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    SEARCH ANALYZER(error.keywords ANY IN TOKENS(@error_keywords, "text_en"), "text_en")
    LET solutions = (FOR sol IN OUTBOUND error resolves_error RETURN sol)
    SORT LENGTH(solutions) DESC
    LIMIT 5
    RETURN { error: error, solutions: solutions }
  ```

### 39. Get Common Mistakes for Warning
- **Query Need**: Find common causes of warning
- **Expected Result**: Warning explanation, common mistakes
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.severity == "warning" AND error.message LIKE @warning_message
    LET mistakes = (FOR m IN INBOUND error causes_error 
                    FILTER m.is_common == true
                    RETURN m)
    RETURN { warning: error, common_mistakes: mistakes }
  ```

### 40. Identify Missing Dependency Errors
- **Query Need**: Determine if error caused by missing dependency
- **Expected Result**: Dependency information, installation steps
- **AQL Pattern**:
  ```aql
  FOR error IN error_codes
    FILTER error.error_type == "dependency_error" AND error.message LIKE @error_message
    LET dependencies = (FOR dep IN OUTBOUND error requires_dependency RETURN dep)
    RETURN { error: error, missing_dependencies: dependencies }
  ```

These retrieval scenarios cover the most common questions I need answers to when writing code. They will guide the database schema design to ensure it can efficiently answer these queries with minimal context usage. 