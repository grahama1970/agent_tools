# Cursor Rules Project

## Project Overview
A tool for AI agents (Claude) to query coding guidelines and best practices stored in markdown files, using both keyword-based (BM25) and semantic vector search to find relevant rules and examples.

## Purpose and Motivation
AI agents like Claude need access to project-specific coding rules and best practices. This tool:
1. Loads markdown files containing coding rules from the `.cursor/rules` directory
2. Stores them in an ArangoDB database with vector embeddings 
3. Provides search capabilities using both keyword and semantic search
4. Makes these rules easily accessible to the AI agent through a simple CLI interface

When an AI agent gets confused or needs to ask questions about coding standards, it can use this tool to search the database and get relevant rules.

## Development Philosophy
1. **Documentation-First**: Thoroughly document the database schema and API
2. **Progressive Enhancement**: Start with a basic working system and enhance incrementally
3. **Testing as Validation**: Use tests to validate functionality at each step
4. **Flexibility**: Support different field names and structures in database results
5. **Error Resilience**: Handle cases where embeddings or other features are unavailable

## Key Components

### Database Schema
A detailed schema for the ArangoDB database is essential for this project. The schema will include:

```json
{
  "database_name": "cursor_rules",
  "description": "Database for storing and searching coding guidelines and best practices",
  "collections": {
    "rules": {
      "type": "document",
      "description": "Primary collection for storing cursor rules from markdown files",
      "fields": {
        "_key": {"type": "string", "description": "Unique identifier for the rule (derived from filename)", "indexed": true},
        "rule_number": {"type": "string", "description": "Numeric rule identifier (e.g., '001')", "indexed": true},
        "title": {"type": "string", "description": "Rule title from H1 heading", "indexed": true, "analyzer": "text_en"},
        "description": {"type": "string", "description": "Brief description of the rule", "analyzer": "text_en"},
        "content": {"type": "string", "description": "Full markdown content of the rule", "analyzer": "text_en"},
        "rule_type": {"type": "string", "description": "Type of rule (e.g., 'code_advice', 'pattern')", "indexed": true},
        "glob_pattern": {"type": "string", "description": "File pattern where rule applies (e.g., '*.py')", "indexed": true},
        "priority": {"type": "integer", "description": "Priority order for rules", "indexed": true},
        "embedding": {"type": "vector", "dimension": 768, "similarity": "cosine", "description": "Vector embedding for semantic search"}
      },
      "primary_sort_field": ["rule_number"],
      "sample_doc": {
        "_key": "001-code-advice-rules",
        "rule_number": "001",
        "title": "Code Advice Rules",
        "description": "Project Code Advice Rules for AI Code Generation",
        "rule_type": "code_advice",
        "glob_pattern": "*.py",
        "priority": 1
      }
    },
    "rule_examples": {
      "type": "document",
      "description": "Code examples demonstrating rules",
      "fields": {
        "_key": {"type": "string", "description": "Unique identifier for the example", "indexed": true},
        "rule_key": {"type": "string", "description": "Reference to rules._key", "indexed": true},
        "title": {"type": "string", "description": "Example title", "indexed": true, "analyzer": "text_en"},
        "description": {"type": "string", "description": "Description of the example", "analyzer": "text_en"},
        "language": {"type": "string", "description": "Programming language of the example", "indexed": true},
        "good_example": {"type": "string", "description": "Code example showing good practice", "analyzer": "text_en"},
        "bad_example": {"type": "string", "description": "Code example showing anti-pattern", "analyzer": "text_en"},
        "embedding": {"type": "vector", "dimension": 768, "similarity": "cosine", "description": "Vector embedding for semantic search"}
      },
      "indexes": [
        {
          "type": "persistent",
          "fields": ["rule_key"],
          "sparse": false
        }
      ]
    }
  },
  "edge_collections": {
    "rule_has_example": {
      "type": "edge",
      "description": "Links rules to their examples",
      "fields": {
        "relationship_type": {"type": "string", "description": "Type of relationship (default: 'example')", "indexed": true}
      },
      "from": ["rules"],
      "to": ["rule_examples"]
    },
    "rule_references_rule": {
      "type": "edge",
      "description": "Links between related rules",
      "fields": {
        "relationship_type": {"type": "string", "description": "Type of relationship (e.g., 'similar', 'prerequisite')", "indexed": true},
        "strength": {"type": "number", "description": "Relationship strength 0-1", "indexed": true}
      },
      "from": ["rules"],
      "to": ["rules"]
    }
  },
  "views": {
    "rules_search_view": {
      "type": "arangosearch",
      "description": "Unified search across rules and examples with BM25 and vector search",
      "properties": {
        "analyzer": "text_en",
        "features": ["frequency", "position", "norm", "bm25"],
        "include_all_fields": false,
        "track_listings": true,
        "store_values": true,
        "cleanup_interval_step": 2,
        "commit_interval_msec": 1000
      },
      "links": {
        "rules": {
          "fields": {
            "title": {"analyzer": "text_en"},
            "description": {"analyzer": "text_en"},
            "content": {"analyzer": "text_en"},
            "embedding": {"type": "vector", "dimension": 768}
          }
        },
        "rule_examples": {
          "fields": {
            "title": {"analyzer": "text_en"},
            "description": {"analyzer": "text_en"},
            "good_example": {"analyzer": "text_en"},
            "bad_example": {"analyzer": "text_en"},
            "embedding": {"type": "vector", "dimension": 768}
          }
        }
      }
    }
  },
  "example_queries": {
    "hybrid_search": {
      "name": "Hybrid Rule Search",
      "description": "Combined BM25 and vector search",
      "query": "FOR doc IN rules_search_view SEARCH ANALYZER(doc.title LIKE @searchText OR doc.description LIKE @searchText OR doc.content LIKE @searchText, \"text_en\") LET score = BM25(doc) + 0.5 * COSINE_SIMILARITY(doc.embedding, @queryVector) SORT score DESC LIMIT 10 RETURN { \"rule\": doc, \"relevance\": score }"
    },
    "rule_with_examples": {
      "name": "Rules with Examples",
      "description": "Find rules with their associated examples",
      "query": "FOR rule IN rules FILTER rule.rule_number == @ruleNumber LET examples = (FOR example IN OUTBOUND rule rule_has_example RETURN example) RETURN { \"rule\": rule, \"examples\": examples }"
    },
    "related_rules": {
      "name": "Related Rules Search",
      "description": "Find rules related to a given rule",
      "query": "FOR r1 IN rules FILTER r1.rule_number == @ruleNumber LET related = (FOR r2 IN OUTBOUND r1 rule_references_rule FILTER r2.relationship_type IN @relationTypes AND r2.strength >= @minStrength RETURN r2) RETURN { \"rule\": r1, \"related_rules\": related }"
    }
  }
}
```

### Vector Embeddings
- Uses a suitable transformer model to generate embeddings for rule content
- Embeddings enable semantic search capabilities
- Gracefully degrades to keyword search when embeddings are unavailable

### Field Name Translation
- Handles different field naming schemes in database results
- Provides fallbacks for alternative field names
- Ensures consistent access to data regardless of underlying structure

### Search Capabilities
- BM25 keyword search with text analyzer
- Vector similarity search using cosine similarity
- Hybrid search combining BM25 and vector search
- Support for finding examples related to rules

## Implementation Plan

### Phase 1: Basic Functionality (Completed)
- [x] Load rules from markdown files
- [x] Store in ArangoDB with basic schema
- [x] Implement simple CLI interface
- [x] Add basic keyword search (BM25)

### Phase 2: Enhanced Search (Completed ✅)
- [✅] Add vector embeddings for semantic search
- [✅] Implement vector similarity search
- [✅] Create hybrid search combining BM25 and vector search
- [✅] Field name translation for resilience to schema changes
- [✅] Add ArangoSearch view for optimized search

### Phase 3: AI-Optimized Knowledge Retrieval System (In Progress)
- [ ] Develop test-first retrieval scenarios to guide implementation:
  - [ ] Document common questions needed when coding (test: `test_common_queries.py`)
  - [ ] Create scenarios for correct method usage patterns (test: `test_method_patterns.py`)
  - [ ] Develop error-message-to-solution mapping (test: `test_error_resolution.py`)
  - [ ] Define scenarios for finding dependencies between patterns (test: `test_pattern_dependencies.py`)
  - [ ] Create scenarios for validating against documentation (test: `test_doc_validation.py`)
  - [ ] Create multi-hop knowledge traversal scenarios (test: `test_knowledge_traversal.py`)
  - [ ] Document AQL query patterns for each scenario (test: `test_aql_patterns.py`)
  - [ ] Define expected result formats for each scenario (test: `test_result_formats.py`)
  - [ ] Create test cases with sample inputs and outputs (test: `test_input_output_cases.py`)

- [ ] Design AI-optimized database schema (depends on retrieval scenarios):
  - [ ] Create specialized collections for different knowledge types (test: `test_collections.py`):
    - [ ] `method_signatures` collection (test: `test_method_signatures_collection.py`)
    - [ ] `error_codes` collection (test: `test_error_codes_collection.py`) 
    - [ ] `code_patterns` collection (test: `test_code_patterns_collection.py`)
    - [ ] `critical_rules` collection (test: `test_critical_rules_collection.py`)
    - [ ] `documentation_references` collection (test: `test_doc_references_collection.py`)
  - [ ] Implement metadata in collections (test: `test_collection_metadata.py`):
    - [ ] Verification status fields (test: `test_verification_status.py`)
    - [ ] Confidence scores (test: `test_confidence_scores.py`)
    - [ ] Usage statistics (test: `test_usage_statistics.py`)
  - [ ] Implement edge collections (test: `test_edge_collections.py`):
    - [ ] `requires_import` edges (test: `test_requires_import_edges.py`)
    - [ ] `has_parameter` edges (test: `test_has_parameter_edges.py`)
    - [ ] `triggers_error` edges (test: `test_triggers_error_edges.py`)
    - [ ] `resolves_error` edges (test: `test_resolves_error_edges.py`)
    - [ ] `alternative_to` edges (test: `test_alternative_edges.py`)
    - [ ] `documented_at` edges (test: `test_documented_at_edges.py`)

- [ ] Create precision-targeted query interfaces (depends on schema implementation):
  - [ ] `query_by_error` interface (test: `test_query_by_error.py`)
  - [ ] `query_by_method` interface (test: `test_query_by_method.py`)
  - [ ] `query_by_task` interface (test: `test_query_by_task.py`)
  - [ ] `query_by_import` interface (test: `test_query_by_import.py`)
  - [ ] `query_by_related_patterns` interface (test: `test_query_by_related_patterns.py`)
  - [ ] `query_critical_rules` interface (test: `test_query_critical_rules.py`)
  - [ ] Additional specialized queries (test: `test_specialized_queries.py`):
    - [ ] `query_by_language` (test: `test_query_by_language.py`)
    - [ ] `query_by_framework` (test: `test_query_by_framework.py`)
    - [ ] `query_by_complexity` (test: `test_query_by_complexity.py`)
    - [ ] `query_by_performance` (test: `test_query_by_performance.py`)
    - [ ] `query_by_security` (test: `test_query_by_security.py`)
    - [ ] `query_by_compatibility` (test: `test_query_by_compatibility.py`)

- [ ] Implement specialized indices (depends on query interfaces):
  - [ ] Full-text search on error messages (test: `test_error_message_index.py`)
  - [ ] Method name trigram indices (test: `test_method_name_index.py`)
  - [ ] Package+method compound indices (test: `test_compound_indices.py`)
  - [ ] File pattern indices (test: `test_file_pattern_index.py`)
  - [ ] Task type indices (test: `test_task_type_indices.py`):
    - [ ] Implementation complexity index (test: `test_complexity_index.py`)
    - [ ] Performance optimization index (test: `test_performance_index.py`)
    - [ ] Security requirement index (test: `test_security_index.py`)
    - [ ] Compatibility index (test: `test_compatibility_index.py`)

- [ ] Design context-optimized response formats (depends on query implementations):
  - [ ] Minimalist JSON responses (test: `test_minimalist_responses.py`)
  - [ ] Directly usable code snippets (test: `test_code_snippet_format.py`)
  - [ ] Standard response format with consistent fields (test: `test_standard_format.py`)
  - [ ] Include confidence and verification metadata (test: `test_metadata_inclusion.py`)
  - [ ] Documentation links integration (test: `test_doc_link_integration.py`)
  - [ ] Hierarchical response format (test: `test_hierarchical_format.py`)
  - [ ] Condensed format for high-confidence matches (test: `test_condensed_format.py`)
  - [ ] Expanded format for low-confidence matches (test: `test_expanded_format.py`)

- [ ] Implement performance optimization (depends on response formats):
  - [ ] Response size limits (test: `test_response_size_limits.py`)
  - [ ] Query timeouts (test: `test_query_timeouts.py`)
  - [ ] Multi-level caching (test: `test_multi_level_caching.py`):
    - [ ] In-memory LRU cache (test: `test_lru_cache.py`)
    - [ ] Persistent cache (test: `test_persistent_cache.py`)
    - [ ] Cache invalidation (test: `test_cache_invalidation.py`)
  - [ ] Result prioritization (test: `test_result_prioritization.py`)
  - [ ] Relevance scoring (test: `test_relevance_scoring.py`)
  - [ ] Query plan optimization (test: `test_query_plan_optimization.py`)
  - [ ] Parallel query execution (test: `test_parallel_queries.py`)

- [ ] Develop error resilience strategies (depends on performance optimization):
  - [ ] Fallback query patterns (test: `test_fallback_queries.py`)
  - [ ] Degradation paths for partial matches (test: `test_partial_matches.py`):
    - [ ] Fuzzy matching for method names (test: `test_fuzzy_method_matching.py`)
    - [ ] Substring matching for errors (test: `test_substring_matching.py`)
    - [ ] Semantic similarity fallbacks (test: `test_semantic_fallbacks.py`)
  - [ ] Alternative retrieval methods (test: `test_alternative_retrieval.py`)
  - [ ] Fuzzy matching for errors (test: `test_fuzzy_error_matching.py`)
  - [ ] Feedback mechanism (test: `test_feedback_mechanism.py`):
    - [ ] Tracking successful resolutions (test: `test_resolution_tracking.py`)
    - [ ] Adjusting confidence scores (test: `test_confidence_adjustment.py`)
    - [ ] Learning from failed retrievals (test: `test_failed_retrieval_learning.py`)

### Expected Tests for AI Knowledge Database

- [ ] Unit Tests in `test_ai_knowledge_db.py`:
  - [ ] Test schema loading
  - [ ] Test document collection creation
  - [ ] Test edge collection creation
  - [ ] Test named graph creation
  - [ ] Test view creation
  - [ ] Test analyzer creation
  - [ ] Test schema document storage and retrieval
  - [ ] Test database setup end-to-end

- [ ] Integration Tests in `test_ai_knowledge_db_integration.py`:
  - [ ] Test database setup with real ArangoDB
  - [ ] Test schema storage and retrieval in real database
  - [ ] Test document creation and retrieval
  - [ ] Test graph operations with real database
  - [ ] Test search with ArangoSearch view

- [ ] Query Tests in `test_ai_knowledge_queries.py`:
  - [ ] Test method signature query with parameters
  - [ ] Test error resolution query
  - [ ] Test pattern with potential errors query
  - [ ] Test multi-step graph traversal
  - [ ] Test pattern search by category
  - [ ] Test shortest path query
  - [ ] Test filtered traversal query

- [ ] CLI Tests in `test_enhanced_cli.py`:
  - [ ] Test search command
  - [ ] Test related command
  - [ ] Test path command
  - [ ] Test recommend command
  - [ ] Test setup command
  - [ ] Test verbose output option

- [ ] Basic Tests in `test_ai_knowledge_basic.py`:
  - [ ] Test schema loading
  - [ ] Test document collection creation with mocks
  - [ ] Test edge collection creation with mocks
  - [ ] Test view creation with mocks
  - [ ] Test analyzer creation with mocks
  - [ ] Database connection tests

## Testing Strategy

### Unit Tests
- [x] Test rule loading from markdown files
- [x] Test field name translation functionality
- [x] Test embedding generation
- [x] Test ArangoDB query construction

### Integration Tests
- [x] Test end-to-end rule loading and database storage
- [x] Test search functionality with known queries
- [x] Test CLI interface with simulated inputs

### Method Validation
- [x] Validate ArangoDB Python driver methods exist and match our usage
- [x] Verify asyncio methods work correctly with ArangoDB's synchronous nature

### Benchmark Tests
- [ ] Measure search performance with large rule sets
- [ ] Compare BM25, vector, and hybrid search quality
- [ ] Evaluate embedding generation efficiency

## CLI Interface

### Commands
- `init`: Initialize the database with rules from a directory
- `list`: List all rules in the database
- `show`: Show details for a specific rule
- `search`: Search for rules using different methods
  - `--keyword`: Use BM25 keyword search
  - `--semantic`: Use vector similarity search
  - `--hybrid`: Use combined BM25 and vector search
  - `--limit`: Maximum number of results to return

### Usage Examples

```bash
# Initialize the database with rules from the .cursor/rules directory
python -m src.agent_tools.cursor_rules.cli init --rules-dir .cursor/rules

# List all rules
python -m src.agent_tools.cursor_rules.cli list

# Show details for a specific rule
python -m src.agent_tools.cursor_rules.cli show 001

# Search for rules using keyword search
python -m src.agent_tools.cursor_rules.cli search "async patterns" --keyword

# Search for rules using semantic search
python -m src.agent_tools.cursor_rules.cli search "concurrency best practices" --semantic

# Search for rules using hybrid search (combines BM25 and vector search)
python -m src.agent_tools.cursor_rules.cli search "database operations" --hybrid
```

## AQL Query Examples

### BM25 Keyword Search
```aql
FOR doc IN rules_search_view
  SEARCH ANALYZER(
    doc.title LIKE @searchText OR
    doc.description LIKE @searchText OR
    doc.content LIKE @searchText,
    "text_en"
  )
  LET score = BM25(doc)
  FILTER score > 0.5
  SORT score DESC
  LIMIT 10
  RETURN { rule: doc, score: score }
```

### Vector Similarity Search
```aql
FOR doc IN rules
  LET similarity = COSINE_SIMILARITY(doc.embedding, @queryVector)
  FILTER similarity >= 0.7
  SORT similarity DESC
  LIMIT 10
  RETURN { rule: doc, similarity: similarity }
```

### Hybrid Search
```aql
FOR doc IN rules_search_view
  SEARCH ANALYZER(
    doc.title LIKE @searchText OR
    doc.description LIKE @searchText OR
    doc.content LIKE @searchText,
    "text_en"
  )
  LET bm25_score = BM25(doc)
  LET vector_score = COSINE_SIMILARITY(doc.embedding, @queryVector)
  LET hybrid_score = bm25_score * 0.5 + vector_score * 0.5
  FILTER hybrid_score > 0.5
  SORT hybrid_score DESC
  LIMIT 10
  RETURN { 
    rule: doc, 
    hybrid_score: hybrid_score,
    bm25_score: bm25_score,
    vector_score: vector_score
  }
```

### Rule with Examples
```aql
FOR rule IN rules
  FILTER rule.rule_number == @ruleNumber
  LET examples = (
    FOR example IN rule_examples
    FILTER example.rule_key == rule._key
    RETURN example
  )
  RETURN { rule: rule, examples: examples }
```

## Future Enhancements

1. **Rule Versioning**: Track changes to rules over time
2. **Automated Updates**: Automatically detect and incorporate rule changes
3. **Usage Analytics**: Track which rules are most frequently accessed
4. **Relevance Feedback**: Allow the agent to provide feedback on search results
5. **Custom Embedding Models**: Use domain-specific embedding models for better similarity
6. **Caching Layer**: Add caching for frequently accessed rules and queries
7. **Query Optimization**: Analyze and optimize complex queries for better performance
8. **Cross-Project Rules**: Support sharing rules across multiple projects

## Test Status

### Field Translation
- ✅ Tests in `test_field_translation.py` are passing

### Semantic Search
- ✅ Tests in `test_semantic_search.py` are passing
- ✅ Tests in `test_semantic_search_simple.py` are passing

### BM25 Search
- ✅ Tests in `test_bm25_search.py` are passing
  - Fixed by using exact AQL pattern from `bm25_search.aql`
- ✅ Tests in `test_bm25_search_simple.py` are passing

### Hybrid Search
- ✅ Tests in `test_hybrid_search.py` are passing
  - Fixed by using the correct AQL pattern with `IN TOKENS()` instead of `LIKE`
  - Implemented using separate subqueries for embedding and BM25 results
  - Used `textwrap.dedent()` to properly format multi-line strings

### Cursor Rules
- ❌ Tests in `test_cursor_rules.py` are failing
  - Issue: Database connectivity issues
  - Issue: Permissions for ArangoSearch view creation

## CLI Validation

### Commands Tested and Documented
- ✅ `init` - Initializes the cursor rules database
- ✅ `list` - Lists all rules
- ✅ `show` - Shows a specific rule
- ✅ `search` - Searches for rules
  - ✅ BM25 search (keyword)
  - ✅ Semantic search
  - ✅ Hybrid search

### Issues Discovered
- ✅ Missing fields in rule examples
- ✅ Functions need to handle alternative field names
- ❌ Database connectivity issues
- ❌ Permissions for ArangoSearch view creation

### Agent Validation
- ✅ BM25 search verified with CLI for both matching ("python style guide") and non-matching ("mustard") queries
- ✅ Semantic search verified with CLI for both matching ("python style") and non-matching ("quantum physics theories") queries
- ✅ Hybrid search verified with test for matching ("python style guide") query
- ✅ Hybrid search verified with CLI for both matching ("python style") and non-matching ("quantum physics theories") queries

## Summary of Fixes

1. **BM25 Search**
   - Fixed by using the correct AQL pattern with `IN TOKENS()` instead of `LIKE`
   - Ensured proper handling of results in the CLI

2. **Semantic Search**
   - Created a simple test to verify both matching and non-matching results
   - Ensured proper handling of results in the CLI

3. **Hybrid Search**
   - Implemented using separate subqueries for embedding and BM25 results
   - Fixed by using the correct AQL pattern with `IN TOKENS()` instead of `LIKE`
   - Used `textwrap.dedent()` to properly format multi-line strings
   - Updated the CLI to handle the tuple format of results
   - Added robust error handling for missing fields in rules

4. **CLI Improvements**
   - Made the CLI more robust by handling missing fields in rules and examples
   - Improved error handling and user feedback
   - Ensured consistent output format across different search methods
