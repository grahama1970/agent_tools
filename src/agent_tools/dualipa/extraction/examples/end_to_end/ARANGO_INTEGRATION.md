# ArangoDB Integration Guide

This document explains how the extraction modules integrate with ArangoDB, including data models, collection structures, and example queries.

## Database Architecture

### Collections

#### Document Collections

| Collection Name | Description | Source |
|----------------|------------|--------|
| `files` | Source code and markdown files | All extractors |
| `sections` | Markdown sections and headings | Markdown extractor |
| `classes` | Programming language classes | AST & tree-sitter extractors |
| `functions` | Functions and methods | AST & tree-sitter extractors |
| `tables` | Markdown tables | Markdown extractor |
| `code_blocks` | Code blocks in markdown | Markdown extractor |
| `images` | Images referenced in markdown | Markdown extractor |

#### Edge Collections

| Collection Name | Description | Connects |
|----------------|------------|---------|
| `contains` | Parent-child relationship | files → sections, classes → methods, etc. |
| `imports` | Import relationships | files → files |
| `references` | Cross-references | various entities → various entities |
| `extends` | Inheritance relationships | classes → classes |
| `follows` | Sequential ordering | sections → sections, elements → elements |

### Data Models

#### Document Structure

All documents (nodes) have a common base structure:

```json
{
  "_key": "auto-generated-key",
  "uuid": "uuid-from-extraction",
  "type": "file|section|class|function|table|code_block|image",
  "name": "Entity name",
  "language": "python|javascript|typescript|markdown|etc",
  "content": "Full content of the entity",
  "metadata": {
    "position": 1234,
    "line_start": 10,
    "line_end": 20,
    "extraction_timestamp": "2025-03-21T12:00:00Z"
  }
}
```

#### Edge Structure

All edges have a common base structure:

```json
{
  "_from": "collection/document-key",
  "_to": "collection/document-key",
  "relationship_type": "contains|imports|references|extends|follows",
  "metadata": {
    "confidence": 1.0,
    "extracted_by": "markdown_extractor|python_ast_extractor|treesitter_extractor"
  }
}
```

## Integration Pipeline

1. **Extraction**: Each extractor processes files and generates JSON output
2. **Normalization**: Outputs are normalized to a consistent format
3. **UUID Generation**: Ensure all entities have unique identifiers
4. **Relationship Mapping**: Create edges between related entities
5. **Database Loading**: Insert documents and edges into ArangoDB
6. **Indexing**: Create indexes for efficient querying
7. **Validation**: Verify data integrity and relationships

## Insertion Process

### Markdown Extraction Import

```javascript
// ArangoDB JavaScript function for importing markdown extraction output
function importMarkdownExtraction(data) {
  const sections = [];
  const tables = [];
  const codeBlocks = [];
  const images = [];
  const containsEdges = [];
  const followsEdges = [];
  
  // Process each section
  data.forEach((section, i) => {
    // Create section document
    const sectionDoc = {
      _key: section.uuid,
      uuid: section.uuid,
      type: "section",
      name: section.title,
      content: section.content,
      hierarchy_depth: section.section_hierarchy_depth,
      metadata: {
        extraction_timestamp: new Date().toISOString()
      }
    };
    sections.push(sectionDoc);
    
    // Create parent-child edges
    if (section.section_hierarchy_depth.length > 1) {
      // Find parent section by hierarchy
      const parentHierarchy = section.section_hierarchy_depth.slice(0, -1);
      const parentSection = data.find(s => 
        JSON.stringify(s.section_hierarchy_depth) === JSON.stringify(parentHierarchy)
      );
      
      if (parentSection) {
        containsEdges.push({
          _from: `sections/${parentSection.uuid}`,
          _to: `sections/${section.uuid}`,
          relationship_type: "contains"
        });
      }
    }
    
    // Create follows edges (for ordering)
    if (i > 0) {
      const prevSection = data[i-1];
      followsEdges.push({
        _from: `sections/${prevSection.uuid}`,
        _to: `sections/${section.uuid}`,
        relationship_type: "follows"
      });
    }
    
    // Process tables
    section.tables.forEach(table => {
      const tableDoc = {
        _key: table.uuid,
        uuid: table.uuid,
        type: "table",
        name: `Table in ${section.title}`,
        content: table.content,
        metadata: {
          extraction_timestamp: new Date().toISOString()
        }
      };
      tables.push(tableDoc);
      
      // Create contains edge
      containsEdges.push({
        _from: `sections/${section.uuid}`,
        _to: `tables/${table.uuid}`,
        relationship_type: "contains"
      });
    });
    
    // Process code blocks
    section.code.forEach(code => {
      const codeDoc = {
        _key: code.uuid,
        uuid: code.uuid,
        type: "code_block",
        name: `Code block in ${section.title}`,
        content: code.content,
        language: code.language,
        metadata: {
          extraction_timestamp: new Date().toISOString()
        }
      };
      codeBlocks.push(codeDoc);
      
      // Create contains edge
      containsEdges.push({
        _from: `sections/${section.uuid}`,
        _to: `code_blocks/${code.uuid}`,
        relationship_type: "contains"
      });
    });
    
    // Process images
    section.images.forEach(image => {
      const imageDoc = {
        _key: image.uuid,
        uuid: image.uuid,
        type: "image",
        name: image.alt,
        url: image.src,
        metadata: {
          extraction_timestamp: new Date().toISOString()
        }
      };
      images.push(imageDoc);
      
      // Create contains edge
      containsEdges.push({
        _from: `sections/${section.uuid}`,
        _to: `images/${image.uuid}`,
        relationship_type: "contains"
      });
    });
  });
  
  // Insert into collections
  db.sections.save(sections);
  db.tables.save(tables);
  db.code_blocks.save(codeBlocks);
  db.images.save(images);
  db.contains.save(containsEdges);
  db.follows.save(followsEdges);
}
```

## Example Queries

### Finding All Sections in a Document

```aql
FOR section IN sections
  FILTER section.hierarchy_depth[0] == "DeepSeek Usage"
  SORT LENGTH(section.hierarchy_depth) ASC
  RETURN {
    title: section.name,
    depth: LENGTH(section.hierarchy_depth),
    content: section.content
  }
```

### Finding Tables in a Section

```aql
LET section = DOCUMENT("sections/550e8400-e29b-41d4-a716-446655440000")
FOR edge IN contains
  FILTER edge._from == section._id
  FOR table IN tables
    FILTER edge._to == table._id
    RETURN {
      section_title: section.name,
      table_headers: table.content.headers,
      table_rows: table.content.rows
    }
```

### Finding Related Code Across Documents

```aql
// Find all code blocks mentioning "DeepSeek"
FOR code IN code_blocks
  FILTER CONTAINS(code.content, "DeepSeek")
  FOR edge IN contains
    FILTER edge._to == code._id
    FOR section IN sections
      FILTER edge._from == section._id
      RETURN {
        section: section.name,
        code_language: code.language,
        code_snippet: SUBSTRING(code.content, 0, 200)
      }
```

## Performance Considerations

1. **Indexing Strategy**:
   - Create indexes on frequently queried fields: `uuid`, `name`, `type`
   - Create full-text indexes on `content` fields
   - Create geo-spatial indexes if location data is present

2. **Batching**:
   - Insert documents in batches of 1000
   - Use ArangoDB's bulk import APIs for large datasets

3. **Graph Traversal Optimization**:
   - Use directed traversals when possible
   - Limit depth for recursive queries
   - Use collection filtering in graph queries

4. **Sharding Considerations**:
   - Consider sharding by repository or project for large datasets
   - Use smart graph for efficient cross-shard traversals

## Data Maintenance

1. **Versioning Strategy**:
   - Add version field to documents
   - Keep history of changes in separate collections
   - Use time-to-live (TTL) for temporary data

2. **Consistency Checks**:
   - Verify all edges point to valid documents
   - Check for orphaned documents
   - Ensure hierarchy properties match actual graph structure

3. **Cleanup Procedures**:
   - Remove temporary collections after processing
   - Archive old versions
   - Compact databases regularly