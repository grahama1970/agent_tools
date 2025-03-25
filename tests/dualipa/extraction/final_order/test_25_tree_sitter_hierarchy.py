"""
TEST EXPECTATIONS

1. test_typescript_interface_hierarchy:
   Input: TypeScript interface + implementing class
   Expected Output:
   {
       "file": "src/repositories/user.repository.ts",
       "language": "typescript",
       "blocks": [
           {
               "type": "interface",
               "name": "Repository",
               "methods": [
                   {"name": "find", "metadata": {"visibility": "public"}},
                   {"name": "save", "metadata": {"visibility": "public"}},
                   {"name": "delete", "metadata": {"visibility": "public"}}
               ]
           },
           {
               "type": "class",
               "name": "UserRepository",
               "decorators": ["Injectable"],
               "methods": [
                   {"name": "constructor", "metadata": {"visibility": "public"}},
                   {"name": "find", "metadata": {"visibility": "public", "async": true}},
                   {"name": "save", "metadata": {"visibility": "public", "async": true}},
                   {"name": "delete", "metadata": {"visibility": "public", "async": true}}
               ]
           }
       ],
       "order": ["Repository", "UserRepository"],
       "stats": {
           "total_blocks": 8,  # Interface + Class + 6 methods
           "by_type": {
               "interface": 1,
               "class": 1,
               "method": 6  # 3 interface methods + 3 class methods (excluding constructor)
           }
       }
   }

2. test_class_hierarchy_with_nested_structure:
   Input: TypeScript class with various method types
   Expected Output:
   {
       "file": "src/services/user.service.ts",
       "language": "typescript",
       "blocks": [
           {
               "type": "class",
               "name": "UserService",
               "methods": [
                   {"name": "constructor", "metadata": {"visibility": "public"}},
                   {"name": "createAdmin", "metadata": {"visibility": "public", "static": true}},
                   {"name": "findById", "metadata": {"visibility": "public", "async": true}},
                   {"name": "validateUser", "metadata": {"visibility": "private"}}
               ]
           }
       ],
       "order": ["UserService"],
       "stats": {
           "total_blocks": 5,  # 1 class + 4 methods
           "by_type": {
               "class": 1,
               "method": 4  # All methods including constructor
           }
       }
   }

CRITICAL COUNTING RULES:
1. Interface Methods:
   - Count all methods EXCEPT constructors
   - Each method counts as one block in stats
   - Methods inherit interface's visibility unless specified

2. Class Methods:
   - Count ALL methods INCLUDING constructors
   - Each method counts as one block in stats
   - Methods inherit class's visibility unless specified
   - Constructor is always counted but excluded from interface implementation count

3. Total Blocks Calculation:
   - Sum of: interfaces + classes + all methods
   - Example 1: 1 interface + 1 class + 6 methods = 8 blocks
   - Example 2: 1 class + 4 methods = 5 blocks

4. Metadata Rules:
   - visibility: "public" (default) | "private" | "protected"
   - static: true if static method
   - async: true if async method
   - All metadata fields are optional, omit if false/default

Test tree-sitter hierarchy extraction with clearly defined input/output structure.

Input:
- Source code (string)
- Language identifier (string)
- File path (string, optional)

Output Structure:
{
    "file": "path/to/file.ext",
    "language": "language_id",
    "blocks": [
        {
            "type": "interface|class|function|method",
            "name": "string",
            "content": "exact source code",
            "start_line": number,
            "end_line": number,
            "methods": [...],  # For classes
            "implementations": [...],  # For interfaces
            "decorators": [...],  # For Python/TypeScript
            "metadata": {
                "visibility": "public|private|protected",
                "static": boolean,
                "async": boolean
            }
        }
    ],
    "order": ["block_names_in_declaration_order"],
    "stats": {
        "total_blocks": number,
        "by_type": {"class": number, ...}
    }
}

This test verifies that tree-sitter correctly extracts hierarchical structure
that will be used for training data generation in Stage 3 of the DuaLipa pipeline.
"""

import pytest
from pathlib import Path
from agent_tools.dualipa.extraction.extractors.code.hierarchy import _extract_hierarchical_structure_treesitter

def test_typescript_interface_hierarchy():
    """Test extraction of TypeScript interface with implementations."""
    # Input: TypeScript code with interface and implementation
    source_code = '''
interface Repository<T> {
    find(id: string): Promise<T>;
    save(entity: T): Promise<T>;
    delete(id: string): Promise<void>;
}

@Injectable()
class UserRepository implements Repository<User> {
    constructor(private db: Database) {}

    async find(id: string): Promise<User> {
        return this.db.users.findUnique({ where: { id } });
    }

    async save(user: User): Promise<User> {
        return this.db.users.upsert({
            where: { id: user.id },
            create: user,
            update: user
        });
    }

    async delete(id: string): Promise<void> {
        await this.db.users.delete({ where: { id } });
  }
}
'''
    file_path = "src/repositories/user.repository.ts"

    # Extract hierarchy
    result = _extract_hierarchical_structure_treesitter(source_code, "typescript", file_path)

    # Verify output structure
    assert isinstance(result, dict)
    assert result["file"] == file_path
    assert result["language"] == "typescript"
    assert "blocks" in result
    assert "order" in result
    assert "stats" in result

    # Verify blocks
    blocks = result["blocks"]
    assert len(blocks) == 2  # Interface and class

    # Verify interface
    interface = next(b for b in blocks if b["type"] == "interface")
    assert interface["name"] == "Repository"
    assert interface["content"].startswith("interface Repository")
    assert interface["start_line"] == 2
    assert interface["end_line"] == 6
    assert len(interface["methods"]) == 3
    assert {m["name"] for m in interface["methods"]} == {"find", "save", "delete"}

    # Verify class
    class_block = next(b for b in blocks if b["type"] == "class")
    assert class_block["name"] == "UserRepository"
    assert class_block["content"].startswith("@Injectable()")
    assert class_block["start_line"] == 8
    assert class_block["end_line"] == 27
    assert len(class_block["methods"]) == 3
    assert {m["name"] for m in class_block["methods"]} == {"find", "save", "delete"}
    assert class_block["decorators"] == ["Injectable"]
    assert class_block["metadata"]["visibility"] == "public"

    # Verify order preserves source code ordering
    assert result["order"] == ["Repository", "UserRepository"]

    # Verify stats
    assert result["stats"]["total_blocks"] == 8  # Interface + Class + 6 methods
    assert result["stats"]["by_type"] == {
        "interface": 1,
        "class": 1,
        "method": 6
    }

def test_class_hierarchy_with_nested_structure():
    """Test extraction of class with nested structure."""
    source_code = '''
class UserService {
    private static readonly DEFAULT_ROLE = 'user';
    #apiKey: string;

    constructor(apiKey: string) {
        this.#apiKey = apiKey;
    }

    static createAdmin(): User {
        return new User({ role: 'admin' });
    }

    async findById(id: string): Promise<User | null> {
        try {
            const user = await this.db.findUnique({ id });
            return user ? new User(user) : null;
        } catch {
            return null;
        }
    }

    private validateUser(user: User): boolean {
        return user.role === UserService.DEFAULT_ROLE || user.role === 'admin';
    }
}
'''
    file_path = "src/services/user.service.ts"

    # Extract hierarchy
    result = _extract_hierarchical_structure_treesitter(source_code, "typescript", file_path)

    # Verify output structure
    assert isinstance(result, dict)
    assert result["file"] == file_path
    assert result["language"] == "typescript"

    # Verify class block
    blocks = result["blocks"]
    assert len(blocks) == 1  # One class

    class_block = blocks[0]
    assert class_block["name"] == "UserService"
    assert class_block["type"] == "class"
    assert len(class_block["methods"]) == 4  # constructor + 3 methods

    # Verify methods
    methods = class_block["methods"]
    method_names = {m["name"] for m in methods}
    assert method_names == {"constructor", "createAdmin", "findById", "validateUser"}

    # Verify method metadata
    find_by_id = next(m for m in methods if m["name"] == "findById")
    assert find_by_id["metadata"]["async"] == True
    assert find_by_id["metadata"]["visibility"] == "public"

    validate_user = next(m for m in methods if m["name"] == "validateUser")
    assert validate_user["metadata"]["visibility"] == "private"

    create_admin = next(m for m in methods if m["name"] == "createAdmin")
    assert create_admin["metadata"]["static"] == True

    # Verify stats
    assert result["stats"]["total_blocks"] == 5  # 1 class + 4 methods
    assert result["stats"]["by_type"] == {
        "class": 1,
        "method": 4
    }

if __name__ == "__main__":
    pytest.main([__file__]) 