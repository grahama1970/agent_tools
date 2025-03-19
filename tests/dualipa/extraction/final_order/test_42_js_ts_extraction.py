"""
TEST EXPECTATIONS

test_extract_js_blocks:
Input: JavaScript file with functions and classes
Expected Output:
{
    "code_blocks": > 0,
    "total_files": 1,
    "file_blocks": {
        "example.js": [
            {
                "block_type": "function",
                "name": "greet",
                "content": "function greet(name)..."
            },
            {
                "block_type": "class",
                "name": "Person",
                "content": "class Person..."
            }
        ]
    }
}

test_extract_ts_blocks:
Input: TypeScript file with interfaces and classes
Expected Output:
{
    "code_blocks": > 0,
    "total_files": 1,
    "file_blocks": {
        "example.ts": [
            {
                "block_type": "interface",
                "name": "User",
                "content": "interface User..."
            },
            {
                "block_type": "class",
                "name": "UserService",
                "content": "class UserService..."
            }
        ]
    }
}

CRITICAL RULES:
1. Block Extraction Rules:
   - Each block must have a block_type
   - Each block must have a name
   - Each block must have content
   - Each block must preserve original formatting

2. Stats Tracking Rules:
   - Track total files processed
   - Track blocks per file
   - Track languages encountered
   - Track errors during extraction

3. Output File Rules:
   - All blocks must be written to output directory
   - All paths must be relative to output directory
   - Block files must have .js or .ts extension
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys
import shutil

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.code_extractor import (
        _extract_js_ts_blocks,
        _verify_code_block,
        _save_stats_to_json,
        initialize_stats_dict
    )
    IMPORTS_AVAILABLE = True
except ImportError as import_error:
    import traceback
    print(f"IMPORT ERROR: {import_error}")
    print("Traceback:")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required code extractor modules not available")

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    yield tmp_path
    shutil.rmtree(tmp_path)

@pytest.fixture
def stats_dict():
    """Initialize a stats dictionary."""
    return {
        "total_files": 0,
        "code_files": 0,
        "documentation_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "languages": {},
        "file_types": {},
        "file_blocks": {},
        "errors": []
    }

@pytest.fixture
def javascript_file_fixture(temp_dir):
    """Create a JavaScript file with known functions and classes for testing."""
    js_file = Path(temp_dir) / "example.js"
    content = """
/**
 * A simple JavaScript function
 * @param {string} name - The name to greet
 * @returns {string} A greeting message
 */
function greet(name) {
    return `Hello, ${name}!`;
}

/**
 * A Person class
 */
class Person {
    /**
     * Create a new Person
     * @param {string} name - The person's name
     * @param {number} age - The person's age
     */
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    /**
     * Get a greeting for this person
     * @returns {string} A personalized greeting
     */
    getGreeting() {
        return greet(this.name);
    }
    
    /**
     * Check if the person is an adult
     * @returns {boolean} True if the person is an adult
     */
    isAdult() {
        return this.age >= 18;
    }
}

// A constant
const MAX_AGE = 120;

// Export the functions and classes
module.exports = {
    greet,
    Person,
    MAX_AGE
};
"""
    
    js_file.write_text(content)
    return js_file

@pytest.fixture
def typescript_file_fixture(temp_dir):
    """Create a TypeScript file with interfaces and classes for testing."""
    ts_file = Path(temp_dir) / "example.ts"
    content = """
interface User {
    id: string;
    name: string;
    age: number;
    email?: string;
}

interface UserRepository {
    find(id: string): Promise<User>;
    save(user: User): Promise<void>;
    delete(id: string): Promise<void>;
}

class UserService implements UserRepository {
    private users: Map<string, User>;

    constructor() {
        this.users = new Map();
    }

    async find(id: string): Promise<User> {
        const user = this.users.get(id);
        if (!user) {
            throw new Error(`User not found: ${id}`);
        }
        return user;
    }

    async save(user: User): Promise<void> {
        this.users.set(user.id, user);
    }

    async delete(id: string): Promise<void> {
        this.users.delete(id);
    }

    private validateUser(user: User): boolean {
        return user.name.length > 0 && user.age >= 0;
    }
}

// Export types and class
export { User, UserRepository, UserService };
"""
    
    ts_file.write_text(content)
    return ts_file

def test_extract_js_blocks(temp_dir, stats_dict):
    """Test JavaScript block extraction."""
    # Create a test file
    test_file = temp_dir / "test.js"
    content = """
function greet() {
    console.log("Hello");
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    sayHello() {
        console.log(`Hello, ${this.name}!`);
    }
}

const add = (a, b) => a + b;
"""
    test_file.write_text(content)
    
    # Extract blocks
    blocks = _extract_js_ts_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict
    )
    
    assert blocks > 0
    assert stats_dict["code_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check block files were created
    blocks_dir = temp_dir / "code_blocks" / "javascript"
    assert blocks_dir.exists()
    assert len(list(blocks_dir.glob("*.js"))) > 0

def test_extract_ts_blocks(temp_dir, stats_dict):
    """Test TypeScript block extraction."""
    # Create a test file
    test_file = temp_dir / "test.ts"
    content = """
interface Person {
    name: string;
    age: number;
}

class Employee implements Person {
    constructor(
        public name: string,
        public age: number,
        private salary: number
    ) {}
    
    getSalary(): number {
        return this.salary;
    }
}

function processEmployee(emp: Employee): void {
    console.log(`Processing ${emp.name}`);
}
"""
    test_file.write_text(content)
    
    # Extract blocks
    blocks = _extract_js_ts_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict
    )
    
    assert blocks > 0
    assert stats_dict["code_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check block files were created
    blocks_dir = temp_dir / "code_blocks" / "typescript"
    assert blocks_dir.exists()
    assert len(list(blocks_dir.glob("*.ts"))) > 0

def test_verify_js_ts_blocks(temp_dir, stats_dict):
    """Test verification of JavaScript and TypeScript blocks."""
    # Valid JavaScript block
    js_block = {
        "content": "function test() { return true; }",
        "language": "javascript"
    }
    assert _verify_code_block(js_block) is True
    
    # Invalid JavaScript block
    invalid_js = {
        "content": "function test( { return true; }",  # Missing parenthesis
        "language": "javascript"
    }
    assert _verify_code_block(invalid_js) is False
    
    # Valid TypeScript block
    ts_block = {
        "content": "interface Test { prop: string; }",
        "language": "typescript"
    }
    assert _verify_code_block(ts_block) is True
    
    # Invalid TypeScript block
    invalid_ts = {
        "content": "interface Test { prop string }",  # Missing colon
        "language": "typescript"
    }
    assert _verify_code_block(invalid_ts) is False

def test_extract_js_ts_blocks_error_handling(temp_dir, stats_dict):
    """Test error handling in JS/TS block extraction."""
    # Create an invalid file path
    invalid_file = temp_dir / "nonexistent.js"
    
    # Try to extract blocks
    blocks = _extract_js_ts_blocks(
        invalid_file,
        "some content",
        temp_dir,
        stats_dict
    )
    
    assert blocks == 0
    assert len(stats_dict["errors"]) > 0

def test_extract_tsx_component(temp_dir):
    """Test extracting React component from TSX file."""
    tsx_file = Path(temp_dir) / "ListItem.tsx"
    tsx_content = """
import React from 'react';

interface ListItemProps {
    title: string;
    description?: string;
    onSelect?: () => void;
}

const ListItem: React.FC<ListItemProps> = ({
    title,
    description,
    onSelect
}) => {
    const handleClick = () => {
        if (onSelect) {
            onSelect();
        }
    };

    return (
        <div className="list-item" onClick={handleClick}>
            <h3>{title}</h3>
            {description && <p>{description}</p>}
        </div>
    );
};

export default ListItem;
"""
    tsx_file.write_text(tsx_content)
    
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=tsx_file, output_dir=output_dir)
    
    num_blocks = _extract_js_ts_blocks(tsx_file, tsx_content, output_dir, stats, "typescript")
    
    # Verify block count
    assert num_blocks > 0, "No blocks were extracted"
    
    # Verify extracted blocks
    blocks = stats["file_blocks"][str(tsx_file)]
    
    # Verify interface blocks
    interface_blocks = [b for b in blocks if b.get("block_type") == "interface"]
    assert len(interface_blocks) == 1, "Expected 1 interface"
    assert interface_blocks[0].get("name") == "ListItemProps", "ListItemProps interface not found"
    
    # Verify component blocks
    component_blocks = [b for b in blocks if b.get("block_type") == "component"]
    assert len(component_blocks) == 1, "Expected 1 component"
    assert component_blocks[0].get("name") == "ListItem", "ListItem component not found"
    
    # Verify output files
    blocks_dir = output_dir / "blocks" / "code" / "typescript"
    assert blocks_dir.exists(), "Blocks directory not created"
    block_files = list(blocks_dir.glob("*.tsx"))
    assert len(block_files) == num_blocks, "Number of block files doesn't match block count"

if __name__ == "__main__":
    pytest.main([__file__]) 