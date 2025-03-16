// Sample TypeScript file with types, interfaces, functions and classes for testing

// Type definitions
type StringOrNumber = string | number;
type UserRole = 'admin' | 'user' | 'guest';

// Interface definitions
interface User {
    id: number;
    name: string;
    email: string;
    role: UserRole;
    lastLogin?: Date;
}

// Function with type annotations
function greet(name: string): string {
    return `Hello, ${name}!`;
}

// Arrow function with type annotations
const add = (a: number, b: number): number => {
    return a + b;
};

// Class with interfaces and type annotations
class UserManager {
    private users: User[];
    
    constructor() {
        this.users = [];
    }
    
    addUser(user: User): void {
        this.users.push(user);
    }
    
    getUserById(id: number): User | undefined {
        return this.users.find(user => user.id === id);
    }
    
    getAllUsers(): User[] {
        return [...this.users];
    }
}

// Generic function
function wrapInArray<T>(item: T): T[] {
    return [item];
}


// Function with multiple newlines between blocks to test splitting



function multiply(a: number, b: number): number {
    return a * b;
}

// Export for module systems
export {
    User,
    UserRole,
    greet,
    add,
    UserManager,
    wrapInArray,
    multiply
}; 