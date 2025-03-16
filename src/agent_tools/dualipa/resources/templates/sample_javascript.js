// Sample JavaScript file with multiple functions and classes for testing

// Regular function declaration
function greet(name) {
    console.log(`Hello, ${name}!`);
    return `Hello, ${name}!`;
}

// Arrow function
const add = (a, b) => {
    // Adding two numbers
    return a + b;
};

// Class declaration
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    sayHello() {
        return `Hello, my name is ${this.name}`;
    }
    
    getAge() {
        return this.age;
    }
}

// Function with multiple newlines between blocks to test splitting



function multiply(a, b) {
    return a * b;
}

// Immediately invoked function expression (IIFE)
(function() {
    console.log('IIFE executed');
})();

// Export for module systems
if (typeof module !== 'undefined') {
    module.exports = {
        greet,
        add,
        Person,
        multiply
    };
} 