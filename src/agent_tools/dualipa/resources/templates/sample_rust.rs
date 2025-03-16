// Sample Rust file with various functions, structs, traits, and implementations for testing

use std::fmt;

// A simple struct with fields
struct Person {
    name: String,
    age: u32,
}

// Implementation for Person
impl Person {
    // Constructor function
    fn new(name: &str, age: u32) -> Person {
        Person {
            name: String::from(name),
            age,
        }
    }
    
    // Method to get the name
    fn name(&self) -> &str {
        &self.name
    }
    
    // Method to get the age
    fn age(&self) -> u32 {
        self.age
    }
}

// Implement the Display trait for Person
impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Person {{ name: {}, age: {} }}", self.name, self.age)
    }
}


// Function with multiple newlines to test block splitting



// A simple trait definition
trait Greeter {
    fn greet(&self) -> String;
}

// Implement the Greeter trait for Person
impl Greeter for Person {
    fn greet(&self) -> String {
        format!("Hello, my name is {} and I am {} years old", self.name, self.age)
    }
}

// A generic function
fn calculate_area<T: std::ops::Mul<Output = T> + Copy>(radius: T, pi: T) -> T {
    pi * radius * radius
}

// Main function
fn main() {
    // Create a person
    let person = Person::new("Alice", 30);
    
    // Print person information
    println!("{}", person);
    
    // Print greeting
    println!("{}", person.greet());
    
    // Calculate and print circle area
    let area = calculate_area(5.0, 3.14159);
    println!("Area of circle: {:.2}", area);
} 