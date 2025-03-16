// Sample Go file with various functions, structs, and interfaces for testing

package main

import (
	"fmt"
	"time"
)

// Person represents a human with a name and age
type Person struct {
	Name string
	Age  int
}

// Greeter defines something that can greet
type Greeter interface {
	Greet() string
}

// Implement the Greeter interface for Person
func (p Person) Greet() string {
	return fmt.Sprintf("Hello, my name is %s and I am %d years old", p.Name, p.Age)
}

// NewPerson creates a new person with the given name and age
func NewPerson(name string, age int) Person {
	return Person{
		Name: name,
		Age:  age,
	}
}

// Function with multiple newlines to test block splitting

// CalculateArea returns the area of a circle
func CalculateArea(radius float64) float64 {
	return 3.14159 * radius * radius
}

// SimpleStruct with minimal fields
type SimpleStruct struct {
	ID   int
	Name string
}

// main is the entry point of the program
func main() {
	// Create a new person
	person := NewPerson("Alice", 30)
	
	// Print the greeting
	fmt.Println(person.Greet())
	
	// Calculate and print the area of a circle
	area := CalculateArea(5.0)
	fmt.Printf("Area of circle: %.2f\n", area)
	
	// Current time
	fmt.Println("Current time:", time.Now())
} 