/**
 * Sample JavaScript file with various code structures for extraction testing.
 * Includes functions, classes, and React components.
 */

// Utility function to calculate sum
function calculateSum(a, b) {
  /**
   * Calculates the sum of two numbers
   * @param {number} a - First number
   * @param {number} b - Second number
   * @returns {number} Sum of a and b
   */
  return a + b;
}

// Arrow function example
const multiply = (a, b) => {
  /**
   * Multiplies two numbers
   * @param {number} a - First number
   * @param {number} b - Second number
   * @returns {number} Product of a and b
   */
  return a * b;
};

// JavaScript class
class Person {
  /**
   * Represents a person
   * @constructor
   * @param {string} name - Person's name
   * @param {number} age - Person's age
   */
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  
  /**
   * Get a greeting from the person
   * @returns {string} A greeting message
   */
  greet() {
    return `Hello, my name is ${this.name} and I am ${this.age} years old.`;
  }
  
  /**
   * Static method to create a person with default age
   * @param {string} name - Person's name
   * @returns {Person} A person with the given name and age 30
   */
  static createDefault(name) {
    return new Person(name, 30);
  }
}

// Inheritance example
class Employee extends Person {
  /**
   * Represents an employee
   * @constructor
   * @param {string} name - Employee's name
   * @param {number} age - Employee's age
   * @param {string} title - Employee's job title
   */
  constructor(name, age, title) {
    super(name, age);
    this.title = title;
  }
  
  /**
   * Get a professional greeting from the employee
   * @override
   * @returns {string} A professional greeting
   */
  greet() {
    return `Hello, I'm ${this.name}, a ${this.title}.`;
  }
}

// React component example
const Button = ({ onClick, children, disabled = false }) => {
  /**
   * A simple button component
   * @param {Function} onClick - Click handler
   * @param {React.ReactNode} children - Button content
   * @param {boolean} disabled - Whether button is disabled
   */
  return (
    <button 
      onClick={onClick} 
      className="button"
      disabled={disabled}
    >
      {children}
    </button>
  );
};

// React class component
class Counter extends React.Component {
  /**
   * A counter component
   * @constructor
   * @param {object} props - Component props
   */
  constructor(props) {
    super(props);
    this.state = {
      count: 0
    };
  }
  
  /**
   * Increment the counter
   */
  increment = () => {
    this.setState(prevState => ({
      count: prevState.count + 1
    }));
  }
  
  /**
   * Render the component
   * @returns {React.ReactNode} The rendered component
   */
  render() {
    return (
      <div>
        <h3>Count: {this.state.count}</h3>
        <Button onClick={this.increment}>
          Increment
        </Button>
      </div>
    );
  }
}

// Module exports
export {
  calculateSum,
  multiply,
  Person,
  Employee,
  Button,
  Counter
}; 