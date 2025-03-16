// Original file: temp_sample.javascript
// Block type: class
// Name: Calculator

class Calculator {
  constructor() {
    this.value = 0;
  }

  add(x) {
    this.value += x;
    return this.value;
  }

  subtract(x) {
    this.value -= x;
    return this.value;
  }

  multiply(x) {
    this.value *= x;
    return this.value;
  }

  divide(x) {
    if (x === 0) throw new Error("Division by zero");
    this.value /= x;
    return this.value;
  }
}