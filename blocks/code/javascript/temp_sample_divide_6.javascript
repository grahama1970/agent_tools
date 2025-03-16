// Original file: temp_sample.javascript
// Block type: function
// Name: divide

divide(x) {
    if (x === 0) throw new Error("Division by zero");
    this.value /= x;
    return this.value;
  }