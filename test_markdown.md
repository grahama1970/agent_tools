# Test Markdown File

This is a sample markdown file to test our parser's ability to handle real files.

## Code Examples

Here are some code examples in different languages:

```python
# Python example
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial(5))  # Output: 120
```

```javascript
// JavaScript example
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n-1) + fibonacci(n-2);
}

console.log(fibonacci(10)); // Output: 55
```

## Edge Cases

### Empty Code Block

```

```

### Code Block With No Language

```
This is a code block with no language specified
```

## Lists and Tables

Here's a list:
- Item 1
- Item 2
- Item 3

And a table:

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   | 