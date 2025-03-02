# Method Validator Task

## Purpose

As an AI assistant, I need method_validator to:
1. Prevent hallucination of non-existent methods when suggesting code
2. Discover existing functionality to avoid duplicating code
3. Validate method signatures, parameters, and behaviors before making suggestions
4. Ground my responses in actual package documentation rather than training data

## Test Case

The optimal test should demonstrate my ability to:

1. Recognize when I need method validation:
   - When working with unfamiliar third-party packages
   - Before suggesting implementations that might duplicate existing functionality
   - When unsure about method signatures or parameters

2. Use the CLI effectively:
   ```bash
   # First, list all available methods to understand package capabilities
   method-validator litellm --list-all
   
   # Then, deep analyze specific methods of interest
   method-validator litellm --method completion
   ```

3. Apply the discovered information:
   ```python
   # WRONG - Don't suggest code without validating methods
   client.send_chat(messages=[{"role": "user", "content": "Hello"}])  # Hallucinated method!

   # RIGHT - First validate method exists and check signature
   # method-validator litellm --method completion shows:
   # - Method exists as completion()
   # - Requires 'model' and 'messages' parameters
   # - Returns ModelResponse or CustomStreamWrapper
   response = litellm.completion(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello"}]
   )
   ```

## Success Criteria

1. Zero hallucinated methods in my suggestions
2. Accurate parameter usage based on actual method signatures
3. Leveraging existing functionality instead of suggesting duplicates
4. Clear understanding of method behaviors and potential exceptions

## Example Test Conversation

User: "I want to use litellm to send a chat message"

Assistant (WRONG):
"Here's how to send a chat message with litellm:
```python
litellm.send_chat("Hello")  # WRONG - Hallucinated method!
```"

Assistant (RIGHT):
"Let me check the available methods in litellm:
```bash
method-validator litellm --list-all
```
I see there's a completion() method. Let me check its details:
```bash
method-validator litellm --method completion
```
Based on the actual documentation, here's how to send a chat message:
```python
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```"

## Implementation Notes

1. Always check available methods before making suggestions
2. Validate method signatures and parameters
3. Use actual documentation over training data
4. Be explicit about required parameters and return types
5. Consider potential exceptions and error cases
