# Implementation Lessons and Best Practices

This document captures key learnings and best practices from implementing the QA Generation Module.

## Test-Driven Development Lessons

1. **Write Tests First, Always**
   - Start with the test, not the implementation
   - Tests clarify requirements and edge cases before coding begins
   - Tests provide immediate feedback on implementation correctness

2. **Run All Tests Before Making Changes**
   - Always run existing tests to establish a baseline
   - Run tests after each significant change
   - Verify that new code doesn't break existing functionality

3. **Fix Issues at the Source**
   - Identify the root cause of bugs, not just symptoms
   - Fix issues in the original location, not with workarounds downstream
   - Look for the earliest point in the pipeline where the error manifests

## Code Organization Principles

1. **Avoid Initialization Order Issues**
   - Define functions before using them
   - Be aware of circular dependencies
   - Place constants and imports at the top of files

2. **Guard Against Duplication**
   - Check for duplicate function names across modules
   - Watch for overlapping variable names that might cause confusion
   - Use namespaces and packages to isolate functionality

3. **Explicit Dependencies**
   - The project uses `uv` instead of `pip` for package management
   - Document all dependencies in `pyproject.toml`
   - Use explicit imports to make dependencies clear

## Implementation Gotchas

1. **HTML Sanitization Behavior**
   - Bleach preserves text content inside HTML tags when sanitizing
   - Be careful with assumptions about sanitized content
   - Test sanitized output thoroughly, don't just check for tag removal

2. **Prompt Injection Detection Complexity**
   - Simple word matching produces many false positives
   - Context-aware detection improves accuracy
   - Balance security needs with legitimate use cases

3. **Environment Variables in Configuration**
   - Always validate configuration values after loading
   - Support environment variable substitution for flexibility
   - Provide sensible defaults for missing configuration

4. **Temperature Iteration & Context Isolation**
   - Each temperature setting must have isolated context
   - Use fresh generation context to prevent cross-contamination
   - Concurrent generation requires proper rate limiting
   - Semaphores are essential to prevent API rate limit errors
   - Higher temperatures generally produce more creative but less reliable results

5. **Bidirectional Generation Strategy**
   - Reversing QA pairs (creating questions from answers) enhances coverage
   - Highest confidence forward pairs produce the best reverse pairs
   - Reverse pairs should use higher temperatures for creativity
   - Always verify that reversed questions differ from originals
   - Different question formulations improve search/retrieval effectiveness

6. **Circuit Breaker Pattern Implementation**
   - State tracking requires careful consideration for concurrency
   - Ensure state transitions are atomic to prevent race conditions
   - Use timestamps for time-based reset logic to ensure proper healing
   - Consider persistent state for multi-process environments
   - Test both opening and resetting of the circuit

7. **Cost-Aware Routing**
   - Content length alone is not always the best indicator for model selection
   - Consider additional factors like task complexity and required capabilities
   - Keep model selection logic separate from retry logic for better maintainability
   - Allow for configuration of thresholds to adjust routing based on real-world feedback
   - Track and log model choices to enable optimization and cost analysis

8. **Cache Implementation**
   - Use deterministic key generation for consistent lookup
   - Include all relevant parameters in cache key calculations
   - Consider both in-memory and disk-based caching options
   - Implement graceful fallbacks for cache failures
   - Track and log cache hit rates to measure effectiveness
   - Use dependency injection for testing cache functionality
   - Remember to restore original cache state after tests

9. **Batch Processing and Worker Pools**
   - Use semaphores to control concurrent execution
   - Avoid nested semaphores to prevent deadlocks
   - Manage worker pools at the appropriate abstraction level
   - Use asyncio.gather with return_exceptions=True for error handling
   - Track performance metrics to identify bottlenecks
   - Chunk large datasets to prevent memory issues
   - Consider system resource constraints when setting worker counts
   - Implement graceful degradation for overloaded conditions
   - Log start and completion times for performance analysis
   - Calculate worker utilization to optimize resource usage
   - Respect configuration constraints during batch processing
   - Provide detailed performance statistics for optimization

10. **Monitoring System Design**
   - Provide fallback mechanisms for environments without Prometheus
   - Use context managers for timing operations automatically
   - Separate metric recording from metric collection infrastructure
   - Allow for multiple metric types (counters, gauges, histograms)
   - Include detailed labels for better filtering and analysis
   - Track both success/failure counts and timing measurements
   - Implement alerting with appropriate thresholds
   - Integrate monitoring with all major components (cache, processing)
   - Make monitoring optional but enabled by default
   - Store metrics in a thread-safe manner

11. **Constraint Tracking and Enforcement**
   - Validate configuration constraints at loading time
   - Enforce runtime constraints during processing
   - Test constraint boundaries specifically, not just happy paths
   - Apply constraints at appropriate levels of abstraction
   - Document constraints clearly in both code and documentation
   - Use Pydantic validators for model-level constraints
   - Apply business logic constraints in service layer
   - Maintain constraint consistency across multiple operations

## Testing Principles

1. **Tests as Relationship Documentation**
   - Tests should document relationships between components
   - Focus on core functionality, not edge cases
   - Use tests to reason about system behavior, not just to "pass"
   - Tests should serve as executable documentation
   - The goal is understanding the system, not 100% coverage

2. **Test Core Paths First**
   - Prioritize testing the main functionality paths
   - Avoid overinvesting in remote edge cases
   - Test the 80% common use cases thoroughly
   - Add edge case tests only when they clarify important constraints

3. **Use Clear Test Names**
   - Tests should describe what they're testing
   - Follow a consistent naming convention
   - Group related tests together
   - Names should explain intent, not implementation details
   
4. **Test Concurrent Operations**
   - Use mocks to simulate concurrent execution
   - Check for race conditions and deadlocks
   - Verify concurrency limits are enforced
   - Test both success and failure paths in parallel operations

5. **Module Constant Patching Issues**
   - Be aware of how modules are imported and accessed
   - When importing `from module import function`, you can't access module-level constants through the function
   - Instead, use `import module as module_name` to access module-level constants
   - For patching constants, modify the module directly rather than using `unittest.mock.patch`
   - Consider defining constants in a separate configuration module for easier access and testing

## Security Considerations

1. **Input Validation**
   - Sanitize all external inputs
   - Validate against schema before processing
   - Check for malicious patterns (injections, XSS)

2. **PII Detection**
   - Identify and handle personally identifiable information
   - Balance detection strictness with false positives
   - Log appropriately without exposing sensitive data

3. **Environment Protection**
   - Use environment variables for sensitive configuration
   - Never hardcode credentials
   - Validate all environment variables before use

These lessons have been incorporated into the implementation and should be followed for all future development.