
# Instructions to Claude Code
## Files
parent directory: /home/grahama/workspace/experiments/agent_tools

# How to run a pytest:
 clear;uv run pytest src/agent_tools/cursor_rules/tests/memory/test_agent_memory.py::test_domain_filtering_simple -v

## Documentation
issues with agent_memory: src/agent_tools/cursor_rules/core/memory/docs/AGENT_MEMORY_PROBLEMS.md     
lessons learned on agent_tools project: .cursor/LESSONS_LEARNED.md  
code  documentationreferences:
- https://docs.arangodb.com/3.12/
- https://docs.python-arango.com/en/main/

code to fix: src/agent_tools/cursor_rules/core/memory/agent_memory.py

pytest to pass: src/agent_tools/cursor_rules/tests/memory/test_agent_memory.py::test_domain_filtering_simple -v


## Instructons:
Can you look at the markdown file and then look at the code to see what is failing. 
Ask clarifying questions of the human which I will answer.
My hunch (as the human) is that ArangoDB documentaion is not being used over outdated model training
Then run the test on the amended agent_memory.py file to see what is failing. Do not iterate over 3 times on running the test_domain_filtering_simple test. I want to see if you solve the problem without going into an endless feedback loop.

Confirm that you  understand and proceed