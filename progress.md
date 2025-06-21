# Our Working Style: The Core Principles

1. Step-by-Step, Incremental Implementation:

We will build this project one logical step at a time. Each of your responses should focus on a single, small, and complete component. For example, "Create the test_harness.py tool" is a great step. "Build the entire tools directory" is too large.
I want to review short, clean diffs. This is the most important rule. It allows me to understand every change and learn from the process.
2. Explain Your Reasoning (Teach Me):

Before you write any code, briefly explain your design choices for the upcoming step. For example, if you're about to write test_harness.py, you might say: "For the test harness, I'll use Python's subprocess module instead of exec() to ensure we have a secure sandbox. This prevents the agent's generated code from accessing the filesystem. I'll also wrap the output in a Pydantic model for type-safe, structured data."
I want to understand the "why" behind the code.
3. Modern Best Practices are Non-Negotiable:

Pydantic for Data Structures: We will use Pydantic extensively. All data structures, especially the TestCase JSON format, tool inputs/outputs, and LLM responses, should be defined with Pydantic models. This ensures data validation and provides a clear schema.
Strict Type Hinting: All functions and methods must have clear type hints.
Modular and Clean Code: Adhere to SOLID principles where appropriate. Code should be easy to read and test.
4. Frequent Commits for a Clean History:

After you complete a logical step and I approve the code, I will ask you to commit the changes with a clear, conventional commit message. For example: feat(tools): implement initial test_harness or fix(agent): correct prompt formatting.
Project Management

1. Maintaining progress.md:

We will maintain a progress.md file as our shared project checklist. I want you to create this file in our first step. It should have three sections: ## To Do, ## In Progress, and ## Done.
After every commit, you must update progress.md to reflect the new state of the project.

# Patchwork Project Progress

## To Do

- [ ] **Phase 2:** Source Your Data (`test_cases/`)
- [ ] **Phase 3:** Build the ReAct Agent (`agent.py`)
- [ ] **Phase 4:** Craft Your ReAct Prompt (`prompts/react_prompt.txt`)
- [ ] **Phase 5:** Implement Multi-Layered Evaluations (`evals.py`)

## In Progress

- [ ] **Phase 1:** Build the Developer's Toolkit (`tools/`)
  - [x] `test_harness.py`
  - [x] `linter.py`
  - [x] `debugger.py`

## Done
