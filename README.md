# Patchwork 🔧

An intelligent code debugging agent that automatically fixes broken Python functions using LLM-powered iterative refinement.

## Overview

Patchwork is a Python framework for automated code debugging and repair. It combines large language models with dynamic tool execution to iteratively analyze, test, and fix broken code. The agent can handle complex debugging scenarios by using multiple tools like linters, test harnesses, and code analysis utilities.

## Features

- 🤖 **LLM-Powered Debugging**: Uses GPT-4.1 series models for intelligent code analysis
- 🔄 **Iterative Refinement**: Automatically iterates through fix attempts until success
- 🛠️ **Tool Integration**: Extensible tool system (linter, test harness, debugger, etc.)
- 📊 **Multi-Level Evaluation**: Comprehensive evaluation framework with syntax, execution, and semantic scoring
- 🎯 **Best-of-N Sampling**: Generate multiple solutions and pick the best one
- ⚡ **Flexible Configuration**: Support for different models, temperatures, and iteration limits

## Installation

```bash
# Clone the repository
git clone https://github.com/tejaskhot/patchwork.git
cd patchwork

# Install dependencies
pip install -e .

# Set up your API key
export OPENAI_API_KEY="your_api_key_here"
```

## Quick Start

### Basic Usage

```python
from agent import create_agent, ProblemContext

# Create an agent
agent = create_agent(
    model="gpt-4.1-nano",  # or gpt-4.1-mini, gpt-4.1
    max_iterations=5,
    temperature=0.1
)

# Define your problem
problem = ProblemContext(
    entry_point="fibonacci",
    goal="Fix the fibonacci function to be efficient",
    quality_criteria="Must handle n=30 in under 1 second",
    tests_formatted="fibonacci(5) should return 5, fibonacci(10) should return 55",
    broken_code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Too slow!
"""
)

# Run the agent
solution = agent.run(problem)
print(solution)
```

### Command Line Interface

```bash
# Run experiments with different models
python run_patchwork.py --problem max_list --model gpt-4.1-nano
python run_patchwork.py --problem filter_top_students --model gpt-4.1-mini

# List available test problems
python run_patchwork.py --list-problems

# Run with custom parameters
python run_patchwork.py --problem fibonacci --model gpt-4.1 --max-iterations 10
```

### Best-of-N Sampling

```python
# High temperature for diverse generation
generator = create_agent(model="gpt-4.1-mini", temperature=0.8)

# Low temperature for consistent evaluation
judge = create_agent(model="gpt-4.1-mini", temperature=0.1)

# Generate multiple candidates
candidates = []
for i in range(3):
    solution = generator.run(problem)
    candidates.append(solution)

# Judge picks the best one
best_solution = judge.run(evaluation_problem)
```

## Model Options

### GPT-4.1 Series (Recommended)

- `gpt-4.1-nano` - Smallest, fastest, most cost-effective
- `gpt-4.1-mini` - Balanced performance and cost
- `gpt-4.1` - Most capable, highest quality

### Other Supported Models

- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `deepseek/deepseek-coder-v2`
- `gemini/gemini-1.5-pro-latest`

## Architecture

### Core Components

- **PatchworkAgent**: Main agent class that orchestrates the debugging process
- **ToolRegistry**: Manages available tools and their execution
- **ProblemContext**: Structured representation of debugging problems
- **Evaluators**: Multi-level evaluation system for solution quality

### Available Tools

- **Linter**: Static code analysis and syntax checking
- **TestHarness**: Dynamic code execution and testing
- **Debugger**: Advanced debugging and error analysis
- **PlotInspector**: Visualization and analysis utilities

### Evaluation Framework

1. **Level 1**: Syntax and basic correctness
2. **Level 2**: Execution and test case validation
3. **Level 3**: Semantic quality and LLM-based evaluation

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"  # Optional, for Claude models
```

### Custom Tool Registry

```python
from tools.registry import ToolRegistry
from tools.linter import LinterTool

# Create custom registry
registry = ToolRegistry()
registry.register("linter", LinterTool())
registry.register("custom_tool", YourCustomTool())

# Use with agent
agent = PatchworkAgent(tool_registry=registry)
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic fibonacci optimization
- Custom tool registry usage
- Best-of-N sampling implementation
- Error handling and logging

## Evaluation

Run the evaluation suite:

```python
from evals import PatchworkEvaluator

evaluator = PatchworkEvaluator()
results = evaluator.evaluate_solution(problem, solution)
print(f"Overall Score: {results.overall_score}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code formatting with Black
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
