#!/usr/bin/env python3
"""
Example usage of the PatchworkAgent.

This script demonstrates how to use the agent with proper error handling,
logging, and advanced features like dependency injection and validation.
"""

import logging
import os

from agent import PatchworkAgent, ProblemContext, create_agent
from tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_fibonacci_optimization():
    """Example 1: Optimize a slow recursive fibonacci function."""

    broken_function = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # This will be slow for large n
    """

    test_cases = """
Test cases:
- fibonacci(0) should return 0
- fibonacci(1) should return 1  
- fibonacci(5) should return 5
- fibonacci(10) should return 55
- Should handle n=30 efficiently (under 1 second)
"""

    # Create problem context
    try:
        problem = ProblemContext(
            entry_point="fibonacci",
            goal="Fix the fibonacci function to be efficient for large inputs",
            quality_criteria="Must be fast enough to handle n=30 in under 1 second",
            tests_formatted=test_cases,
            broken_code=broken_function,
        )
        logger.info("Problem context created successfully")
    except ValueError as e:
        logger.error(f"Invalid problem context: {e}")
        return

    # Create agent - try different model sizes for different complexity needs
    agent = create_agent(
        model="gpt-4.1-nano",  # Use gpt-4.1-mini or gpt-4.1 for more complex problems
        max_iterations=5,
        temperature=0.1,
        max_tokens=3000,
        message_window_size=20,
    )

    logger.info("Agent created successfully")

    # Run the debugging session
    logger.info("Starting debugging session...")
    try:
        result = agent.run(problem)

        # Check result
        if agent.run_log.status == "success":
            logger.info("✅ Debugging successful!")
            print("\n" + "=" * 50)
            print("FIXED CODE:")
            print("=" * 50)
            print(result)
            print("=" * 50)
        else:
            logger.warning(f"⚠️ Debugging failed with status: {agent.run_log.status}")
            print(f"\nResult: {result}")

        # Print session statistics
        run_log = agent.get_run_log()
        print(f"\nSession Stats:")
        print(f"- Total steps: {len(run_log.steps)}")
        print(
            f"- Total tool calls: {sum(len(step.tool_calls) for step in run_log.steps)}"
        )
        print(f"- Status: {run_log.status}")
        print(f"- Has solution: {run_log.final_code is not None}")

        # Print detailed log if verbose mode enabled
        if os.getenv("VERBOSE"):
            print(f"\nDetailed Log:")
            for i, step in enumerate(run_log.steps):
                print(f"\nStep {i+1}:")
                print(f"  Tool calls: {len(step.tool_calls)}")
                if step.tool_calls:
                    for call in step.tool_calls:
                        print(f"    - {call.tool_name}: {call.status}")

    except Exception as e:
        logger.error(f"Debugging session failed: {e}")
        return


def example_custom_registry():
    """Example 2: Advanced usage with custom tool registry."""

    logger.info("\n" + "=" * 50)
    logger.info("Advanced Example: Custom Registry")
    logger.info("=" * 50)

    # Create custom registry
    custom_registry = ToolRegistry()

    # Check registry stats
    registry_stats = custom_registry.get_stats()
    logger.info(f"Registry loaded with {registry_stats['total_tools']} tools")

    # Create agent with custom registry (dependency injection)
    custom_agent = PatchworkAgent(
        tool_registry=custom_registry,
        model="gpt-4.1-mini",
        max_iterations=3,
        temperature=0.3,
    )

    logger.info("Custom agent created with dependency injection")

    # Simple problem for demonstration
    simple_problem = ProblemContext(
        entry_point="add_numbers",
        goal="Fix the add function to correctly add two numbers",
        quality_criteria="Should correctly add two numbers",
        tests_formatted="add_numbers(2, 3) should return 5",
        broken_code="def add_numbers(a, b):\n    return a - b  # Bug: using subtraction instead of addition",
    )

    try:
        result = custom_agent.run(simple_problem)
        logger.info(f"Custom agent result: {custom_agent.run_log.status}")

        if custom_agent.run_log.status == "success":
            print(f"\nFixed simple function:\n{result}")
        else:
            print(f"Custom agent failed to fix the problem")

    except Exception as e:
        logger.error(f"Custom agent failed: {e}")


def example_best_of_n_sampling():
    """Example 3: Best-of-N sampling with proper temperature control."""

    logger.info("\n" + "=" * 50)
    logger.info("Best-of-N Sampling Example")
    logger.info("=" * 50)

    # Problem setup
    broken_function = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Too slow for large n
    """

    test_cases = """
Test cases:
- fibonacci(0) should return 0
- fibonacci(1) should return 1  
- fibonacci(5) should return 5
- fibonacci(10) should return 55
- Should handle n=30 efficiently (under 1 second)
"""

    problem = ProblemContext(
        entry_point="fibonacci",
        goal="Fix the fibonacci function to be efficient for large inputs",
        quality_criteria="Must be fast enough to handle n=30 in under 1 second",
        tests_formatted=test_cases,
        broken_code=broken_function,
    )

    # Create HIGH temperature agent for diverse generation
    generator = create_agent(model="gpt-4.1-mini", max_iterations=3, temperature=0.8)

    # Create LOW temperature agent for consistent judging
    judge = create_agent(model="gpt-4.1-mini", max_iterations=1, temperature=0.1)

    # Generate N diverse candidates
    N = 3
    candidates = []
    logger.info(f"Generating {N} diverse solutions with temperature=0.8...")

    for i in range(N):
        try:
            solution = generator.run(problem)
            candidates.append(solution)
            logger.info(f"Generated candidate {i+1}: {len(solution)} chars")
        except Exception as e:
            logger.warning(f"Failed to generate candidate {i+1}: {e}")

    if not candidates:
        logger.error("No candidates generated!")
        return

    # Judge picks the best candidate
    logger.info(f"Judging {len(candidates)} candidates with temperature=0.1...")

    judge_prompt = f"""
You are evaluating multiple solutions to fix a fibonacci function.

Original problem: {problem.goal}
Quality criteria: {problem.quality_criteria}

Here are the candidate solutions:

"""

    for i, candidate in enumerate(candidates):
        judge_prompt += f"\n--- Candidate {i+1} ---\n{candidate}\n"

    judge_prompt += """

Evaluate each candidate and pick the BEST one based on:
1. Correctness (passes all test cases)
2. Efficiency (handles large inputs quickly)
3. Code quality and readability

Respond with just: "Best candidate: X" where X is the number.
"""

    # Create judging problem
    judge_problem = ProblemContext(
        entry_point="judge",
        goal="Pick the best fibonacci solution",
        quality_criteria="Select the most efficient and correct solution",
        tests_formatted="N/A",
        broken_code=judge_prompt,
    )

    try:
        judgment = judge.run(judge_problem)
        logger.info(f"Judge decision: {judgment}")

        if "candidate" in judgment.lower():
            print(f"\n🏆 Best-of-{N} Result:")
            print("-" * 40)
            print(judgment)

    except Exception as e:
        logger.error(f"Judging failed: {e}")
        logger.info("Falling back to first candidate")


def main():
    """Demonstrate PatchworkAgent usage with example problems."""

    print("🔧 PatchworkAgent Examples")
    print("=" * 50)

    # Run fibonacci optimization example
    example_fibonacci_optimization()

    # Run custom registry example
    example_custom_registry()

    # Run Best-of-N sampling example
    example_best_of_n_sampling()

    print("\n✅ Examples completed!")


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("DEBUG_LITELLM", "false")

    # Set your API key as an environment variable:
    # export OPENAI_API_KEY="your_key_here"
    # export ANTHROPIC_API_KEY="your_key_here"  # if using Claude models

    main()
