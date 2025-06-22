#!/usr/bin/env python3
"""
Simple experiment runner for Patchwork agent.

Makes it easy to test different models and problems for code debugging experiments.

Usage:
    python run_patchwork.py --problem max_list --model gpt-4.1-mini
    python run_patchwork.py --problem filter_students --model gpt-4.1
    python run_patchwork.py --list-problems
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from agent import ProblemContext, create_agent
from evals import PatchworkEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset() -> Dict[str, Any]:
    """Load the test cases dataset."""
    dataset_path = Path("test_cases/dataset.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        return json.load(f)


def list_available_problems():
    """Show all available problems in the dataset."""
    dataset = load_dataset()

    print("\n🔧 Available Problems:")
    print("=" * 50)

    for problem_id, problem_data in dataset.items():
        print(f"📋 {problem_id}")
        print(f"   Function: {problem_data.get('entry_point', 'N/A')}")
        print(f"   Goal: {problem_data.get('goal', 'N/A')[:60]}...")
        print()


def create_problem_context(problem_id: str, dataset: Dict[str, Any]) -> ProblemContext:
    """Create a ProblemContext from dataset entry."""
    if problem_id not in dataset:
        available = list(dataset.keys())
        logger.error(f"Problem '{problem_id}' not found. Available: {available}")
        sys.exit(1)

    problem_data = dataset[problem_id]

    # Format test cases for display
    test_cases = problem_data.get("test_cases", [])
    tests_formatted = "\n".join(
        [
            f"Test {i+1}: {test['input']} → {test['expected']}"
            for i, test in enumerate(test_cases)
        ]
    )

    return ProblemContext(
        entry_point=problem_data["entry_point"],
        goal=problem_data["goal"],
        quality_criteria=problem_data.get(
            "quality_criteria", "Code should pass all tests"
        ),
        tests_formatted=tests_formatted,
        broken_code=problem_data["broken_code"],
    )


def save_results(problem_id: str, model: str, result: str, run_log, evaluation_results):
    """Save experiment results to files."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed run log
    log_file = (
        results_dir / f"run_log_{problem_id}_{model.replace('/', '_')}_{timestamp}.json"
    )
    with open(log_file, "w") as f:
        json.dump(
            {
                "problem_id": problem_id,
                "model": model,
                "timestamp": timestamp,
                "final_result": result,
                "run_log": run_log.model_dump(),
                "evaluation": {
                    "metrics": (
                        evaluation_results[0].model_dump()
                        if evaluation_results
                        else None
                    ),
                    "patchwork_score": (
                        evaluation_results[1].model_dump()
                        if evaluation_results
                        else None
                    ),
                },
            },
            f,
            indent=2,
        )

    logger.info(f"📄 Detailed results saved to: {log_file}")
    return log_file


def print_summary(
    problem_id: str, model: str, result: str, run_log, evaluation_results
):
    """Print a nice summary of the experiment."""
    print("\n" + "=" * 60)
    print(f"🤖 PATCHWORK EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"📋 Problem: {problem_id}")
    print(f"🧠 Model: {model}")
    print(f"⏱️  Total Steps: {len(run_log.steps)}")
    print(f"🔧 Total Tool Calls: {sum(len(step.tool_calls) for step in run_log.steps)}")
    print(f"📊 Final Status: {run_log.status}")

    if evaluation_results:
        metrics, patchwork_score = evaluation_results
        print(f"\n🎯 EVALUATION RESULTS:")
        print(f"   Success Rate: {metrics.success_rate:.1%}")
        print(f"   Completion Rate: {metrics.completion_rate:.1%}")
        print(f"   Efficiency Score: {metrics.efficiency_score:.3f}")
        print(f"   Linter Score: {metrics.linter_score:.1f}/10")
        print(f"   Code Elegance: {metrics.code_elegance_score:.1f}/10")
        print(f"   Strategy Score: {metrics.strategic_efficiency_score:.1f}/10")
        print(f"\n⭐ PATCHWORK SCORE: {patchwork_score.score:.4f}")

    print(f"\n📝 FINAL RESULT:")
    print("-" * 40)
    if run_log.final_code:
        print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print(result)
    print("=" * 60)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run Patchwork agent experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_patchwork.py --problem max_list --model gpt-4.1-nano
  python run_patchwork.py --problem filter_students --model gpt-4.1-mini
  python run_patchwork.py --problem complex_task --model gpt-4.1
  python run_patchwork.py --list-problems
  
GPT-4.1 Series Models (recommended):
  gpt-4.1-nano  (smallest, fastest, cheapest)
  gpt-4.1-mini  (medium size and capability)
  gpt-4.1       (largest, most capable)
  
Other Available Models:
  claude-3-5-sonnet-20241022
  claude-3-5-haiku-20241022
  deepseek/deepseek-coder-v2
  gemini/gemini-1.5-pro-latest
        """,
    )

    parser.add_argument("--problem", type=str, help="Problem ID from the dataset")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="LiteLLM model string (default: gpt-4.1-nano)",
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation (faster for quick testing)",
    )

    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List all available problems and exit",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations (default: 10)",
    )

    args = parser.parse_args()

    # Handle list problems
    if args.list_problems:
        list_available_problems()
        return

    # Validate required arguments
    if not args.problem:
        parser.error("--problem is required (or use --list-problems)")

    # Load dataset and create problem context
    print(f"🔧 Loading problem: {args.problem}")
    dataset = load_dataset()
    problem_context = create_problem_context(args.problem, dataset)

    # Create agent
    print(f"🤖 Initializing agent with model: {args.model}")
    agent = create_agent(model=args.model, max_iterations=args.max_iterations)

    # Run the agent
    print(f"🚀 Running agent...")
    try:
        result = agent.run(problem_context)
        run_log = agent.get_run_log()
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        sys.exit(1)

    # Run evaluation if requested
    evaluation_results = None
    if not args.no_eval:
        print(f"📊 Running evaluation...")
        try:
            evaluator = PatchworkEvaluator()
            test_cases = dataset[args.problem].get("test_cases", [])
            original_code = dataset[args.problem]["broken_code"]
            evaluation_results = evaluator.evaluate(run_log, test_cases, original_code)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    # Save results
    save_results(args.problem, args.model, result, run_log, evaluation_results)

    # Print summary
    print_summary(args.problem, args.model, result, run_log, evaluation_results)

    print(f"\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
