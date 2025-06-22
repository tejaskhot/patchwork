#!/usr/bin/env python3
"""
Simple experiment runner for Patchwork agent.

Makes it easy to test different models and problems for code debugging experiments.

Usage:
    python run_patchwork.py --problem max_list --model gpt-4.1-mini
    python run_patchwork.py --problem filter_top_students --model gpt-4.1
    python run_patchwork.py --list-problems
"""

import argparse
import json
import logging
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
        dataset_list = json.load(f)

    # Convert list of problems to dictionary with id as key
    return {problem["id"]: problem for problem in dataset_list}


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
    test_cases = problem_data.get("tests", [])
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


def run_problem(
    problem_id: str, model: str, max_iterations: int, no_eval: bool = False
) -> Dict[str, Any]:
    """Run agent on a single problem and return results."""
    # Load dataset and create problem context
    dataset = load_dataset()
    problem_context = create_problem_context(problem_id, dataset)

    # Create agent
    agent = create_agent(model=model, max_iterations=max_iterations)

    # Run the agent
    try:
        result = agent.run(problem_context)
        run_log = agent.get_run_log()
    except Exception as e:
        return {
            "problem_id": problem_id,
            "status": "error",
            "error": str(e),
            "result": None,
            "run_log": None,
            "evaluation_results": None,
        }

    # Run evaluation if requested
    evaluation_results = None
    if not no_eval:
        try:
            evaluator = PatchworkEvaluator()
            test_cases = dataset[problem_id].get("tests", [])
            original_code = dataset[problem_id]["broken_code"]
            entry_point = dataset[problem_id]["entry_point"]
            evaluation_results = evaluator.evaluate(
                run_log, test_cases, original_code, entry_point
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    # Save results
    save_results(problem_id, model, result, run_log, evaluation_results)

    return {
        "problem_id": problem_id,
        "status": "success" if run_log and run_log.status == "success" else "failed",
        "result": result,
        "run_log": run_log,
        "evaluation_results": evaluation_results,
    }


def run_all_problems(args):
    """Run agent on all problems in the dataset."""
    print(f"🚀 Running agent on ALL problems with model: {args.model}")
    print(f"⚙️  Max iterations: {args.max_iterations}")
    print(f"📊 Evaluation: {'Disabled' if args.no_eval else 'Enabled'}")

    # Load all problems
    dataset = load_dataset()
    problem_ids = list(dataset.keys())

    print(f"\n📋 Found {len(problem_ids)} problems: {problem_ids}")
    print("=" * 80)

    # Track overall results
    all_results = []
    successful_runs = 0
    failed_runs = 0

    # Run each problem
    for i, problem_id in enumerate(problem_ids, 1):
        print(f"\n🔄 RUNNING PROBLEM {i}/{len(problem_ids)}: {problem_id}")
        print("-" * 50)

        try:
            print(f"🔧 Loading problem: {problem_id}")
            print(f"🤖 Initializing agent with model: {args.model}")
            print(f"🚀 Running agent...")

            result = run_problem(
                problem_id, args.model, args.max_iterations, args.no_eval
            )
            all_results.append(result)

            if result["status"] == "success":
                successful_runs += 1
            else:
                failed_runs += 1

            # Print individual summary
            if not args.no_eval:
                print(f"📊 Running evaluation...")
            print_summary(
                problem_id,
                args.model,
                result["result"],
                result["run_log"],
                result["evaluation_results"],
            )

        except Exception as e:
            logger.error(f"Failed to run problem {problem_id}: {e}")
            failed_runs += 1
            all_results.append(
                {
                    "problem_id": problem_id,
                    "status": "error",
                    "error": str(e),
                    "result": None,
                    "run_log": None,
                    "evaluation_results": None,
                }
            )

    # Print overall summary
    print("\n" + "=" * 80)
    print(f"🏁 BATCH RUN COMPLETE")
    print("=" * 80)
    print(f"📊 Total Problems: {len(problem_ids)}")
    print(f"✅ Successful: {successful_runs}")
    print(f"❌ Failed: {failed_runs}")
    print(f"📈 Success Rate: {successful_runs/len(problem_ids):.1%}")

    # Save batch summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_summary_file = (
        Path("results")
        / f"batch_summary_{args.model.replace('/', '_')}_{timestamp}.json"
    )

    with open(batch_summary_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": args.model,
                "max_iterations": args.max_iterations,
                "evaluation_enabled": not args.no_eval,
                "total_problems": len(problem_ids),
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / len(problem_ids),
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"📄 Batch summary saved to: {batch_summary_file}")
    print("=" * 80)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run Patchwork agent experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_patchwork.py --problem max_list --model gpt-4.1-nano
  python run_patchwork.py --problem filter_top_students --model gpt-4.1-mini
  python run_patchwork.py --problem complex_task --model gpt-4.1
  python run_patchwork.py --list-problems
  python run_patchwork.py --all-problems --model gpt-4.1-nano
  python run_patchwork.py --all-problems --model gpt-4.1-mini --no-eval
  
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
        "--all-problems",
        action="store_true",
        help="Run agent on all problems in the dataset",
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

    # Handle all problems
    if args.all_problems:
        run_all_problems(args)
        return

    # Validate required arguments
    if not args.problem:
        parser.error("--problem is required (or use --list-problems or --all-problems)")

    # Run single problem
    print(f"🔧 Loading problem: {args.problem}")
    print(f"🤖 Initializing agent with model: {args.model}")
    print(f"🚀 Running agent...")

    result_info = run_problem(
        args.problem, args.model, args.max_iterations, args.no_eval
    )

    if result_info["status"] == "error":
        logger.error(f"Agent run failed: {result_info['error']}")
        sys.exit(1)

    # Print individual summary
    if not args.no_eval:
        print(f"📊 Running evaluation...")
    print_summary(
        args.problem,
        args.model,
        result_info["result"],
        result_info["run_log"],
        result_info["evaluation_results"],
    )

    print(f"\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
