#!/usr/bin/env python3
"""
Plotting utilities for Patchwork agent results.

This script provides comprehensive visualization of agent performance across
different models and problems, making it easy to compare and analyze results.

Usage:
    python plot_results.py --model-comparison
    python plot_results.py --model gpt-4.1-nano --problems
    python plot_results.py --all-models --metric success_rate
    python plot_results.py --heatmap
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyzes and visualizes Patchwork agent results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data = self._load_all_results()

    def _load_all_results(self) -> Dict[str, List[Dict]]:
        """Load all result files organized by model."""
        data = defaultdict(list)

        if not self.results_dir.exists():
            logger.warning(f"Results directory {self.results_dir} not found")
            return data

        # Look for model subdirectories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            logger.info(f"Loading results for model: {model_name}")

            # Load all JSON files in this model directory
            for json_file in model_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        result = json.load(f)

                    # Skip batch summaries for individual problem analysis
                    if "batch_summary" in json_file.name:
                        continue

                    data[model_name].append(result)

                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded results for {len(data)} models")
        for model, results in data.items():
            logger.info(f"  {model}: {len(results)} results")

        return dict(data)

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.data.keys())

    def get_available_problems(self) -> List[str]:
        """Get list of available problems across all models."""
        problems = set()
        for model_results in self.data.values():
            for result in model_results:
                if "problem_id" in result:
                    problems.add(result["problem_id"])
        return sorted(list(problems))

    def extract_metrics(self, result: Dict) -> Dict:
        """Extract key metrics from a result."""
        metrics = {
            "problem_id": result.get("problem_id", "unknown"),
            "model": result.get("model", "unknown"),
            "success": False,  # Default to False, will be set properly below
            "timestamp": result.get("timestamp", ""),
        }

        # Check for success status in run_log first
        if "run_log" in result and "status" in result["run_log"]:
            metrics["success"] = result["run_log"]["status"] == "success"
        elif "status" in result:
            metrics["success"] = result["status"] == "success"

        # If we have evaluation metrics, prefer the success_rate from there
        if "evaluation" in result and result["evaluation"]:
            eval_metrics = result["evaluation"].get("metrics", {})
            if eval_metrics:
                # If success_rate is 1.0, then it's successful
                if "success_rate" in eval_metrics:
                    metrics["success"] = eval_metrics["success_rate"] == 1.0

                metrics.update(
                    {
                        "success_rate": eval_metrics.get("success_rate", 0),
                        "completion_rate": eval_metrics.get("completion_rate", 0),
                        "efficiency_score": eval_metrics.get("efficiency_score", 0),
                        "linter_score": eval_metrics.get("linter_score", 0),
                        "code_elegance_score": eval_metrics.get(
                            "code_elegance_score", 0
                        ),
                        "strategic_efficiency_score": eval_metrics.get(
                            "strategic_efficiency_score", 0
                        ),
                        "total_iterations": eval_metrics.get("total_iterations", 0),
                        "total_tool_calls": eval_metrics.get("total_tool_calls", 0),
                    }
                )

            # Extract patchwork score
            patchwork_score = result["evaluation"].get("patchwork_score", {})
            if patchwork_score:
                metrics["patchwork_score"] = patchwork_score.get("score", 0)

        return metrics

    def create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from all results."""
        all_metrics = []

        for model_name, results in self.data.items():
            for result in results:
                metrics = self.extract_metrics(result)
                all_metrics.append(metrics)

        return pd.DataFrame(all_metrics)

    def plot_model_comparison_overall(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Plot overall performance comparison across models."""
        df = self.create_dataframe()

        if df.empty:
            logger.warning("No data to plot")
            return

        # Group by model and calculate averages
        model_stats = (
            df.groupby("model")
            .agg(
                {
                    "success": "mean",
                    "success_rate": "mean",
                    "linter_score": "mean",
                    "code_elegance_score": "mean",
                    "patchwork_score": "mean",
                    "total_iterations": "mean",
                }
            )
            .round(3)
        )

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Model Performance Comparison - Overall", fontsize=16, fontweight="bold"
        )

        # Plot 1: Success Rate
        model_stats["success"].plot(kind="bar", ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title("Success Rate (Problem Completion)")
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].set_ylim(0, 1)

        # Plot 2: Test Success Rate
        if (
            "success_rate" in model_stats.columns
            and not model_stats["success_rate"].isna().all()
        ):
            model_stats["success_rate"].plot(
                kind="bar", ax=axes[0, 1], color="lightgreen"
            )
            axes[0, 1].set_title("Test Success Rate")
            axes[0, 1].set_ylabel("Test Success Rate")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].set_ylim(0, 1)

        # Plot 3: Code Quality (Linter Score)
        if (
            "linter_score" in model_stats.columns
            and not model_stats["linter_score"].isna().all()
        ):
            model_stats["linter_score"].plot(kind="bar", ax=axes[0, 2], color="orange")
            axes[0, 2].set_title("Code Quality (Linter Score)")
            axes[0, 2].set_ylabel("Linter Score")
            axes[0, 2].tick_params(axis="x", rotation=45)
            axes[0, 2].set_ylim(0, 10)

        # Plot 4: Code Elegance
        if (
            "code_elegance_score" in model_stats.columns
            and not model_stats["code_elegance_score"].isna().all()
        ):
            model_stats["code_elegance_score"].plot(
                kind="bar", ax=axes[1, 0], color="purple"
            )
            axes[1, 0].set_title("Code Elegance Score")
            axes[1, 0].set_ylabel("Elegance Score")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].set_ylim(0, 10)

        # Plot 5: Overall Patchwork Score
        if (
            "patchwork_score" in model_stats.columns
            and not model_stats["patchwork_score"].isna().all()
        ):
            model_stats["patchwork_score"].plot(kind="bar", ax=axes[1, 1], color="red")
            axes[1, 1].set_title("Overall Patchwork Score")
            axes[1, 1].set_ylabel("Patchwork Score")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].set_ylim(0, 1)

        # Plot 6: Efficiency (Iterations)
        if (
            "total_iterations" in model_stats.columns
            and not model_stats["total_iterations"].isna().all()
        ):
            model_stats["total_iterations"].plot(
                kind="bar", ax=axes[1, 2], color="brown"
            )
            axes[1, 2].set_title("Average Iterations (Lower is Better)")
            axes[1, 2].set_ylabel("Average Iterations")
            axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_per_problem_comparison(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Plot performance comparison per problem."""
        df = self.create_dataframe()

        if df.empty:
            logger.warning("No data to plot")
            return

        problems = df["problem_id"].unique()
        models = df["model"].unique()

        # Create success rate matrix
        success_matrix = []
        for problem in problems:
            row = []
            for model in models:
                problem_model_data = df[
                    (df["problem_id"] == problem) & (df["model"] == model)
                ]
                if not problem_model_data.empty:
                    success = problem_model_data["success"].iloc[0]
                    row.append(1.0 if success else 0.0)
                else:
                    row.append(np.nan)
            success_matrix.append(row)

        # Create heatmap
        fig, ax = plt.subplots(
            figsize=(max(8, len(models) * 1.5), max(6, len(problems) * 0.8))
        )

        im = ax.imshow(success_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticks(range(len(problems)))
        ax.set_yticklabels(problems)

        # Add text annotations
        for i, problem in enumerate(problems):
            for j, model in enumerate(models):
                value = success_matrix[i][j]
                if not np.isnan(value):
                    text = "✓" if value == 1.0 else "✗"
                    color = "white" if value == 0.0 else "black"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=12,
                        fontweight="bold",
                    )

        ax.set_title(
            "Success Rate by Problem and Model", fontsize=14, fontweight="bold", pad=20
        )
        ax.set_xlabel("Models")
        ax.set_ylabel("Problems")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Success Rate", rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def generate_summary_report(self) -> str:
        """Generate a text summary report of all results."""
        df = self.create_dataframe()

        if df.empty:
            return "No data available for summary report."

        report = []
        report.append("=" * 60)
        report.append("PATCHWORK AGENT PERFORMANCE SUMMARY")
        report.append("=" * 60)

        # Overall statistics
        total_runs = len(df)
        models = df["model"].nunique()
        problems = df["problem_id"].nunique()

        report.append(f"\n📊 OVERVIEW:")
        report.append(f"   Total Runs: {total_runs}")
        report.append(f"   Models Tested: {models}")
        report.append(f"   Problems Tested: {problems}")

        # Model performance summary
        report.append(f"\n🤖 MODEL PERFORMANCE:")

        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            total = len(model_data)
            successful = model_data["success"].sum()
            success_rate = successful / total if total > 0 else 0
            avg_score = model_data["patchwork_score"].mean()

            report.append(f"   {model}:")
            report.append(f"     Runs: {total}")
            report.append(f"     Success Rate: {success_rate:.1%}")
            if not pd.isna(avg_score):
                report.append(f"     Avg Patchwork Score: {avg_score:.3f}")

        # Problem difficulty analysis
        report.append(f"\n📋 PROBLEM DIFFICULTY:")
        problem_stats = (
            df.groupby("problem_id")["success"].agg(["count", "sum", "mean"]).round(3)
        )
        problem_stats = problem_stats.sort_values("mean", ascending=False)

        for problem in problem_stats.index:
            total = problem_stats.loc[problem, "count"]
            successful = problem_stats.loc[problem, "sum"]
            success_rate = problem_stats.loc[problem, "mean"]

            difficulty = (
                "Easy"
                if success_rate >= 0.8
                else "Medium" if success_rate >= 0.5 else "Hard"
            )
            report.append(
                f"   {problem}: {success_rate:.1%} success ({total} runs) - {difficulty}"
            )

        return "\n".join(report)


def main():
    """Main plotting interface."""
    parser = argparse.ArgumentParser(
        description="Visualize Patchwork agent results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_results.py --model-comparison
  python plot_results.py --per-problem
  python plot_results.py --summary
  python plot_results.py --all
        """,
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results (default: results)",
    )

    parser.add_argument(
        "--model-comparison", action="store_true", help="Plot overall model comparison"
    )

    parser.add_argument(
        "--per-problem",
        action="store_true",
        help="Plot per-problem performance heatmap",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Generate and print summary report"
    )

    parser.add_argument(
        "--all", action="store_true", help="Generate all plots and summary"
    )

    parser.add_argument("--save", action="store_true", help="Save plots to files")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)

    if not analyzer.data:
        logger.error("No results found. Make sure to run some experiments first.")
        return

    save_dir = Path("plots") if args.save else None
    if save_dir:
        save_dir.mkdir(exist_ok=True)

    # Generate requested plots
    if args.model_comparison or args.all:
        save_path = save_dir / "model_comparison.png" if save_dir else None
        analyzer.plot_model_comparison_overall(save_path, show=not args.save)

    if args.per_problem or args.all:
        save_path = save_dir / "per_problem_heatmap.png" if save_dir else None
        analyzer.plot_per_problem_comparison(save_path, show=not args.save)

    if args.summary or args.all:
        print(analyzer.generate_summary_report())

    if not any([args.model_comparison, args.per_problem, args.summary, args.all]):
        parser.print_help()
        print(f"\nAvailable models: {analyzer.get_available_models()}")
        print(f"Available problems: {analyzer.get_available_problems()}")


if __name__ == "__main__":
    main()
