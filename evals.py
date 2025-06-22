"""
Multi-Layered Evaluation System for Patchwork Agent

This module implements a three-tier evaluation framework:
1. Level 1: Primary deterministic rewards (task completion)
2. Level 2: Secondary objective rewards (process & code quality)  
3. Level 3: Tertiary subjective rewards (LLM-as-judge)

The unified "Patchwork Score" combines all metrics into a single performance measure.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel, Field, field_validator

from agent import RunLog
from tools.linter import lint
from tools.test_harness import _execute_tests

# Set up logging
logger = logging.getLogger(__name__)


class EvaluationMetrics(BaseModel):
    """Complete set of evaluation metrics for a single agent run."""

    # Level 1: Primary Rewards (Task Completion)
    success_rate: float = Field(
        ge=0.0, le=1.0, description="Binary success (1.0 if all tests pass)"
    )
    completion_rate: float = Field(
        ge=0.0, le=1.0, description="Percentage of tests passed"
    )
    efficiency_score: float = Field(
        ge=0.0, le=1.0, description="Inverse of tool calls used"
    )

    # Level 2: Secondary Rewards (Process & Code Quality)
    invalid_action_penalty: float = Field(
        ge=0.0, description="Penalty for invalid tool calls"
    )
    regression_penalty: float = Field(
        ge=0.0, description="Penalty for decreasing test pass rate"
    )
    linter_score: float = Field(
        ge=0.0, le=10.0, description="Objective code quality score"
    )

    # Level 3: Tertiary Rewards (LLM-as-Judge)
    code_elegance_score: float = Field(
        ge=0.0, le=10.0, description="LLM judge: code elegance"
    )
    strategic_efficiency_score: float = Field(
        ge=0.0, le=10.0, description="LLM judge: strategy quality"
    )

    # Metadata
    total_iterations: int = Field(ge=0, description="Total debugging iterations")
    total_tool_calls: int = Field(ge=0, description="Total tool calls made")
    final_status: str = Field(description="Final agent status")


class PatchworkScore(BaseModel):
    """The unified Patchwork Score combining all evaluation metrics."""

    score: float = Field(description="Unified score combining all metrics")
    breakdown: Dict[str, float] = Field(description="Individual metric contributions")

    @field_validator("score")
    def validate_score_range(cls, v):
        """Patchwork score can be negative due to penalties."""
        return round(v, 4)


class Level1Evaluator:
    """Level 1: Primary deterministic rewards focused on task completion."""

    @staticmethod
    def evaluate_success_rate(
        final_code: Optional[str], test_cases: List[Dict[str, Any]], entry_point: str
    ) -> float:
        """Binary success: 1.0 if all tests pass, 0.0 otherwise."""
        if not final_code:
            return 0.0

        try:
            result = _execute_tests(final_code, test_cases, entry_point)
            return 1.0 if result.success else 0.0
        except Exception as e:
            logger.error(f"Error evaluating success rate: {e}")
            return 0.0

    @staticmethod
    def evaluate_completion_rate(
        final_code: Optional[str], test_cases: List[Dict[str, Any]], entry_point: str
    ) -> float:
        """Percentage of tests that passed."""
        if not final_code:
            return 0.0

        try:
            result = _execute_tests(final_code, test_cases, entry_point)
            return (
                result.passed_count / result.total_count
                if result.total_count > 0
                else 0.0
            )
        except Exception as e:
            logger.error(f"Error evaluating completion rate: {e}")
            return 0.0

    @staticmethod
    def evaluate_efficiency_score(run_log: RunLog) -> float:
        """Efficiency based on inverse of tool calls used."""
        total_tool_calls = sum(len(step.tool_calls) for step in run_log.steps)
        if total_tool_calls == 0:
            return 1.0  # Perfect efficiency if no tools needed

        # Normalize: 1.0 for 1 tool call, approaching 0 for many calls
        return 1.0 / (1.0 + total_tool_calls)


class Level2Evaluator:
    """Level 2: Secondary objective rewards for process and code quality."""

    @staticmethod
    def evaluate_invalid_action_penalty(run_log: RunLog) -> float:
        """Penalty for invalid tool calls or errors."""
        penalty = 0.0

        for step in run_log.steps:
            # Check structured results for errors
            for tool_result in step.tool_results_structured:
                result = tool_result["result"]
                # Check if result indicates an error
                if isinstance(result, str) and (
                    "Error" in result or "Exception" in result
                ):
                    penalty += 0.1  # 0.1 penalty per error
                elif isinstance(result, dict) and result.get("error"):
                    penalty += 0.1  # 0.1 penalty per structured error

        return penalty

    @staticmethod
    def evaluate_regression_penalty(run_log: RunLog) -> float:
        """Penalty for decreasing the number of passing tests."""
        penalty = 0.0
        previous_pass_count = None

        for step in run_log.steps:
            # Use structured data instead of regex parsing
            for tool_result in step.tool_results_structured:
                if tool_result["tool_name"] == "run_tests" and isinstance(
                    tool_result["result"], dict
                ):
                    result_data = tool_result["result"]
                    if "passed_count" in result_data:
                        current_pass_count = result_data["passed_count"]

                        if (
                            previous_pass_count is not None
                            and current_pass_count < previous_pass_count
                        ):
                            penalty += 0.2  # 0.2 penalty per regression

                        previous_pass_count = current_pass_count

        return penalty

    @staticmethod
    def evaluate_linter_score(final_code: Optional[str]) -> float:
        """Objective code quality using linter."""
        if not final_code:
            return 0.0

        try:
            result = lint(final_code)
            # Extract numerical score from linter output
            # Assuming linter returns a score out of 10
            if isinstance(result, dict) and "score" in result:
                return float(result["score"])
            else:
                # Parse score from text output (e.g., "Your code has been rated at 8.5/10")
                score_match = re.search(r"(\d+\.?\d*)/10", str(result))
                if score_match:
                    return float(score_match.group(1))
                else:
                    return 5.0  # Default neutral score
        except Exception as e:
            logger.error(f"Error evaluating linter score: {e}")
            return 0.0


class Level3Evaluator:
    """Level 3: Tertiary subjective rewards using LLM-as-judge."""

    def __init__(self, judge_model: str = "gpt-4.1-mini"):
        """Initialize with a judge model."""
        self.judge_model = judge_model

    def _call_judge(self, prompt: str) -> str:
        """Call the LLM judge with error handling."""
        try:
            response = litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM judge: {e}")
            return "Error: Could not evaluate"

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM judge response."""
        # Look for patterns like "Score: 8/10" or "8.5/10" or "Rating: 7"
        patterns = [
            r"Score:\s*(\d+\.?\d*)/10",
            r"Rating:\s*(\d+\.?\d*)/10",
            r"(\d+\.?\d*)/10",
            r"Score:\s*(\d+\.?\d*)",
            r"Rating:\s*(\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-10 scale
                return min(10.0, max(0.0, score))

        # If no score found, return neutral score
        logger.warning(f"Could not extract score from judge response: {response}")
        return 5.0

    def evaluate_code_elegance(
        self, original_code: str, final_code: Optional[str]
    ) -> float:
        """LLM judge evaluation of code elegance and quality."""
        if not final_code:
            return 0.0

        prompt = f"""You are a senior Python developer. On a scale of 1-10, please rate the following code fix for readability, simplicity, and maintainability.

Original broken code:
```python
{original_code}
```

Fixed code:
```python
{final_code}
```

Please provide a score from 1-10 and explain your reasoning. Focus on:
- Code readability and clarity
- Simplicity and elegance of the solution
- Maintainability and best practices
- How well the fix addresses the original problem

Format your response as: "Score: X/10" followed by your explanation."""

        response = self._call_judge(prompt)
        return self._extract_score(response)

    def evaluate_strategic_efficiency(self, run_log: RunLog) -> float:
        """LLM judge evaluation of the agent's problem-solving strategy."""
        if not run_log.steps:
            return 0.0

        # Create a summary of the agent's actions
        strategy_summary = []
        for i, step in enumerate(run_log.steps):
            if step.tool_calls:
                tools_used = [call.split("(")[0] for call in step.tool_calls]
                strategy_summary.append(f"Step {i+1}: Used tools {tools_used}")
            strategy_summary.append(f"  Response: {step.assistant_response[:100]}...")

        strategy_text = "\n".join(strategy_summary)

        prompt = f"""You are an expert in AI agent design. Given this complete log of an agent's debugging session, please rate the agent's problem-solving strategy on a scale of 1-10.

Agent's debugging strategy:
{strategy_text}

Final status: {run_log.status}
Total steps: {len(run_log.steps)}

Please evaluate:
- Did the agent use the right tools for each situation?
- Was the strategy logical and well-structured?
- Did the agent avoid unnecessary repetition or loops?
- How efficiently did the agent work towards the solution?

Format your response as: "Score: X/10" followed by your explanation."""

        response = self._call_judge(prompt)
        return self._extract_score(response)


class PatchworkEvaluator:
    """Main evaluator that combines all three levels into a unified score."""

    def __init__(self, judge_model: str = "gpt-4.1-mini"):
        """Initialize with evaluation components."""
        self.level1 = Level1Evaluator()
        self.level2 = Level2Evaluator()
        self.level3 = Level3Evaluator(judge_model)

    def evaluate(
        self,
        run_log: RunLog,
        test_cases: List[Dict[str, Any]],
        original_code: str,
        entry_point: str,
    ) -> tuple[EvaluationMetrics, PatchworkScore]:
        """
        Perform complete evaluation of an agent run.

        Args:
            run_log: Complete log of the agent's debugging session
            test_cases: Test cases used for evaluation
            original_code: The original broken code
            entry_point: The function name to test

        Returns:
            Tuple of (detailed metrics, unified score)
        """
        logger.info("Starting comprehensive evaluation")

        # Level 1 Evaluations
        success_rate = self.level1.evaluate_success_rate(
            run_log.final_code, test_cases, entry_point
        )
        completion_rate = self.level1.evaluate_completion_rate(
            run_log.final_code, test_cases, entry_point
        )
        efficiency_score = self.level1.evaluate_efficiency_score(run_log)

        # Level 2 Evaluations
        invalid_action_penalty = self.level2.evaluate_invalid_action_penalty(run_log)
        regression_penalty = self.level2.evaluate_regression_penalty(run_log)
        linter_score = self.level2.evaluate_linter_score(run_log.final_code)

        # Level 3 Evaluations
        code_elegance_score = self.level3.evaluate_code_elegance(
            original_code, run_log.final_code
        )
        strategic_efficiency_score = self.level3.evaluate_strategic_efficiency(run_log)

        # Create metrics object
        metrics = EvaluationMetrics(
            success_rate=success_rate,
            completion_rate=completion_rate,
            efficiency_score=efficiency_score,
            invalid_action_penalty=invalid_action_penalty,
            regression_penalty=regression_penalty,
            linter_score=linter_score,
            code_elegance_score=code_elegance_score,
            strategic_efficiency_score=strategic_efficiency_score,
            total_iterations=len(run_log.steps),
            total_tool_calls=sum(len(step.tool_calls) for step in run_log.steps),
            final_status=run_log.status or "unknown",
        )

        # Calculate unified Patchwork Score
        patchwork_score = self._calculate_patchwork_score(metrics)

        logger.info(f"Evaluation complete. Patchwork Score: {patchwork_score.score}")
        return metrics, patchwork_score

    def _calculate_patchwork_score(self, metrics: EvaluationMetrics) -> PatchworkScore:
        """Calculate the unified Patchwork Score with detailed breakdown."""

        # Normalize linter score and LLM judge scores to 0-1 scale
        normalized_linter = metrics.linter_score / 10.0
        normalized_elegance = metrics.code_elegance_score / 10.0
        normalized_strategy = metrics.strategic_efficiency_score / 10.0

        # Calculate weighted components
        components = {
            "success_rate": 0.5 * metrics.success_rate,
            "linter_score": 0.2 * normalized_linter,
            "code_elegance": 0.1 * normalized_elegance,
            "strategic_efficiency": 0.2 * normalized_strategy,
            "invalid_action_penalty": -0.1 * metrics.invalid_action_penalty,
            "regression_penalty": -0.1 * metrics.regression_penalty,
        }

        # Sum all components
        total_score = sum(components.values())

        return PatchworkScore(score=total_score, breakdown=components)
