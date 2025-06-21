"""
Python linting tool for the Patchwork agent.
Provides code quality analysis that the agent can understand and act upon.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import List

from pydantic import BaseModel, Field, field_validator

LINT_TIMEOUT_SECONDS = 30


class LintMessage(BaseModel):
    """A single linting issue found in the code."""

    line: int = Field(..., description="Line number where the issue occurs")
    column: int = Field(..., description="Column number where the issue occurs")
    symbol: str = Field(..., description="The pylint symbol (e.g., 'C0103')")
    message: str = Field(..., description="Human-readable description of the issue")
    type: str = Field(
        ..., description="Category: 'error', 'warning', 'convention', etc."
    )

    @field_validator("line", "column")
    @classmethod
    def validate_positions(cls, v):
        if v < 0:
            raise ValueError("Line and column numbers must be non-negative")
        return v


class LintResult(BaseModel):
    """Complete linting results with score and issues."""

    score: float = Field(..., description="Pylint score from 0.0 to 10.0")
    issues: List[LintMessage] = Field(
        default_factory=list, description="List of issues found"
    )
    summary: str = Field(..., description="Human-readable summary")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v):
        if not 0.0 <= v <= 10.0:
            raise ValueError("Score must be between 0.0 and 10.0")
        return v


def lint(code: str) -> str:
    """
    Lint Python code and return human-readable analysis for the agent.

    Args:
        code: Python code to analyze

    Returns:
        String containing score, issues, and summary that the LLM can understand
    """
    if not code.strip():
        return "No code provided to lint"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pylint",
                "--output-format=json",
                "--reports=yes",
                temp_file,
            ],
            capture_output=True,
            text=True,
            timeout=LINT_TIMEOUT_SECONDS,
        )

        issues = []
        if result.stdout.strip():
            try:
                messages = json.loads(result.stdout)
                for msg in messages:
                    if isinstance(msg, dict) and "line" in msg:
                        issues.append(
                            LintMessage(
                                line=msg.get("line", 0),
                                column=msg.get("column", 0),
                                symbol=msg.get("symbol", ""),
                                message=msg.get("message", ""),
                                type=msg.get("type", "").lower(),
                            )
                        )
            except (json.JSONDecodeError, ValueError):
                issues = _parse_text_output(result.stdout)

        score = _extract_score(result.stderr)

        if not issues:
            summary = f"Code looks good! Score: {score:.1f}/10.0"
        else:
            summary = f"Found {len(issues)} issues. Score: {score:.1f}/10.0"

        result = LintResult(score=score, issues=issues, summary=summary)
        return _format_for_agent(result)

    except subprocess.TimeoutExpired:
        return f"Linting timed out after {LINT_TIMEOUT_SECONDS} seconds"
    except FileNotFoundError:
        return "Pylint not found - install with: pip install pylint"
    except Exception as e:
        return f"Linting failed: {str(e)}"
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            pass


def _extract_score(stderr: str) -> float:
    """Extract pylint score from stderr."""
    match = re.search(r"Your code has been rated at ([\d.-]+)/10", stderr)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _parse_text_output(stdout: str) -> List[LintMessage]:
    """Fallback text parser when JSON parsing fails."""
    issues = []
    pattern = r"^[^:]+:(\d+):(\d+):\s*(\w+):\s*(.+?)\s*\(([^)]+)\)"

    for line in stdout.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            line_num, col_num, msg_type, message, symbol = match.groups()
            try:
                issues.append(
                    LintMessage(
                        line=int(line_num),
                        column=int(col_num),
                        type=msg_type.lower(),
                        symbol=symbol,
                        message=message.strip(),
                    )
                )
            except ValueError:
                # Skip malformed messages
                continue

    return issues


def _format_for_agent(result: LintResult) -> str:
    """
    Format linting results in a way that's easy for the LLM agent to understand.

    Returns a clear, structured string that the agent can parse and reason about.
    """
    output = [result.summary]

    if result.issues:
        output.append("\nIssues found:")
        for i, issue in enumerate(result.issues[:10], 1):
            output.append(f"{i}. Line {issue.line}: {issue.message} ({issue.symbol})")

        if len(result.issues) > 10:
            output.append(f"... and {len(result.issues) - 10} more issues")

    return "\n".join(output)
