import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError


class TestCase(BaseModel):
    test_input: Any = Field(..., alias="input")
    expected_output: Any = Field(..., alias="expected")


class TestResult(BaseModel):
    success: bool
    passed_count: int
    total_count: int
    feedback: str


def run_tests(code: str, tests: List[Dict], entry_point: str) -> str:
    """
    Executes a given Python code snippet against a list of test cases.

    This function provides a secure test harness to run untrusted code in an
    isolated subprocess, check its output against expected results, and return
    a human-readable summary of the outcome.

    Args:
        code: A string containing the Python function to be tested.
        tests: A list of dictionaries, where each dictionary represents a
            single test case and must contain 'input' and 'expected' keys.
        entry_point: The name of the function within the `code` to execute.

    Returns:
        A formatted string summarizing the test results, including the number
        of tests passed and detailed feedback on the first failure.
    """
    try:
        test_cases = [TestCase(**t) for t in tests]
    except ValidationError as e:
        result = TestResult(
            success=False,
            passed_count=0,
            total_count=len(tests),
            feedback=f"Invalid test case format: {e}",
        )
        return _format_for_agent(result)

    total_count = len(test_cases)

    runner_script = f"""
import sys

# --- Start of Agent's Code ---
{code}
# --- End of Agent's Code ---

def run_all_tests():
    tests = {[(t.test_input, t.expected_output) for t in test_cases]}
    passed_count = 0
    
    for i, (test_input, expected_output) in enumerate(tests):
        try:
            actual_output = None
            if isinstance(test_input, list):
                actual_output = {entry_point}(*test_input)
            else:
                actual_output = {entry_point}(test_input)
            
            assert actual_output == expected_output
            passed_count += 1
        except Exception as e:
            print(passed_count, file=sys.stdout)
            print(f"Test {{i+1}} failed for input: {{repr(test_input)}}\\n"
                  f"  - Expected: {{repr(expected_output)}}\\n"
                  f"  - Got: {{repr(actual_output)}}\\n"
                  f"  - Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
            sys.exit(1)
            
    print(passed_count, file=sys.stdout)
    sys.exit(0)
    
if __name__ == "__main__":
    run_all_tests()
"""

    script_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py", encoding="utf-8"
        ) as f:
            f.write(runner_script)
            script_path = f.name

        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
        )

        passed_count = (
            int(process.stdout.strip()) if process.stdout.strip().isdigit() else 0
        )

        if process.returncode == 0:
            result = TestResult(
                success=True,
                passed_count=passed_count,
                total_count=total_count,
                feedback="All tests passed.",
            )
        else:
            result = TestResult(
                success=False,
                passed_count=passed_count,
                total_count=total_count,
                feedback=process.stderr.strip()
                or "Test script failed with no error message.",
            )
        return _format_for_agent(result)
    except subprocess.TimeoutExpired:
        result = TestResult(
            success=False,
            passed_count=0,
            total_count=total_count,
            feedback="Code execution timed out after 15 seconds.",
        )
        return _format_for_agent(result)
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def _format_for_agent(result: TestResult) -> str:
    """Formats the test results into a human-readable string for the agent."""
    summary = f"Test Result: {result.passed_count}/{result.total_count} tests passed."
    output = [summary]

    if not result.success:
        output.append("Feedback:")
        output.append(result.feedback)

    return "\\n".join(output)
