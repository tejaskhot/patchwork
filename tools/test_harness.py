import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Union

from pydantic import BaseModel


class TestResult(BaseModel):
    success: bool
    passed_count: int
    total_count: int
    feedback: str


def _execute_tests(
    code: str, tests: Union[str, List[Dict]], entry_point: str
) -> TestResult:
    """Core test execution logic. Returns TestResult object."""
    try:
        # Handle both JSON string and already-parsed list inputs
        if isinstance(tests, str):
            test_data = json.loads(tests)
        else:
            test_data = tests

        # Validate that we have a list
        if not isinstance(test_data, list):
            raise ValueError("Tests must be a list of test cases")

        # Validate each test case has required keys
        for i, test_case in enumerate(test_data):
            if not isinstance(test_case, dict):
                raise ValueError(f"Test case {i} must be a dictionary")
            if "input" not in test_case or "expected" not in test_case:
                raise ValueError(f"Test case {i} missing 'input' or 'expected' key")

    except (json.JSONDecodeError, ValueError) as e:
        return TestResult(
            success=False,
            passed_count=0,
            total_count=0,
            feedback=f"Invalid test case format: {e}",
        )

    total_count = len(test_data)

    # Extract test inputs and expected outputs directly from dicts
    test_cases_for_script = [
        (test_case["input"], test_case["expected"]) for test_case in test_data
    ]

    runner_script = f"""
import sys

# --- Start of Agent's Code ---
{code}
# --- End of Agent's Code ---

def run_all_tests():
    tests = {test_cases_for_script}
    passed_count = 0
    
    for i, (test_input, expected_output) in enumerate(tests):
        try:
            actual_output = None
            # Always pass test_input as a single argument
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
            return TestResult(
                success=True,
                passed_count=passed_count,
                total_count=total_count,
                feedback="All tests passed.",
            )
        else:
            return TestResult(
                success=False,
                passed_count=passed_count,
                total_count=total_count,
                feedback=process.stderr.strip()
                or "Test script failed with no error message.",
            )
    except subprocess.TimeoutExpired:
        return TestResult(
            success=False,
            passed_count=0,
            total_count=total_count,
            feedback="Code execution timed out after 15 seconds.",
        )
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def run_tests(code: str, tests: Union[str, List[Dict]], entry_point: str) -> str:
    """
    Executes a given Python code snippet against a list of test cases.

    This function provides a secure test harness to run untrusted code in an
    isolated subprocess, check its output against expected results, and return
    a human-readable summary of the outcome.

    Args:
        code: A string containing the Python function to be tested.
        tests: Either a JSON string or a list of dictionaries, where each dictionary
            represents a single test case and must contain 'input' and 'expected' keys.
        entry_point: The name of the function within the `code` to execute.

    Returns:
        A formatted string summarizing the test results, including the number
        of tests passed and detailed feedback on the first failure.
    """
    result = _execute_tests(code, tests, entry_point)
    return _format_for_agent(result)


def _format_for_agent(result: TestResult) -> str:
    """Formats the test results into a human-readable string for the agent."""
    summary = f"Test Result: {result.passed_count}/{result.total_count} tests passed."
    output = [summary]

    if not result.success:
        output.append("Feedback:")
        output.append(result.feedback)

    return "\\n".join(output)
