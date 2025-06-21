import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError


class TestCase(BaseModel):
    test_input: Any
    expected_output: Any


class TestResult(BaseModel):
    success: bool
    passed_count: int
    total_count: int
    feedback: str


def run_tests(code: str, tests: List[Dict], entry_point: str) -> TestResult:
    try:
        test_cases = [TestCase(**t) for t in tests]
    except ValidationError as e:
        return TestResult(
            success=False,
            passed_count=0,
            total_count=len(tests),
            feedback=f"Invalid test case format: {e}",
        )

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
