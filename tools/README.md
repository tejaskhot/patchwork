# Patchwork Developer Toolkit

This directory contains the suite of specialized tools that the Patchwork agent uses to diagnose, debug, and repair Python code. Each tool is designed to be run in a secure, sandboxed subprocess and provides structured feedback that the agent can reason about.

Below is a summary of the available tools.

---

### 1. Test Harness (`test_harness.py`)

- **Function:** `run_tests(code: str, tests: List[Dict], entry_point: str) -> str`
- **Purpose:** The primary tool for verifying code correctness. It runs the provided `code` against a full suite of test cases.
- **Key Features:**
  - Executes code in a secure, isolated subprocess.
  - Compares the function's output against expected results for multiple inputs.
  - Returns a formatted string summary of the test results.

---

### 2. Linter (`linter.py`)

- **Function:** `lint(code: str) -> str`
- **Purpose:** To analyze code for static errors, style violations, and potential bugs without executing it.
- **Key Features:**
  - Uses `pylint` to perform static analysis.
  - Returns a formatted string that includes a quality score (out of 10) and a list of specific issues found (e.g., "invalid-name", "unused-variable").
  - Helps the agent improve code quality and adhere to Python best practices.

---

### 3. Debugger (`debugger.py`)

- **Function:** `run_with_debugger(code: str, test_input: Any, entry_point: str) -> str`
- **Purpose:** To provide deep insight into the execution of code when a specific test case fails. This is the agent's tool for understanding *why* something is broken.
- **Key Features:**
  - Uses Python's `sys.settrace` to run code with a debugger attached.
  - Pinpoints the exact line of code where an error occurred.
  - Captures the state of all local variables at the moment of failure, allowing the agent to inspect the program's state.
  - Provides a human-readable summary of the error and variable states.

---

### 4. Plot Inspector (`plot_inspector.py`)

- **Function:** `inspect_plot(code: str, data: Any, entry_point: str) -> str`
- **Purpose:** A specialized tool to verify the visual properties of `matplotlib` plots without rendering an image.
- **Key Features:**
  - Executes plotting code using a non-interactive `'Agg'` backend to prevent crashes in a server environment.
  - Inspects the generated `matplotlib` `Axes` object for properties like title, axis labels, line color, and line style.
  - Returns a structured summary of the plot's visual elements, allowing the agent to fix visual bugs (e.g., "the plot title is wrong," "the line should be dashed").
