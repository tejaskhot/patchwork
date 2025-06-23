# Patchwork Developer Toolkit

This directory contains the suite of specialized tools that the Patchwork agent uses to diagnose, debug, and repair Python code. Each tool is designed to be run in a secure, sandboxed subprocess and provides structured feedback that the agent can reason about.

The tools are automatically discovered and registered by the `ToolRegistry` system, which provides parameter validation, execution monitoring, and structured result handling.

---

## Tool Registry System

The `registry.py` module provides a dynamic tool discovery and management system:

- **Automatic Discovery**: Tools are automatically registered from the tools package
- **Parameter Validation**: Validates tool parameters before execution
- **LiteLLM Integration**: Generates tool schemas compatible with LiteLLM function calling
- **Execution Monitoring**: Tracks tool usage and performance
- **Error Handling**: Provides structured error messages and fallback behavior

---

## Available Tools

### 1. Test Harness (`test_harness.py`)

- **Function:** `run_tests(code: str, tests: Union[str, List[Dict]], entry_point: str) -> str`
- **Purpose:** Primary tool for verifying code correctness by executing code against test cases in a secure, isolated subprocess.
- **Parameters:**
  - `code`: Python function code as a string
  - `tests`: JSON string like `'[{"input": [1, 2], "expected": 3}]'` or list of test dictionaries with 'input' and 'expected' keys
  - `entry_point`: Function name to test (e.g., "my_function")
- **Key Features:**
  - Executes code in secure 15-second timeout subprocess
  - Handles both JSON string and pre-parsed list formats for test cases
  - Compares function output against expected results for multiple test cases
  - Returns formatted summaries with pass/fail counts and detailed failure feedback
  - Uses `TestResult` Pydantic model for structured result handling
  - Automatically cleans up temporary files after execution

---

### 2. Linter (`linter.py`)

- **Function:** `lint(code: str) -> str`
- **Purpose:** Analyze code for static errors, style violations, and potential bugs using pylint without executing the code.
- **Parameters:**
  - `code`: Python code to analyze
- **Key Features:**
  - Uses pylint for comprehensive static analysis with 30-second timeout protection
  - Custom scoring formula: `10.0 - (5*errors + warnings + refactors + conventions)` for fairer evaluation
  - Returns quality scores (0-10) with detailed issue reports
  - Provides line numbers, symbols, and descriptions for each issue
  - Categorizes issues by type (error, warning, convention, refactor)
  - Uses `LintResult` and `LintMessage` Pydantic models for structured results
  - Handles both JSON and text output formats from pylint
  - Graceful fallback when pylint is not available

---

### 3. Debugger (`debugger.py`)

- **Function:** `run_with_debugger(code: str, test_input: Any, entry_point: str) -> str`
- **Purpose:** Provide detailed execution tracing and variable inspection when code fails to understand *why* something is broken.
- **Parameters:**
  - `code`: Python code to debug
  - `test_input`: Input to pass to function (wrap single lists in outer list to prevent unpacking)
  - `entry_point`: Name of the function to call
- **Key Features:**
  - Uses Python's `sys.settrace` for comprehensive execution tracing with 30-second timeout
  - Pinpoints exact error locations with line numbers and exception details
  - Captures local variable states at failure points with safe repr() handling
  - Provides execution traces limited to relevant code sections (max 1000 lines)
  - Matplotlib compatibility with non-interactive 'Agg' backend
  - Uses `DebugResult` Pydantic model for structured debugging information
  - Handles complex input parsing (JSON, ast.literal_eval, fallback to raw)
  - Filters out system libraries from trace to focus on user code

---

### 4. Plot Inspector (`plot_inspector.py`)

- **Function:** `inspect_plot(code: str, data: Any, entry_point: str) -> str`
- **Purpose:** Verify visual properties of matplotlib plots without rendering images, enabling automated visual debugging.
- **Parameters:**
  - `code`: Python code that defines a plotting function
  - `data`: Data to pass to the plotting function (supports JSON parsing and ast.literal_eval)
  - `entry_point`: Name of the plotting function to call
- **Key Features:**
  - Executes plotting code using non-interactive 'Agg' backend to prevent display issues
  - In-process matplotlib interception with subprocess fallback for safety
  - Inspects plot objects for titles, axis labels, colors, line styles, and data series
  - Supports line plots, bar charts, and scatter plots with automatic type detection
  - Uses `PlotInspectionResult` and `PlotSeries` Pydantic models for structured analysis
  - 30-second timeout protection with automatic cleanup
  - Handles matplotlib color conversion to hex format
  - Returns detailed analysis of plot visual properties and data series

---

## Data Models

All tools use Pydantic models for structured data validation and result handling:

- **TestResult**: Structured test execution results with success status and feedback
- **LintResult** & **LintMessage**: Code quality analysis with scores and categorized issues  
- **DebugResult**: Execution tracing results with error details and variable states
- **PlotInspectionResult** & **PlotSeries**: Visual plot analysis with series data and properties

## Security Features

- **Subprocess Isolation**: All code execution happens in isolated subprocesses
- **Timeout Protection**: 15-30 second timeouts prevent infinite loops and hanging
- **Temporary File Management**: Automatic cleanup of temporary files
- **Safe Repr Handling**: Limited string representations to prevent memory issues
- **Library Filtering**: Traces focus on user code, filtering out system libraries
- **Non-Interactive Backends**: Matplotlib uses 'Agg' backend to prevent display issues
