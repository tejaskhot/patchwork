"""
Python debugger tool for the Patchwork agent.
Provides detailed execution tracing and variable inspection when code fails.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

DEBUG_TIMEOUT_SECONDS = 30


class DebugResult(BaseModel):
    """Results from debugging a failed code execution."""

    success: bool = Field(..., description="Whether the code executed without error")
    error_line: Optional[int] = Field(
        None, description="Line number where error occurred"
    )
    error_type: Optional[str] = Field(
        None, description="Type of exception that occurred"
    )
    error_message: Optional[str] = Field(None, description="Exception message")
    local_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Local variables at the point of failure"
    )
    execution_trace: str = Field(
        default="", description="Trace of execution leading to the error"
    )
    summary: str = Field(..., description="Human-readable summary for the agent")

    @field_validator("error_line")
    @classmethod
    def validate_error_line(cls, v):
        if v is not None and v < 0:
            raise ValueError("Error line number must be non-negative")
        return v


def run_with_debugger(code: str, test_input: Any, entry_point: str) -> str:
    """
    Execute code with debugging instrumentation to capture failure details.

    Args:
        code: Python code to debug
        test_input: Input to pass to the function. Note: If the function
            expects a single list as its argument, it must be wrapped in an
            outer list to prevent it from being unpacked. For example, to
            debug `my_func([1, 2])`, `test_input` should be `[[1, 2]]`.
        entry_point: Name of the function to call

    Returns:
        String containing debugging information that the LLM can understand
    """
    if not code.strip():
        return "No code provided to debug"

    # Create the debugging script that will trace execution
    debug_script = f"""
import sys
import traceback
import json
from types import FrameType
from typing import Any, Dict, Optional

# Set up matplotlib to prevent interactive plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Override plt.show to prevent hanging
original_show = plt.show
def non_blocking_show(*args, **kwargs):
    # Don't actually show the plot to prevent hanging
    pass
plt.show = non_blocking_show

# --- Start of Agent's Code ---
{code}
# --- End of Agent's Code ---

class ExecutionTracer:
    def __init__(self):
        self.trace_lines = []
        self.last_locals = {{}}
        self.error_line = None
        self.error_type = None
        self.error_message = None
        self.max_trace_lines = 1000  # Limit trace accumulation
        self.target_file_name = None
        
    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Optional['ExecutionTracer']:
        try:
            if event == 'line':
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                
                # Initialize target file name from the first file we see that contains our function
                if self.target_file_name is None and '{entry_point}' in frame.f_code.co_names:
                    self.target_file_name = filename
                
                # Only trace lines from our target script or temp files
                should_trace = (
                    filename.endswith('.py') and 
                    (
                        self.target_file_name and filename == self.target_file_name or
                        '/tmp/' in filename or  # temp files
                        'temp' in filename.lower() or
                        filename.endswith('.py') and not any(skip in filename.lower() for skip in [
                            'matplotlib', 'numpy', 'site-packages', 'lib/python', 
                            'importlib', 'pkgutil', 'inspect', 'linecache',
                            '/opt/', '/usr/', '/Library/', 'conda', 'pip'
                        ])
                    ) and
                    len(self.trace_lines) < self.max_trace_lines  # Prevent infinite accumulation
                )
                
                if should_trace:
                    self.trace_lines.append(f"Line {{lineno}}")
                    # Capture local variables only if we're in our target function
                    if '{entry_point}' in frame.f_code.co_name or frame.f_code.co_name == '{entry_point}':
                        try:
                            safe_locals = {{}}
                            for k, v in frame.f_locals.items():
                                if not k.startswith('__'):
                                    try:
                                        safe_locals[k] = repr(v)[:200]  # Limit repr length
                                    except:
                                        safe_locals[k] = "<repr failed>"
                            self.last_locals = safe_locals
                        except:
                            # If locals access fails, just continue
                            pass
                    
            elif event == 'exception':
                self.error_line = frame.f_lineno
                exc_type, exc_value, exc_tb = arg
                self.error_type = exc_type.__name__
                self.error_message = str(exc_value)
        except:
            # If tracing itself fails, just continue silently
            pass
            
        return self.trace_calls

def debug_execution():
    tracer = ExecutionTracer()
    
    # Handle test_input that might be passed as string
    test_input_raw = {repr(test_input)}
    
    # Try to parse it if it's a string representation
    if isinstance(test_input_raw, str):
        try:
            # First try JSON parsing
            test_input = json.loads(test_input_raw)
        except json.JSONDecodeError:
            try:
                # Then try ast.literal_eval for Python literals
                import ast
                test_input = ast.literal_eval(test_input_raw)
            except (ValueError, SyntaxError):
                # If all else fails, use the raw string
                test_input = test_input_raw
    else:
        test_input = test_input_raw
    
    try:
        # Enable tracing
        sys.settrace(tracer.trace_calls)
        
        # Execute the function
        if isinstance(test_input, list):
            result = {entry_point}(*test_input)
        else:
            result = {entry_point}(test_input)
            
        # If we get here, execution was successful
        debug_result = {{
            "success": True,
            "error_line": None,
            "error_type": None,
            "error_message": None,
            "local_variables": tracer.last_locals,
            "execution_trace": " -> ".join(tracer.trace_lines[-20:]),  # Last 20 steps only
            "summary": f"Code executed successfully. Result: {{repr(result)[:200]}}"  # Limit result length
        }}
        
    except Exception as e:
        # Capture detailed error information
        debug_result = {{
            "success": False,
            "error_line": tracer.error_line or sys.exc_info()[2].tb_lineno,
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],  # Limit error message length
            "local_variables": tracer.last_locals,
            "execution_trace": " -> ".join(tracer.trace_lines[-50:]),  # Last 50 steps for errors
            "summary": f"Error on line {{tracer.error_line or 'unknown'}}: {{type(e).__name__}}: {{str(e)[:200]}}"
        }}
        
    finally:
        # Disable tracing
        sys.settrace(None)
    
    # Output results as JSON for parsing
    print(json.dumps(debug_result))

if __name__ == "__main__":
    debug_execution()
"""

    script_path = ""
    try:
        # Write debug script to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py", encoding="utf-8"
        ) as f:
            f.write(debug_script)
            script_path = f.name

        # Execute the debug script
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=DEBUG_TIMEOUT_SECONDS,
            encoding="utf-8",
        )

        if process.returncode == 0 and process.stdout.strip():
            try:
                # Parse JSON output from debug script - take only the last line that looks like JSON
                lines = process.stdout.strip().split("\n")
                json_line = None

                # Find the last line that starts with '{' (our JSON output)
                for line in reversed(lines):
                    if line.strip().startswith("{"):
                        json_line = line.strip()
                        break

                if json_line:
                    debug_data = json.loads(json_line)
                    result = DebugResult(**debug_data)
                    return _format_for_agent(result)
                else:
                    return f"No valid JSON output found in debug script output:\\n{process.stdout}"

            except (json.JSONDecodeError, ValueError) as e:
                return (
                    f"Failed to parse debug output: {e}\\nRaw output: {process.stdout}"
                )
        else:
            error_msg = (
                process.stderr.strip() or "Debug script failed with no error message"
            )
            return f"Debug execution failed: {error_msg}"

    except subprocess.TimeoutExpired:
        return f"Debug execution timed out after {DEBUG_TIMEOUT_SECONDS} seconds"
    except FileNotFoundError:
        return "Python interpreter not found"
    except Exception as e:
        return f"Debug setup failed: {str(e)}"
    finally:
        # Clean up temporary file
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except OSError:
                pass


def _format_for_agent(result: DebugResult) -> str:
    """
    Format debug results in a way that's easy for the LLM agent to understand.

    Returns a clear, structured string that the agent can parse and reason about.
    """
    output = [f"Debug Summary: {result.summary}"]

    if not result.success:
        output.append(f"\\nError Details:")
        output.append(f"  - Line: {result.error_line}")
        output.append(f"  - Type: {result.error_type}")
        output.append(f"  - Message: {result.error_message}")

        if result.local_variables:
            output.append(f"\\nLocal Variables at Error:")
            # Limit to 8 vars
            for var_name, var_value in list(result.local_variables.items())[:8]:
                output.append(f"  - {var_name}: {var_value}")

            if len(result.local_variables) > 8:
                output.append(
                    f"  ... and {len(result.local_variables) - 8} more variables"
                )

    if result.execution_trace:
        output.append(f"\\nExecution Trace: {result.execution_trace}")

    return "\\n".join(output)
