"""
Matplotlib plot inspection tool for the Patchwork agent.

Allows the agent to verify the properties of a generated plot without
actually rendering or displaying it.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PlotSeries(BaseModel):
    """Represents a single data series in a plot."""

    plot_type: str = Field(
        ..., description="Type of plot (e.g., 'line', 'bar', 'scatter')"
    )
    color: Optional[str] = Field(None, description="The color of the data series")
    linestyle: Optional[str] = Field(
        None, description="The style of the line (e.g., '--', ':')"
    )
    x_data: List[Any] = Field(..., description="Data points for the x-axis")
    y_data: List[Any] = Field(..., description="Data points for the y-axis")
    label: Optional[str] = Field(
        None, description="Label for the data series (for legends)"
    )


class PlotInspectionResult(BaseModel):
    """Structured result of a plot inspection."""

    success: bool = Field(
        ..., description="Whether the plot was generated successfully"
    )
    title: Optional[str] = Field(None, description="The title of the plot")
    xlabel: Optional[str] = Field(None, description="The label for the x-axis")
    ylabel: Optional[str] = Field(None, description="The label for the y-axis")
    series: List[PlotSeries] = Field(
        default_factory=list, description="Data series found in the plot"
    )
    summary: str = Field(..., description="Human-readable summary for the agent")
    error_message: Optional[str] = Field(
        None, description="Error message if plot generation failed"
    )


INSPECTION_TIMEOUT_SECONDS = 30


def inspect_plot(code: str, data: Any, entry_point: str) -> str:
    """
    Executes plotting code and inspects the resulting matplotlib object.

    Args:
        code: Python code that defines a plotting function. The function
            is expected to return a tuple of (figure, axes) from matplotlib.
        data: The data to pass to the plotting function.
        entry_point: The name of the plotting function to call.

    Returns:
        A formatted string summarizing the plot's properties for the LLM agent.
    """
    if not code.strip():
        return "No code provided to inspect."

    runner_script = f"""
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

# --- Start of Agent's Code ---
{code}
# --- End of Agent's Code ---

def get_plot_details():
    results = {{
        "success": False,
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "series": [],
        "summary": "Plot generation failed.",
        "error_message": None,
    }}
    try:
        data_input = {repr(data)}
        
        # The agent's function should return the fig and ax objects
        fig, ax = {entry_point}(data_input)

        results["title"] = ax.get_title()
        results["xlabel"] = ax.get_xlabel()
        results["ylabel"] = ax.get_ylabel()

        # Inspect lines, bars, scatter plots, etc.
        for line in ax.get_lines():
            results["series"].append({{
                "plot_type": "line",
                "color": matplotlib.colors.to_hex(line.get_color()),
                "linestyle": line.get_linestyle(),
                "x_data": line.get_xdata().tolist(),
                "y_data": line.get_ydata().tolist(),
                "label": line.get_label(),
            }})
        
        for container in ax.containers:
            if isinstance(container, BarContainer):
                x_data = [rect.get_x() + rect.get_width() / 2 for rect in container]
                y_data = [rect.get_height() for rect in container]
                results["series"].append({{
                    "plot_type": "bar",
                    "x_data": x_data,
                    "y_data": y_data,
                    "label": container.get_label(),
                }})

        for collection in ax.collections:
            if isinstance(collection, PathCollection): # Scatter plots
                offsets = collection.get_offsets().tolist()
                results["series"].append({{
                    "plot_type": "scatter",
                    "x_data": [p[0] for p in offsets],
                    "y_data": [p[1] for p in offsets],
                    "label": collection.get_label(),
                }})
        
        plt.close(fig) # Prevent memory leaks
        results["success"] = True
        results["summary"] = f"Plot inspected successfully. Found {len(results['series'])} data series."

    except Exception as e:
        results["summary"] = f"Failed to generate or inspect plot: {{type(e).__name__}}"
        results["error_message"] = str(e)
    
    print(json.dumps(results))

if __name__ == "__main__":
    get_plot_details()
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
            timeout=INSPECTION_TIMEOUT_SECONDS,
            encoding="utf-8",
        )

        if process.returncode == 0 and process.stdout:
            try:
                inspection_data = json.loads(process.stdout)
                result = PlotInspectionResult(**inspection_data)
                return _format_for_agent(result)
            except (json.JSONDecodeError, ValueError) as e:
                return f"Failed to parse plot inspection output: {e}\\nRaw Output: {process.stdout}"
        else:
            error_msg = (
                process.stderr.strip()
                or "Plot inspection script failed with no error message."
            )
            return f"Plot inspection failed: {error_msg}"

    except subprocess.TimeoutExpired:
        return f"Plot inspection timed out after {INSPECTION_TIMEOUT_SECONDS} seconds."
    except Exception as e:
        return f"Plot inspection setup failed: {str(e)}"
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def _format_for_agent(result: PlotInspectionResult) -> str:
    """Formats the plot inspection results into a human-readable string for the agent."""
    output = [f"Plot Inspection Summary: {result.summary}"]

    if result.success:
        output.append(f"  - Title: '{result.title or 'Not set'}'")
        output.append(f"  - X-Axis Label: '{result.xlabel or 'Not set'}'")
        output.append(f"  - Y-Axis Label: '{result.ylabel or 'Not set'}'")

        if result.series:
            output.append("\\nData Series Found:")
            for i, s in enumerate(result.series, 1):
                output.append(
                    f"  {i}. Type: {s.plot_type}, Color: {s.color}, Style: {s.linestyle}, Label: '{s.label or 'None'}', Points: {len(s.x_data)}"
                )
    else:
        output.append(f"  - Error: {result.error_message}")

    return "\\n".join(output)
