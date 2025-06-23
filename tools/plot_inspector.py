"""
Matplotlib plot inspection tool for the Patchwork agent.

Allows the agent to verify the properties of a generated plot without
actually rendering or displaying it using in-process matplotlib interception.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Any, List, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


def _inspect_figure(fig) -> PlotInspectionResult:
    """Inspect a matplotlib figure and extract plot information."""
    result = PlotInspectionResult(
        success=False,
        title=None,
        xlabel=None,
        ylabel=None,
        series=[],
        summary="Plot generation failed.",
        error_message=None,
    )

    try:
        # Get the primary axes (usually the first one)
        if not fig.axes:
            result.summary = "No axes found in figure"
            return result

        ax = fig.axes[0]  # Primary axes

        # Extract basic plot properties
        result.title = ax.get_title() if ax.get_title() else None
        result.xlabel = ax.get_xlabel() if ax.get_xlabel() else None
        result.ylabel = ax.get_ylabel() if ax.get_ylabel() else None

        # Inspect lines, bars, scatter plots, etc.
        series_found = []

        # Check for line plots
        for line in ax.get_lines():
            try:
                color = line.get_color()
                # Handle matplotlib's color objects
                if hasattr(color, "value"):
                    color = color.value
                elif color == "auto":
                    color = "blue"  # default fallback

                # Convert to hex if possible
                try:
                    if color != "auto":
                        color_hex = matplotlib.colors.to_hex(color)
                    else:
                        color_hex = "#1f77b4"  # matplotlib default blue
                except (ValueError, TypeError):
                    color_hex = "#1f77b4"  # fallback

                series_found.append(
                    PlotSeries(
                        plot_type="line",
                        color=color_hex,
                        linestyle=line.get_linestyle(),
                        x_data=line.get_xdata().tolist(),
                        y_data=line.get_ydata().tolist(),
                        label=(
                            line.get_label()
                            if line.get_label() and not line.get_label().startswith("_")
                            else None
                        ),
                    )
                )
            except Exception as e:
                # If we can't inspect this line, add a fallback entry
                series_found.append(
                    PlotSeries(
                        plot_type="line",
                        color="#1f77b4",  # default blue
                        linestyle="-",
                        x_data=[],
                        y_data=[],
                        label=f"<error inspecting line: {str(e)}>",
                    )
                )

        # Check for bar plots
        for container in ax.containers:
            if isinstance(container, BarContainer):
                x_data = [rect.get_x() + rect.get_width() / 2 for rect in container]
                y_data = [rect.get_height() for rect in container]
                series_found.append(
                    PlotSeries(
                        plot_type="bar",
                        color="#1f77b4",  # default color for bars
                        linestyle=None,
                        x_data=x_data,
                        y_data=y_data,
                        label=(
                            container.get_label()
                            if hasattr(container, "get_label")
                            else None
                        ),
                    )
                )

        # Check for scatter plots
        for collection in ax.collections:
            if isinstance(collection, PathCollection):  # Scatter plots
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    series_found.append(
                        PlotSeries(
                            plot_type="scatter",
                            color="#1f77b4",  # default color
                            linestyle=None,
                            x_data=[p[0] for p in offsets],
                            y_data=[p[1] for p in offsets],
                            label=(
                                collection.get_label()
                                if hasattr(collection, "get_label")
                                else None
                            ),
                        )
                    )

        result.series = series_found
        result.success = True
        result.summary = (
            f"Plot inspected successfully. Found {len(series_found)} data series."
        )

    except Exception as e:
        result.summary = f"Failed to inspect plot: {type(e).__name__}"
        result.error_message = str(e)

    return result


def inspect_plot(code: str, data: Any, entry_point: str) -> str:
    """
    Executes plotting code and inspects the resulting matplotlib object using in-process interception.

    Args:
        code: Python code that defines a plotting function.
        data: The data to pass to the plotting function.
        entry_point: The name of the plotting function to call.

    Returns:
        A formatted string summarizing the plot's properties for the LLM agent.
    """
    if not code.strip():
        return "No code provided to inspect."

    # Parse data parameter if it's passed as a string
    if isinstance(data, str):
        try:
            # First try JSON parsing
            import json

            data = json.loads(data)
        except json.JSONDecodeError:
            try:
                # Then try ast.literal_eval for Python literals
                import ast

                data = ast.literal_eval(data)
            except (ValueError, SyntaxError):
                # If parsing fails, keep as string but warn
                logger.warning(f"Could not parse data parameter: {data}")

    # Store original matplotlib functions for restoration
    original_show = plt.show
    original_savefig = plt.savefig
    original_close = plt.close

    # Track captured figures
    captured_figures = []

    def capture_show(*args, **kwargs):
        """Intercept plt.show() calls to capture figures instead of displaying."""
        fig = plt.gcf()
        if fig not in captured_figures:
            captured_figures.append(fig)
        # Don't actually show the plot

    def capture_savefig(filename, *args, **kwargs):
        """Intercept plt.savefig() calls to capture figures."""
        fig = plt.gcf()
        if fig not in captured_figures:
            captured_figures.append(fig)
        # Don't actually save the file

    def capture_close(*args, **kwargs):
        """Intercept plt.close() calls to prevent premature closing."""
        # Don't actually close during inspection
        pass

    try:
        # Set up matplotlib to use non-interactive backend
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        # Monkey patch matplotlib functions
        plt.show = capture_show
        plt.savefig = capture_savefig
        plt.close = capture_close

        # Suppress matplotlib warnings during execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Execute the user code in a clean namespace
            namespace = {}
            exec(code, namespace)

            # Call the plotting function
            if entry_point not in namespace:
                return f"Error: Function '{entry_point}' not found in provided code"

            plotting_function = namespace[entry_point]
            result = plotting_function(data)

            # If function returns a figure, capture it
            if hasattr(result, "axes"):  # It's a figure
                captured_figures.append(result)
            elif hasattr(result, "figure"):  # It's an axes
                captured_figures.append(result.figure)

            # Also capture the current figure if it exists and has content
            current_fig = plt.gcf()
            if current_fig.axes and current_fig not in captured_figures:
                captured_figures.append(current_fig)

        # Inspect captured figures
        if not captured_figures:
            return "No plots were generated or captured during execution."

        # Inspect the primary figure (usually the last one created)
        primary_figure = captured_figures[-1]
        inspection_result = _inspect_figure(primary_figure)

        return _format_for_agent(inspection_result)

    except Exception as e:
        return f"Error during plot inspection: {type(e).__name__}: {str(e)}"

    finally:
        # Always restore original matplotlib functions
        plt.show = original_show
        plt.savefig = original_savefig
        plt.close = original_close

        # Clean up captured figures to prevent memory leaks
        for fig in captured_figures:
            try:
                original_close(fig)
            except:
                pass

        # Reset backend
        try:
            matplotlib.use(original_backend)
        except:
            pass


def inspect_plot_fallback(code: str, data: Any, entry_point: str) -> str:
    """
    Fallback subprocess-based plot inspection for edge cases.

    This is the original implementation kept as a backup.
    """
    if not code.strip():
        return "No code provided to inspect."

    # Parse data parameter if it's passed as a string
    if isinstance(data, str):
        try:
            # First try JSON parsing
            import json

            data = json.loads(data)
        except json.JSONDecodeError:
            try:
                # Then try ast.literal_eval for Python literals
                import ast

                data = ast.literal_eval(data)
            except (ValueError, SyntaxError):
                # If parsing fails, keep as string but warn
                logger.warning(f"Could not parse data parameter in fallback: {data}")

    runner_script = f"""
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

# Override plt.show to prevent hanging
original_show = plt.show
def non_blocking_show(*args, **kwargs):
    # Don't actually show the plot to prevent hanging
    pass
plt.show = non_blocking_show

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
        
        # Try to call the function - it might return fig, ax or just create a plot
        try:
            result = {entry_point}(data_input)
            if isinstance(result, tuple) and len(result) == 2:
                fig, ax = result
            else:
                # Function doesn't return fig, ax - get current figure
                fig = plt.gcf()
                ax = plt.gca()
        except Exception:
            # If function call fails, try to get current figure anyway
            fig = plt.gcf()
            ax = plt.gca()

        results["title"] = ax.get_title()
        results["xlabel"] = ax.get_xlabel()
        results["ylabel"] = ax.get_ylabel()

        # Inspect lines, bars, scatter plots, etc.
        for line in ax.get_lines():
            try:
                color = line.get_color()
                # Handle matplotlib's 'auto' color objects
                if hasattr(color, 'value'):
                    color = color.value
                elif color == 'auto':
                    color = 'blue'  # default fallback
                
                results["series"].append({{
                    "plot_type": "line",
                    "color": matplotlib.colors.to_hex(color) if color != 'auto' else '#1f77b4',
                    "linestyle": line.get_linestyle(),
                    "x_data": line.get_xdata().tolist(),
                    "y_data": line.get_ydata().tolist(),
                    "label": line.get_label(),
                }})
            except Exception as e:
                # If we can't inspect this line, add a fallback entry
                results["series"].append({{
                    "plot_type": "line",
                    "color": "#1f77b4",  # default blue
                    "linestyle": "-",
                    "x_data": [],
                    "y_data": [],
                    "label": f"<error inspecting line: {{str(e)}}>",
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
        results["summary"] = f"Plot inspected successfully. Found {{len(results['series'])}} data series."

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
                label_text = (
                    s.label if s.label and not s.label.startswith("<") else "None"
                )
                output.append(
                    f"  {i}. Type: {s.plot_type}, Color: {s.color}, Style: {s.linestyle}, Label: '{label_text}', Points: {len(s.x_data)}"
                )
    else:
        output.append(f"  - Error: {result.error_message}")

    return "\\n".join(output)
