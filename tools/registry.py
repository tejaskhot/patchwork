"""
Dynamic tool registry for the Patchwork agent.

This module automatically discovers available tools and generates their descriptions
from function signatures and docstrings, enabling extensible tool management.
"""

import inspect
from typing import Callable, Dict, List

from . import debugger, linter, plot_inspector, test_harness


class ToolRegistry:
    """Registry for dynamically discovering and managing debugging tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._discover_tools()

    def _discover_tools(self) -> None:
        """Automatically discover all available tools from the tools package."""
        # Import all tool modules and extract their main functions
        tool_modules = {
            "run_tests": test_harness.run_tests,
            "lint": linter.lint,
            "run_with_debugger": debugger.run_with_debugger,
            "inspect_plot": plot_inspector.inspect_plot,
        }

        for name, func in tool_modules.items():
            if callable(func):
                self._tools[name] = func

    def get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions for the prompt."""
        descriptions = []

        for name, func in self._tools.items():
            desc = self._generate_tool_description(name, func)
            descriptions.append(desc)

        return "\n\n".join(descriptions)

    def _generate_tool_description(self, name: str, func: Callable) -> str:
        """Generate a description for a single tool from its signature and docstring."""
        # Extract docstring
        docstring = inspect.getdoc(func) or "No description available."

        # Extract signature
        sig = inspect.signature(func)

        # Generate parameter descriptions
        params = []
        for param_name, param in sig.parameters.items():
            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else "Any"
            )
            if hasattr(param_type, "__name__"):
                type_str = param_type.__name__
            else:
                type_str = str(param_type)

            params.append(f"{param_name}: {type_str}")

        param_str = ", ".join(params)

        # Extract return type
        return_type = (
            sig.return_annotation
            if sig.return_annotation != inspect.Parameter.empty
            else "Any"
        )
        if hasattr(return_type, "__name__"):
            return_str = return_type.__name__
        else:
            return_str = str(return_type)

        # Format the description
        description = f"**{name}({param_str}) -> {return_str}**\n"

        # Add a concise summary from the docstring (first line)
        if docstring:
            summary = docstring.split("\n")[0].strip()
            description += f"{summary}"

        return description

    def get_tool(self, name: str) -> Callable:
        """Get a tool function by name."""
        if name not in self._tools:
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters."""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"

        try:
            tool_func = self._tools[tool_name]
            result = tool_func(**kwargs)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"


# Global registry instance
tool_registry = ToolRegistry()
