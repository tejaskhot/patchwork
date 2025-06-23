"""
Dynamic tool registry for the Patchwork agent.

This module automatically discovers available tools and generates their descriptions
from function signatures and docstrings, enabling extensible tool management.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from . import debugger, linter, plot_inspector, test_harness

logger = logging.getLogger(__name__)


class ParameterValidationResult:
    """Result of parameter validation for a tool call."""

    def __init__(
        self, is_valid: bool, error_message: str = "", missing_params: List[str] = None
    ):
        self.is_valid = is_valid
        self.error_message = error_message
        self.missing_params = missing_params or []

    @property
    def error_message_with_examples(self) -> str:
        """Get error message with helpful examples."""
        return self.error_message


class ToolRegistry:
    """Registry for dynamically discovering and managing debugging tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions_cache: Optional[str] = None
        self._litellm_schemas_cache: Optional[List[Dict[str, Any]]] = None
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

        discovered_count = 0
        for name, func in tool_modules.items():
            if callable(func):
                self._tools[name] = func
                discovered_count += 1
                logger.debug(f"Registered tool: {name}")

        logger.info(f"Discovered {discovered_count} tools")

    def get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions for the prompt."""
        if self._tool_descriptions_cache is not None:
            return self._tool_descriptions_cache

        descriptions = []
        for name, func in self._tools.items():
            try:
                desc = self._generate_tool_description(name, func)
                descriptions.append(desc)
            except Exception as e:
                logger.error(f"Failed to generate description for tool {name}: {e}")
                # Add fallback description
                descriptions.append(f"`{name}()` - Tool description unavailable")

        self._tool_descriptions_cache = "\n\n".join(descriptions)
        return self._tool_descriptions_cache

    def _generate_tool_description(self, name: str, func: Callable) -> str:
        """Generate a description for a single tool from its signature and docstring."""
        try:
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

                # Format the description as a clean code signature with full docstring
            signature = f"{name}({param_str}) -> {return_str}"

            # Include the COMPLETE docstring, not just the first line
            if docstring:
                # Clean up the docstring formatting while preserving structure
                cleaned_docstring = docstring.strip()
                description = f"`{signature}`\n\n{cleaned_docstring}"
            else:
                description = f"`{signature}`\nNo description available."

            return description
        except Exception as e:
            logger.error(f"Error generating description for {name}: {e}")
            return f"`{name}()` - Description generation failed"

    def get_litellm_tools_schema(self) -> List[Dict[str, Any]]:
        """Get LiteLLM-compatible tool schemas for all registered tools."""
        if self._litellm_schemas_cache is not None:
            return self._litellm_schemas_cache

        tools = []
        for tool_name, tool_func in self._tools.items():
            try:
                schema = self._generate_tool_schema(tool_name, tool_func)
                tools.append(schema)
            except Exception as e:
                logger.error(f"Failed to generate schema for tool {tool_name}: {e}")
                # Add fallback schema
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": f"Tool {tool_name} (schema generation failed)",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    }
                )

        self._litellm_schemas_cache = tools
        logger.debug(f"Generated schemas for {len(tools)} tools")
        return self._litellm_schemas_cache

    def _generate_tool_schema(
        self, tool_name: str, tool_func: Callable
    ) -> Dict[str, Any]:
        """Generate a single tool schema for LiteLLM."""
        sig = inspect.signature(tool_func)

        # Build parameter schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else str
            )

            # Convert Python types to JSON schema types
            if param_type == str:
                json_type = "string"
            elif param_type == int:
                json_type = "integer"
            elif param_type == bool:
                json_type = "boolean"
            elif param_type == float:
                json_type = "number"
            elif hasattr(param_type, "__origin__"):
                # Handle Union types (like Union[str, List[Dict]])
                if param_type.__origin__ is Union:
                    # For tests parameter specifically, we want it to accept arrays
                    if param_name == "tests":
                        properties[param_name] = {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "JSON string of test cases",
                                },
                                {
                                    "type": "array",
                                    "description": "Array of test case objects",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "input": {"description": "Test input"},
                                            "expected": {
                                                "description": "Expected output"
                                            },
                                        },
                                        "required": ["input", "expected"],
                                    },
                                },
                            ]
                        }
                        # Don't continue here - let it fall through to check if required
                    else:
                        json_type = "string"  # Default for other unions
                        properties[param_name] = {"type": json_type}
                else:
                    json_type = "string"  # Default fallback
                    properties[param_name] = {"type": json_type}
            else:
                json_type = "string"  # Default fallback
                properties[param_name] = {"type": json_type}

            # Add default property if not already set by special handling above
            if param_name not in properties:
                properties[param_name] = {"type": json_type}

            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": inspect.getdoc(tool_func) or f"Execute {tool_name} tool",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _validate_parameters(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> ParameterValidationResult:
        """Validate parameters for a tool call and return detailed validation result."""
        if tool_name not in self._tools:
            return ParameterValidationResult(
                False,
                f"Error: Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}",
            )

        tool_func = self._tools[tool_name]
        sig = inspect.signature(tool_func)

        # Get required parameters
        required_params = [
            param.name
            for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
        ]

        # Check for missing parameters
        missing_params = [param for param in required_params if param not in kwargs]

        if missing_params:
            # Generate helpful examples based on tool type
            example_params = self._generate_example_parameters(
                tool_name, required_params
            )
            suggestion = f"{tool_name}({', '.join(example_params)})"

            error_msg = (
                f"Error: Tool '{tool_name}' missing required parameters: {missing_params}. "
                f"Required: {required_params}\n\n"
                f"Example: {suggestion}"
            )

            return ParameterValidationResult(False, error_msg, missing_params)

        return ParameterValidationResult(True)

    def _generate_example_parameters(
        self, tool_name: str, required_params: List[str]
    ) -> List[str]:
        """Generate helpful example parameters for a specific tool."""
        examples = []

        for param in required_params:
            if param == "code":
                if tool_name == "inspect_plot":
                    examples.append(
                        'code="def plot_func(data): import matplotlib.pyplot as plt; plt.plot(data); plt.show()"'
                    )
                else:
                    examples.append('code="def my_func(x): return x + 1"')
            elif param == "tests":
                examples.append('tests=\'[{"input": [1, 2], "expected": 3}]\'')
            elif param == "entry_point":
                if tool_name == "inspect_plot":
                    examples.append('entry_point="plot_func"')
                else:
                    examples.append('entry_point="my_func"')
            elif param == "data":
                if tool_name == "inspect_plot":
                    examples.append('data={"x": [1, 2, 3], "y": [4, 5, 6]}')
                else:
                    examples.append('data="sample_data"')
            else:
                examples.append(f'{param}="value"')

        return examples

    def _generate_tool_not_found_error(self, tool_name: str) -> str:
        """Generate a helpful error message when a tool is not found."""
        available = list(self._tools.keys())
        return f"Error: Tool '{tool_name}' not found. " f"Available tools: {available}"

    def get_tool(self, name: str) -> Callable:
        """Get a tool function by name."""
        if name not in self._tools:
            available = list(self._tools.keys())
            logger.warning(f"Tool '{name}' not found. Available: {available}")
            raise ValueError(f"Tool '{name}' not found. Available tools: {available}")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def _execute_with_monitoring(self, tool_name: str, **kwargs) -> str:
        """Execute a tool with monitoring and error handling."""
        tool_func = self._tools[tool_name]

        logger.debug(f"Executing tool {tool_name} with args: {list(kwargs.keys())}")

        try:
            result = tool_func(**kwargs)
            logger.debug(f"Tool {tool_name} completed successfully")

            # Simple output size guard to prevent token explosion
            result_str = str(result)
            MAX_OUTPUT_SIZE = 50000  # 50KB limit
            if len(result_str) > MAX_OUTPUT_SIZE:
                logger.warning(
                    f"Tool {tool_name} output truncated: {len(result_str):,} chars -> {MAX_OUTPUT_SIZE:,} chars"
                )
                return (
                    result_str[: MAX_OUTPUT_SIZE // 2]
                    + f"\n\n... [TRUNCATED: Output was {len(result_str):,} characters, showing first {MAX_OUTPUT_SIZE//2:,} chars] ...\n\n"
                    + result_str[-MAX_OUTPUT_SIZE // 4 :]
                )

            return result_str

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters."""
        # Validate tool exists
        if tool_name not in self._tools:
            return self._generate_tool_not_found_error(tool_name)

        try:
            # Special handling for plot inspection tests
            if (
                tool_name == "run_tests"
                and "test_type" in kwargs
                and kwargs["test_type"] == "plot_inspection"
            ):
                logger.debug("Redirecting plot test to plot inspector")
                # Extract test data for plot inspection
                tests = kwargs.get("tests", [])
                if isinstance(tests, str):
                    import json

                    tests = json.loads(tests)

                if tests and len(tests) > 0:
                    test_input = tests[0].get("input", {})
                else:
                    test_input = {}

                # Call plot inspector instead
                return self._tools["inspect_plot"](
                    code=kwargs.get("code", ""),
                    data=test_input,
                    entry_point=kwargs.get("entry_point", ""),
                )

            # Validate required parameters with enhanced error messages
            validation_result = self._validate_parameters(tool_name, kwargs)
            if not validation_result.is_valid:
                logger.error(
                    f"Parameter validation failed for {tool_name}: {validation_result.missing_params}"
                )
                return validation_result.error_message_with_examples

            # Execute with monitoring
            return self._execute_with_monitoring(tool_name, **kwargs)

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clear_cache(self) -> None:
        """Clear cached descriptions and schemas."""
        self._tool_descriptions_cache = None
        self._litellm_schemas_cache = None
        logger.debug("Tool registry cache cleared")

    def reload_tools(self) -> None:
        """Reload all tools and clear caches."""
        self._tools.clear()
        self.clear_cache()
        self._discover_tools()
        logger.info("Tools reloaded")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "cache_populated": {
                "descriptions": self._tool_descriptions_cache is not None,
                "schemas": self._litellm_schemas_cache is not None,
            },
        }


# Global registry instance
tool_registry = ToolRegistry()
