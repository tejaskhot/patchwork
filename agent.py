"""
Patchwork: Autonomous Python debugging agent using LiteLLM.

This agent uses the tools registry and react prompt to systematically debug
and fix Python code issues.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import litellm
from pydantic import BaseModel, Field, field_validator

from tools.registry import (
    tool_registry,  # Expose default registry instance for easy patching in tests and factory method
)
from tools.registry import ToolRegistry

# Default prompt template paths
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")
DEFAULT_USER_PROMPT_PATH = Path("prompts/user_message.txt")

# Set up logging
logger = logging.getLogger(__name__)

# LiteLLM message type
LiteLLMMessage = Dict[str, Union[str, List[Dict[str, Any]]]]


class ToolResult(TypedDict):
    """Structured result of a single tool call."""

    tool_call_id: str
    content: str
    is_error: bool


class Step(BaseModel):
    """A single step in the agent execution process."""

    iteration: int
    user_message: str
    assistant_response: str
    tool_calls: List[str] = Field(default_factory=list)
    tool_results: List[str] = Field(default_factory=list)  # For display/logging
    tool_results_structured: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # For evaluation


class RunLog(BaseModel):
    """Complete log of an agent run."""

    steps: List[Step] = Field(default_factory=list)
    final_code: Optional[str] = None
    status: Optional[str] = None  # "success", "failed", "timeout"


class ProblemContext(BaseModel):
    """Context for the debugging problem."""

    entry_point: str
    goal: str
    quality_criteria: str
    tests_formatted: str
    broken_code: str

    @field_validator("broken_code")
    def validate_broken_code(cls, v):
        """Validate broken code input."""
        if not v.strip():
            raise ValueError("broken_code cannot be empty")
        if len(v) > 50000:  # 50KB limit
            raise ValueError("broken_code too large (max 50KB)")
        return v

    @field_validator("entry_point")
    def validate_entry_point(cls, v):
        """Validate entry point is a valid Python identifier."""
        if not v.strip():
            raise ValueError("entry_point cannot be empty")
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("entry_point must be a valid Python identifier")
        return v


class PatchworkAgent:
    """An autonomous LLM-powered agent for repairing Python code via tool-assisted reasoning."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: str = "gpt-4.1-nano",
        max_iterations: int = 10,
        temperature: float = 0.1,
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None,
        max_tokens: int = 4000,
        message_window_size: int = 500,  # Limit conversation length
    ):
        """
        Initialize the Patchwork agent.

        Args:
            tool_registry: ToolRegistry instance for dependency injection
            model: LiteLLM model identifier (e.g., "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1")
            max_iterations: Maximum debugging iterations before giving up
            temperature: Model temperature for reasoning
            system_prompt_path: Path to system prompt template (defaults to DEFAULT_SYSTEM_PROMPT_PATH)
            user_prompt_path: Path to user message template (defaults to DEFAULT_USER_PROMPT_PATH)
            max_tokens: Maximum tokens per model call
            message_window_size: Maximum messages to keep in conversation memory
        """
        self.tool_registry = tool_registry
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.system_prompt_path = system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH
        self.user_prompt_path = user_prompt_path or DEFAULT_USER_PROMPT_PATH
        self.max_tokens = max_tokens
        self.message_window_size = message_window_size

        # Initialize run tracking with Pydantic model
        self.run_log = RunLog()

        # Cache tool schemas (performance optimization)
        self._tool_schemas = self.tool_registry.get_litellm_tools_schema()
        logger.info(f"Initialized agent with {len(self._tool_schemas)} tools")

        # Load the prompt templates
        self.system_prompt_template = self._load_system_prompt_template()
        self.user_prompt_template = self._load_user_prompt_template()

        # Configure LiteLLM logging
        if os.getenv("DEBUG_LITELLM"):
            litellm.set_verbose = True
        else:
            litellm.suppress_debug_info = True

            for logger_name in logging.Logger.manager.loggerDict:
                if "litellm" in logger_name.lower():
                    logging.getLogger(logger_name).setLevel(logging.WARNING)

    def _load_system_prompt_template(self) -> str:
        """Load the system prompt template from file."""
        if not self.system_prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt template not found at {self.system_prompt_path}. "
                "Did you forget to generate or copy system_prompt.txt?"
            )

        return self.system_prompt_path.read_text(encoding="utf-8")

    def _load_user_prompt_template(self) -> str:
        """Load the user message template from file."""
        if not self.user_prompt_path.exists():
            raise FileNotFoundError(
                f"User prompt template not found at {self.user_prompt_path}. "
                "Did you forget to generate or copy user_message.txt?"
            )

        return self.user_prompt_path.read_text(encoding="utf-8")

    def _format_system_message(self) -> str:
        """Format the system prompt with tool descriptions."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        return self.system_prompt_template.format(tool_descriptions=tool_descriptions)

    def _format_user_message(self, problem: ProblemContext) -> str:
        """Format the user message template with problem context."""
        return self.user_prompt_template.format(
            entry_point=problem.entry_point,
            goal=problem.goal,
            quality_criteria=problem.quality_criteria,
            tests_formatted=problem.tests_formatted,
            broken_code=problem.broken_code,
        )

    def _call_model(
        self,
        messages: List[LiteLLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Call the LLM model with given messages and optional tools."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = litellm.completion(**kwargs)
            return response.choices[0].message
        except Exception as e:
            logger.error(f"Model call failed: {str(e)}")
            raise RuntimeError(f"Model call failed: {str(e)}")

    def _execute_tool_calls(self, tool_calls: List[Any]) -> List[ToolResult]:
        """
        Execute tool calls and return a list of structured results.

        Each result is a dictionary containing the tool_call_id and the
        content of the execution result or error message.
        """
        results: List[ToolResult] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            # All tool calls should have an ID. If not, it's an issue with the model/provider.
            if not tool_call_id:
                logger.warning(f"Tool call '{tool_name}' is missing an ID, skipping.")
                continue

            try:
                params = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON arguments for tool {tool_name}: {e}")
                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "content": f"Error: Invalid arguments format for tool '{tool_name}': {e}",
                        "is_error": True,
                    }
                )
                continue

            try:
                result_content = self.tool_registry.execute_tool(tool_name, **params)
                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "content": str(result_content),
                        "is_error": False,
                    }
                )
                logger.debug(f"Successfully executed tool {tool_name}")
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "content": f"Error executing tool '{tool_name}': {e}",
                        "is_error": True,
                    }
                )

        return results

    def _extract_final_solution(self, response: str) -> Optional[str]:
        """Extract the final code solution from <final> tags."""
        pattern = r"<final>\s*```python\s*(.*?)\s*```\s*</final>"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            solution = match.group(1).strip()
            logger.info("Final solution extracted successfully")
            return solution

        return None

    def _trim_messages(self, messages: List[LiteLLMMessage]) -> List[LiteLLMMessage]:
        """Trim messages to stay within memory limits while preserving context."""
        if len(messages) <= self.message_window_size:
            return messages

        logger.info(
            f"Trimming messages from {len(messages)} to {self.message_window_size}"
        )

        # Always keep the first message (initial prompt) and recent messages
        # With the higher limit (500), this should rarely be triggered
        recent_count = self.message_window_size - 1
        trimmed = [messages[0]] + messages[-recent_count:]
        logger.debug(f"Kept first message + {recent_count} recent messages")
        return trimmed

        return messages

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable and worth retrying."""
        recoverable_errors = [
            "rate limit",
            "timeout",
            "temporary",
            "connection",
            "network",
        ]
        error_msg = str(error).lower()
        return any(recoverable in error_msg for recoverable in recoverable_errors)

    def run(self, problem: ProblemContext) -> str:
        """
        Main execution loop that orchestrates the agent workflow.

        Args:
            problem: ProblemContext with all debugging information

        Returns:
            Final fixed code or error message
        """
        logger.info(f"Starting agent run for function: {problem.entry_point}")

        # Reset run log for new run
        self.run_log = RunLog()

        # Format system and user messages
        system_message = self._format_system_message()
        user_message = self._format_user_message(problem)

        # Initialize conversation with proper message structure
        messages: List[LiteLLMMessage] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        consecutive_errors = 0
        max_consecutive_errors = 3

        for iteration in range(self.max_iterations):
            try:
                logger.debug(f"Starting iteration {iteration}")

                # Trim messages if needed (memory management)
                messages = self._trim_messages(messages)

                # Call model with current messages and available tools
                response = self._call_model(messages, self._tool_schemas)

                # Log the step
                step = Step(
                    iteration=iteration,
                    user_message=messages[-1].get("content", "") if messages else "",
                    assistant_response=response.content or "",
                    tool_calls=[],
                    tool_results=[],
                )

                # Check if model wants to call tools
                if response.tool_calls:
                    logger.debug(
                        f"Model requested {len(response.tool_calls)} tool calls"
                    )

                    # Execute tool calls
                    tool_results = self._execute_tool_calls(response.tool_calls)

                    # Log tool usage
                    step.tool_calls = [
                        f"{tc.function.name}({tc.function.arguments})"
                        for tc in response.tool_calls
                    ]
                    step.tool_results = [
                        f"Tool: {tc.function.name}\nResult: {res['content']}"
                        for tc, res in zip(response.tool_calls, tool_results)
                    ]

                    # Store structured results for evaluation
                    step.tool_results_structured = []
                    for tc, res in zip(response.tool_calls, tool_results):
                        try:
                            # Try to parse as JSON for structured access
                            import json

                            parsed_result = (
                                json.loads(res["content"])
                                if isinstance(res["content"], str)
                                else res["content"]
                            )
                            step.tool_results_structured.append(
                                {"tool_name": tc.function.name, "result": parsed_result}
                            )
                        except (json.JSONDecodeError, TypeError):
                            # Fall back to string content
                            step.tool_results_structured.append(
                                {
                                    "tool_name": tc.function.name,
                                    "result": res["content"],
                                }
                            )

                    # Add assistant message with tool calls to conversation
                    messages.append(response.model_dump())

                    # Add tool results as separate messages
                    for result in tool_results:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": result["tool_call_id"],
                                "content": result["content"],
                            }
                        )

                    self.run_log.steps.append(step)
                    consecutive_errors = 0  # Reset error counter on success
                    continue

                # No tool calls - check for final solution
                if response.content:
                    final_code = self._extract_final_solution(response.content)
                    if final_code:
                        self.run_log.final_code = final_code
                        self.run_log.status = "success"
                        self.run_log.steps.append(step)
                        logger.info("Agent run completed successfully")
                        return final_code

                # Add regular assistant response to conversation
                messages.append({"role": "assistant", "content": response.content})
                self.run_log.steps.append(step)

                # If no tools called and no final solution, ask for clarification
                messages.append(
                    {
                        "role": "user",
                        "content": "Please continue working on the problem or provide your final solution using <final>```python ... ```</final> tags.",
                    }
                )

                consecutive_errors = 0  # Reset error counter on success

            except Exception as e:
                consecutive_errors += 1
                error_msg = f"Error in iteration {iteration}: {str(e)}"
                logger.error(error_msg)

                # Check if error is recoverable and we haven't exceeded retry limit
                if (
                    self._is_recoverable_error(e)
                    and consecutive_errors < max_consecutive_errors
                    and iteration < self.max_iterations - 1
                ):

                    logger.info(
                        f"Recoverable error detected, retrying... ({consecutive_errors}/{max_consecutive_errors})"
                    )

                    # Add error context to conversation for the model to learn from
                    messages.append(
                        {
                            "role": "user",
                            "content": f"An error occurred: {str(e)}. Please try a different approach.",
                        }
                    )
                    continue
                else:
                    # Fatal error or too many consecutive errors
                    self.run_log.status = "failed"
                    logger.error(
                        f"Fatal error or too many consecutive errors: {error_msg}"
                    )
                    return error_msg

        # Max iterations reached
        self.run_log.status = "timeout"
        timeout_msg = f"Max iterations ({self.max_iterations}) reached without finding a solution."
        logger.warning(timeout_msg)
        return timeout_msg

    def get_run_log(self) -> RunLog:
        """Get the complete agent run log."""
        return self.run_log

    def reset(self) -> None:
        """Reset the agent state for a new run."""
        self.run_log = RunLog()
        logger.debug("Agent state reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent run statistics."""
        return {
            "total_steps": len(self.run_log.steps),
            "total_tool_calls": sum(
                len(step.tool_calls) for step in self.run_log.steps
            ),
            "status": self.run_log.status,
            "has_solution": self.run_log.final_code is not None,
            "final_code_length": (
                len(self.run_log.final_code) if self.run_log.final_code else 0
            ),
        }


def create_agent(
    model: str = "gpt-4.1-nano",
    max_iterations: int = 10,
    temperature: float = 0.1,
    system_prompt_path: Optional[Path] = None,
    user_prompt_path: Optional[Path] = None,
    max_tokens: int = 4000,
    message_window_size: int = 500,
) -> PatchworkAgent:
    """
    Convenience factory function to create a PatchworkAgent with default registry.

    Args:
        model: LiteLLM model identifier (gpt-4.1-nano, gpt-4.1-mini, gpt-4.1)
        max_iterations: Maximum debugging iterations
        temperature: Model temperature for reasoning
        prompt_path: Path to prompt template
        max_tokens: Maximum tokens per model call
        message_window_size: Maximum messages to keep in memory

    Returns:
        Configured PatchworkAgent instance
    """
    # Use the module-level `tool_registry` imported at the top of this file so that
    # unit tests can patch `agent.tool_registry` reliably without needing to
    # intercept a fresh import inside this function.
    return PatchworkAgent(
        tool_registry=tool_registry,
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        max_tokens=max_tokens,
        message_window_size=message_window_size,
    )
