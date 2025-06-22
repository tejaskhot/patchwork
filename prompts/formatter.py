"""
Dynamic prompt formatter for the Patchwork agent.

This module formats the ReAct prompt template with dynamic tool descriptions
and problem-specific information.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from tools.registry import tool_registry


class PromptFormatter:
    """Formats the ReAct prompt template with dynamic content."""

    def __init__(self):
        self.template_path = Path(__file__).parent / "system_prompt.txt"
        self._load_template()

    def _load_template(self) -> None:
        """Load the prompt template from file."""
        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def format_prompt(
        self,
        entry_point: str,
        goal: str,
        quality_criteria: str,
        tests: List[Dict[str, Any]],
        broken_code: str,
    ) -> str:
        """
        Format the complete prompt with problem details and dynamic tool descriptions.

        Args:
            entry_point: Name of the function to debug
            goal: High-level description of what the function should do
            quality_criteria: Specific quality requirements for the solution
            tests: List of test cases with inputs and expected outputs
            broken_code: The current broken implementation

        Returns:
            Formatted prompt ready for the LLM
        """
        # Get dynamic tool descriptions
        tool_descriptions = tool_registry.get_tool_descriptions()

        # Format test cases for readability
        tests_formatted = self._format_tests(tests)

        # Replace template placeholders
        formatted_prompt = self.template.format(
            tool_descriptions=tool_descriptions,
            entry_point=entry_point,
            goal=goal,
            quality_criteria=quality_criteria,
            tests_formatted=tests_formatted,
            broken_code=broken_code,
        )

        return formatted_prompt

    def _format_tests(self, tests: List[Dict[str, Any]]) -> str:
        """Format test cases for clear presentation in the prompt."""
        if not tests:
            return "No test cases provided."

        formatted_tests = []
        for i, test in enumerate(tests, 1):
            test_input = test.get("input", "Not specified")
            expected = test.get("expected", "Not specified")

            # Format input and expected output nicely
            input_str = (
                json.dumps(test_input, indent=2)
                if isinstance(test_input, (dict, list))
                else repr(test_input)
            )
            expected_str = (
                json.dumps(expected, indent=2)
                if isinstance(expected, (dict, list))
                else repr(expected)
            )

            formatted_test = f"**Test {i}:**\n"
            formatted_test += f"Input: {input_str}\n"
            formatted_test += f"Expected: {expected_str}"

            formatted_tests.append(formatted_test)

        return "\n\n".join(formatted_tests)


# Global formatter instance
prompt_formatter = PromptFormatter()
