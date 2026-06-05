"""Offline unit tests for fastllm.tool_healing.

These tests require no API access. They verify that text-encoded tool calls in
several common malformed formats are recovered into the canonical OpenAI
structure, and that unknown tool names are ignored.
"""

import json
import unittest

from fastllm.tool_healing import heal_tool_calls, strip_tool_call_markup


# A fake tool_map only needs the keys (names). Values are unused by healing.
TOOL_MAP = {"run_bash": object(), "list_windows": object()}


class TestToolHealing(unittest.TestCase):
    def test_empty_tool_map_returns_nothing(self):
        text = '<tool_call>{"name": "run_bash", "arguments": {}}</tool_call>'
        self.assertEqual(heal_tool_calls(text, "", {}), [])

    def test_plain_text_no_tool_call(self):
        self.assertEqual(
            heal_tool_calls("Just a normal answer.", "", TOOL_MAP), []
        )

    def test_hermes_json_tool_call(self):
        text = (
            'Sure!\n<tool_call>{"name": "run_bash", '
            '"arguments": {"command": "ls -la"}}</tool_call>'
        )
        calls = heal_tool_calls(text, "", TOOL_MAP)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")
        self.assertEqual(calls[0]["type"], "function")
        self.assertTrue(calls[0]["id"])
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["command"], "ls -la")

    def test_qwen_xml_with_parameters(self):
        text = (
            "<tool_call><function=run_bash>"
            "<parameter=command>whoami</parameter>"
            "</function></tool_call>"
        )
        calls = heal_tool_calls(text, "", TOOL_MAP)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["command"], "whoami")

    def test_bare_function_block_no_args(self):
        # The exact failure case reported: a no-arg XML function block.
        text = "<tool_call>\n<function=list_windows>\n</function>\n</tool_call>"
        calls = heal_tool_calls(text, "", TOOL_MAP)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_windows")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_unknown_tool_name_ignored(self):
        # 'terminal' is not a known tool, must not be healed.
        text = "<tool_call>\n<function=terminal>\n</function>\n</tool_call>"
        self.assertEqual(heal_tool_calls(text, "", TOOL_MAP), [])

    def test_tool_call_in_reasoning_content(self):
        reasoning = (
            'I should run a command.\n<tool_call>'
            '{"name": "run_bash", "arguments": {"command": "pwd"}}'
            "</tool_call>"
        )
        calls = heal_tool_calls("", reasoning, TOOL_MAP)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")

    def test_fenced_json_code_block(self):
        text = '```json\n{"name": "run_bash", "arguments": {"command": "id"}}\n```'
        calls = heal_tool_calls(text, "", TOOL_MAP)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "run_bash", "arguments": {"command": "ls"}}'
            "</tool_call>\n"
            '<tool_call>{"name": "list_windows", "arguments": {}}</tool_call>'
        )
        calls = heal_tool_calls(text, "", TOOL_MAP)
        self.assertEqual(len(calls), 2)
        names = [c["function"]["name"] for c in calls]
        self.assertEqual(names, ["run_bash", "list_windows"])

    def test_strip_markup(self):
        text = (
            "Vou verificar.\n<tool_call>"
            '{"name": "run_bash", "arguments": {}}</tool_call>'
        )
        self.assertEqual(strip_tool_call_markup(text), "Vou verificar.")

    def test_strip_markup_function_block(self):
        text = "Hi <function=run_bash><parameter=command>ls</parameter></function>"
        self.assertEqual(strip_tool_call_markup(text), "Hi")


if __name__ == "__main__":
    unittest.main()
