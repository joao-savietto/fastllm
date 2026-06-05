"""
This module provides an Agent class for interacting with the OpenAI API to
generate responses based on chat history.

Classes:
    Agent: A class to interact with the OpenAI API for generating AI responses.
"""

import base64
import json
import traceback
from typing import Any, Callable, Dict, Generator, List, Optional

import openai
from pydantic import BaseModel

from fastllm.decorators import pydantic_to_openai_schema, streamable_response
from fastllm.exceptions import EmptyPayload
from fastllm.store import ChatStorageInterface, InMemoryChatStorage
from fastllm.mcp_client import MCPClient
from fastllm.tool_healing import heal_tool_calls, strip_tool_call_markup


class Agent:
    def __init__(
        self,
        model: str = "gpt-5",
        base_url: str = "https://api.openai.com/v1/",
        api_key: str = "some-key",
        tools: List[Callable] = None,
        system_prompt: str = "",
        store: ChatStorageInterface = None,
        mcp_config_path: Optional[str] = None,
        max_tool_rounds: int = 10,
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.store = store
        self.max_tool_rounds = max_tool_rounds
        if not store:
            self.store = InMemoryChatStorage()
        self.mcp_client = None

        initial_tools = tools or []

        if mcp_config_path:
            try:
                self.mcp_client = MCPClient(mcp_config_path)
                self.mcp_client.start()
                initial_tools.extend(self.mcp_client.get_tools())
            except Exception as e:
                print(f"Failed to initialize MCP client: {e}")

        self._initialize_tools(initial_tools)

    def shutdown(self):
        """Cleanly shutdown resources like MCP client."""
        if self.mcp_client:
            self.mcp_client.stop()

    def _initialize_tools(self, tools):
        if tools is not None and len(tools) > 0:
            self.tools = [tool.tool_json() for tool in tools]
            self.tool_map = {
                t["function"]["name"]: tool
                for t, tool in zip(self.tools, tools)
            }
        else:
            self.tools = []
            self.tool_map = {}

    def _initialize_system_message(self, session_id: str) -> None:
        """Initialize system message if none exists."""
        sys_msg = {"role": "system", "content": self.system_prompt}
        self.store.save(sys_msg, session_id)

    def _ensure_system_message(self, session_id: str) -> None:
        """Ensure system message exists and is up-to-date."""
        messages = self.store.get_all(session_id)

        if not messages or messages[0]["content"] != self.system_prompt:
            sys_msg = {"role": "system", "content": self.system_prompt}
            if messages:
                # Replace existing system message
                self.store.set_message(0, sys_msg, session_id)
            else:
                # Create new session with system message
                self.store.save(sys_msg, session_id)

    def _process_user_input(
        self, message: str, image: bytes = None
    ) -> Dict[str, Any]:
        """Prepare user input for storage."""
        if not message and not image:
            raise ValueError("Either text or image must be provided")

        content_parts = []

        if message:
            content_parts.append({"type": "text", "text": message})

        if image:
            # Encode image to base64
            base64_str = base64.b64encode(image).decode("utf-8")
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_str}"
                    },
                }
            )

        return {"role": "user", "content": content_parts}

    def _run_api_call(
        self, args: Dict[str, Any], stream: bool
    ) -> Generator[Dict[str, Any], None, None]:
        """Perform a single chat-completion call.

        Yields stream passthrough events (``partial_content`` /
        ``reasoning_delta``) while running, and finally yields a single
        ``{"_final": {...}}`` event containing the accumulated ``content``,
        ``reasoning`` and structured ``tool_calls``.
        """
        if not stream:
            response = self.client.chat.completions.create(**args)
            message_obj = response.choices[0].message
            content = message_obj.content or ""
            reasoning = getattr(message_obj, "reasoning_content", "") or ""
            raw_tool_calls = getattr(message_obj, "tool_calls", []) or []
            tool_calls = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in raw_tool_calls
            ]
            yield {
                "_final": {
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                }
            }
            return

        # Streaming branch
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
        tool_call_indices: List[int] = []
        content = ""
        reasoning = ""

        for chunk in self.client.chat.completions.create(**args, stream=True):
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Reasoning content (separate field on many OpenAI-compatible servers)
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning += reasoning_delta
                yield {
                    "role": "assistant",
                    "reasoning_delta": reasoning_delta,
                }

            # Assistant content delta
            if getattr(delta, "content", None):
                content += delta.content
                yield {
                    "role": "assistant",
                    "content_delta": delta.content,
                }

            # Structured tool calls
            if getattr(delta, "tool_calls", None):
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    if index not in tool_calls_accumulator:
                        tool_calls_accumulator[index] = {
                            "id": tool_call.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                        tool_call_indices.append(index)

                    tc = tool_calls_accumulator[index]
                    if tool_call.id:
                        tc["id"] = tool_call.id
                    if tool_call.function:
                        if tool_call.function.name:
                            tc["function"]["name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            tc["function"]["arguments"] += (
                                tool_call.function.arguments
                            )

        tool_calls_list = [
            tool_calls_accumulator[idx]
            for idx in tool_call_indices
            if tool_calls_accumulator[idx]["function"]["name"]
        ]

        yield {
            "_final": {
                "content": content,
                "reasoning": reasoning,
                "tool_calls": tool_calls_list,
            }
        }

    def _execute_single_tool(
        self, call: Dict[str, Any], session_id: str
    ) -> None:
        """Execute one tool call and persist the tool response message."""
        function_name = call["function"]["name"]
        arguments_str = call["function"].get("arguments") or "{}"
        tool_call_id = call.get("id", "")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        try:
            if function_name not in self.tool_map:
                raise KeyError(f"Unknown tool: {function_name}")
            result = self.tool_map[function_name].execute(**arguments)
            tool_response = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": (
                    json.dumps(result)
                    if not isinstance(result, str)
                    else result
                ),
            }
            self.store.save(tool_response, session_id)
        except Exception as e:
            error_response = {
                "error": f"Tool {function_name} failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            tool_response = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(error_response),
            }
            self.store.save(tool_response, session_id)

    @streamable_response
    def generate(
        self,
        message: str = "",
        image: bytes = None,
        session_id: str = "default",
        stream: bool = True,
        params: Dict[str, Any] = None,
        tools: List[Callable] = None,
        response_format: BaseModel = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Core generation with a full agentic tool-calling loop.

        The model is called repeatedly: on every round it has access to the
        configured tools. As long as it emits tool calls (either structured or
        recovered from plain text via :mod:`fastllm.tool_healing`), those tools
        are executed and their results fed back for another round. The loop ends
        when the model returns a plain answer or ``max_tool_rounds`` is reached.
        """
        if tools:
            self._initialize_tools(tools)
        if not isinstance(message, str):
            raise Exception(
                f"Wrong type: message is not str, it is {type(message)}"
            )

        self._ensure_system_message(session_id)
        msg_content = self._process_user_input(message, image)
        self.store.save(msg_content, session_id)

        base_params = params or {}
        response_format_arg = None
        if response_format:
            response_format_arg = {
                "type": "json_schema",
                "json_schema": {
                    "schema": pydantic_to_openai_schema(response_format),
                },
            }

        def build_args(with_tools: bool) -> Dict[str, Any]:
            args: Dict[str, Any] = {
                "messages": self.store.get_all(session_id),
                "model": self.model,
            }
            if with_tools and self.tools:
                args["tools"] = self.tools
            if response_format_arg:
                args["response_format"] = response_format_arg
            args.update(base_params)
            return args

        try:
            for _ in range(max(1, self.max_tool_rounds)):
                # --- One API call (with tools available) ---
                final = None
                for event in self._run_api_call(build_args(True), stream):
                    if "_final" in event:
                        final = event["_final"]
                    elif "content_delta" in event:
                        yield {
                            "role": "assistant",
                            "partial_content": event["content_delta"],
                        }
                    elif "reasoning_delta" in event:
                        yield event

                content = final["content"] or ""
                reasoning = final["reasoning"] or ""
                tool_calls = final["tool_calls"]

                # --- Heal text-encoded tool calls when none were structured ---
                healed = False
                if not tool_calls:
                    recovered = heal_tool_calls(
                        content, reasoning, self.tool_map
                    )
                    if recovered:
                        tool_calls = recovered
                        content = strip_tool_call_markup(content)
                        healed = True

                # --- No tool calls => final answer ---
                if not tool_calls:
                    final_msg = {"role": "assistant", "content": content}
                    if reasoning:
                        final_msg["reasoning_content"] = reasoning
                    self.store.save(final_msg, session_id)
                    if not stream:
                        yield final_msg
                    return

                # --- Persist assistant message carrying the tool calls ---
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                }
                self.store.save(assistant_tool_msg, session_id)
                if stream:
                    yield {
                        "tool_call": True,
                        "tool_calls": tool_calls,
                        "healed": healed,
                    }

                # --- Execute every tool call, then loop for another round ---
                for call in tool_calls:
                    self._execute_single_tool(call, session_id)

            # --- Loop budget exhausted: force a final answer without tools ---
            final = None
            for event in self._run_api_call(build_args(False), stream):
                if "_final" in event:
                    final = event["_final"]
                elif "content_delta" in event:
                    yield {
                        "role": "assistant",
                        "partial_content": event["content_delta"],
                    }
                elif "reasoning_delta" in event:
                    yield event

            final_content = final["content"] or ""
            final_msg = {"role": "assistant", "content": final_content}
            self.store.save(final_msg, session_id)
            if not stream:
                yield final_msg
            return

        except Exception as e:
            print(traceback.format_exc())
            raise EmptyPayload(f"API error: {e}")
