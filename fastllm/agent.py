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
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.store = store
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

    def _stream_first_api_call(
        self, args_with_tools: Dict[str, Any], session_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream the first API call and yield content deltas and tool calls."""

        # Dictionary to accumulate tool calls by index
        tool_calls_accumulator = {}
        # List to track the order of tool calls
        tool_call_indices = []

        for chunk in self.client.chat.completions.create(
            **args_with_tools, stream=True
        ):
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = getattr(chunk.choices[0], "finish_reason", None)

            # Stream assistant content delta
            if hasattr(delta, "content") and delta.content:
                yield {
                    "role": "assistant",
                    "content_delta": delta.content,
                }

            # Handle tool calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    if index not in tool_calls_accumulator:
                        # Initialize new tool call
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
                            tc["function"]["arguments"] += tool_call.function.arguments

            # Finalize tool calls at stream end
            if finish_reason is not None:
                tool_calls_list = []
                for idx in tool_call_indices:
                    tc = tool_calls_accumulator[idx]
                    if tc["function"]["name"]:
                        tool_calls_list.append(tc)

                if tool_calls_list:
                    yield {
                        "tool_calls": tool_calls_list,
                    }

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
        """Core generation with tool call sequencing and streaming support."""
        if tools:
            self._initialize_tools(tools)
        if not isinstance(message, str):
            raise Exception(f"Wrong type: message is not str, it is {type(message)}")
            
        self._ensure_system_message(session_id)
        msg_content = self._process_user_input(message, image)
        self.store.save(msg_content, session_id)

        # Prepare base arguments for the first API call
        args_with_tools: Dict[str, Any] = {
            "messages": self.store.get_all(session_id),
            "model": self.model,
            "tools": self.tools if self.tools else None,
        }
        if response_format:
            args_with_tools["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": pydantic_to_openai_schema(response_format),
                },
            }

        if params:
            args_with_tools.update(params)

        try:
            collected_tool_calls = []
            first_call_content = ""

            # 1. First API call
            if stream:
                for chunk in self._stream_first_api_call(args_with_tools, session_id):
                    if "content_delta" in chunk:
                        delta = chunk["content_delta"]
                        first_call_content += delta
                        yield {
                            "role": "assistant",
                            "partial_content": delta,
                        }
                    if "tool_calls" in chunk:
                        collected_tool_calls = chunk["tool_calls"]
                        # We yield the tool call event to the caller
                        yield {
                            "tool_call": True,
                            "tool_calls": collected_tool_calls,
                        }
            else:
                first_response = self.client.chat.completions.create(**args_with_tools)
                message_obj = first_response.choices[0].message
                first_call_content = message_obj.content or ""
                raw_tool_calls = getattr(message_obj, "tool_calls", []) or []
                # Convert to dicts for consistency
                collected_tool_calls = [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc 
                    for tc in raw_tool_calls
                ]
                
                if not collected_tool_calls:
                    final_msg = {"role": "assistant", **message_obj.model_dump()}
                    self.store.save(final_msg, session_id)
                    yield final_msg
                    return

            # If we had tool calls, we MUST save the assistant message with tool calls first
            if collected_tool_calls:
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": first_call_content if first_call_content else None,
                    "tool_calls": collected_tool_calls,
                }
                self.store.save(assistant_tool_msg, session_id)

                # 2. Process tool calls
                for call in collected_tool_calls:
                    function_name = call["function"]["name"]
                    arguments_str = call["function"]["arguments"] or "{}"
                    tool_call_id = call.get("id", "")

                    try:
                        arguments = json.loads(arguments_str) if arguments_str else {}
                    except json.JSONDecodeError:
                        arguments = {}

                    try:
                        result = self.tool_map[function_name].execute(**arguments)
                        tool_response = {
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result,
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

                # 3. Second API call for final response
                args_without_tools = {
                    "messages": self.store.get_all(session_id),
                    "model": self.model,
                }
                if params:
                    args_without_tools.update(params)

                second_call_content = ""
                if stream:
                    for chunk in self.client.chat.completions.create(**args_without_tools, stream=True):
                        if not chunk.choices: continue
                        delta_content = getattr(chunk.choices[0].delta, "content", "")
                        if delta_content:
                            second_call_content += delta_content
                            yield {
                                "role": "assistant",
                                "partial_content": delta_content,
                            }
                    # Save final response
                    self.store.save({"role": "assistant", "content": second_call_content}, session_id)
                else:
                    second_response = self.client.chat.completions.create(**args_without_tools)
                    final_msg = {"role": "assistant", **second_response.choices[0].message.model_dump()}
                    self.store.save(final_msg, session_id)
                    yield final_msg
            else:
                # No tool calls, if we were streaming we already yielded. 
                # Just need to save if we haven't yet (we haven't in streaming case).
                if stream:
                    self.store.save({"role": "assistant", "content": first_call_content}, session_id)

        except Exception as e:
            print(traceback.format_exc())
            raise EmptyPayload(f"API error: {e}")
