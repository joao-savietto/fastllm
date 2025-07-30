"""
This module provides an Agent class for interacting with the OpenAI API to generate responses based on chat history.

Classes:
    Agent: A class to interact with the OpenAI API for generating AI responses.
"""

from typing import Generator, Dict, Any, List
import base64
import json

import openai
from fastllm.store import ChatStorageInterface, InMemoryChatStorage
from fastllm.decorators import streamable_response
from fastllm.exceptions import EmptyPayload


class Agent:
    def __init__(
        self,
        model: str = "got4o",
        base_url: str = "https://api.openai.com/v1/",
        api_key: str = "",
        tools: List[Any] = None,
        system_prompt: str = "",
        store: ChatStorageInterface = InMemoryChatStorage(),
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.store = store

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

    def _ensure_system_message(self, session_id: str):
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
        """Stream the first API call and yield tool calls and assistant content in real time."""

        partial_content = ""
        collected_tool_calls = []
        current_tool_call = {}

        for chunk in self.client.chat.completions.create(
            **args_with_tools, stream=True
        ):
            if not hasattr(chunk.choices[0], "delta"):
                continue

            delta = chunk.choices[0].delta
            finish_reason = getattr(chunk.choices[0], "finish_reason", None)

            # Stream assistant content
            if hasattr(delta, "content") and delta.content:
                partial_content += delta.content
                yield {
                    "role": "assistant",
                    "partial_content": partial_content,
                }

            # Handle tool calls
            elif hasattr(delta, "tool_call"):
                tc = getattr(delta, "tool_call", None)
                if tc is not None:
                    function_name = next(
                        t["function"]["name"]
                        for t in self.tools
                        if t["function"]["name"]
                        == getattr(tc.function, "name", "")
                    )

                    current_tool_call.update(
                        {
                            "tool_call_id": getattr(tc, "id", ""),
                            "function_name": function_name,
                            "arguments": json.loads(
                                getattr(tc.function, "arguments", "{}")
                            ),
                        }
                    )

            # Finalize at stream end
            if finish_reason is not None:
                if hasattr(delta, "content") and delta.content:
                    partial_content += delta.content

                if current_tool_call:
                    collected_tool_calls.append(current_tool_call)
                    yield {
                        "tool_call": True,
                        "tool_calls": collected_tool_calls,
                        "partial_content": partial_content,
                    }

    @streamable_response
    def generate(
        self,
        message: str = "",
        image: bytes = None,
        session_id: str = "default",
        stream: bool = True,
        params: Dict[str, Any] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Core generation with tool call sequencing and streaming support.

        Args:
            message: User's text input.
            image: Optional image data as bytes.
            session_id: Identifier for the chat session.
            stream: Whether to stream responses.
            params: Additional parameters to pass to the OpenAI API (e.g., temperature, top_p).
        """
        # 1. Ensure system prompt is up-to-date
        self._ensure_system_message(session_id)
        msg_content = self._process_user_input(message, image)
        self.store.save(msg_content, session_id)

        # Prepare base arguments for the first API call
        args_with_tools: Dict[str, Any] = {
            "messages": self.store.get_all(session_id),
            "model": self.model,
            "tools": self.tools if self.tools else None,
        }

        # Merge with any extra params provided
        if params:
            args_with_tools.update(params)

        try:
            # When there are no tools, we only need one API call
            if not self.tools:
                if stream:
                    previous_content = ""
                    partial_content = ""
                    for chunk in self.client.chat.completions.create(
                        **args_with_tools, stream=True
                    ):
                        delta_content = getattr(
                            chunk.choices[0].delta, "content", ""
                        )
                        if delta_content:
                            partial_content += delta_content
                            new_chunk = partial_content[
                                len(previous_content) :
                            ]
                            yield {
                                "role": "assistant",
                                "partial_content": new_chunk,
                            }
                            previous_content = partial_content
                    # Save the final message
                    final_msg = {
                        "role": "assistant",
                        "content": partial_content,
                    }
                    self.store.save(final_msg, session_id)
                    return  # We're done
                else:
                    first_response = self.client.chat.completions.create(
                        **args_with_tools
                    )
                    message_obj = first_response.choices[0].message
                    final_msg = {
                        "role": "assistant",
                        **message_obj.model_dump(),
                    }
                    self.store.save(final_msg, session_id)
                    yield final_msg
                    return  # We're done

            # If we have tools, proceed with the two API call sequence
            collected_tool_calls = []
            partial_content = ""

            # Streamed first API call with tools
            if stream:
                previous_content = ""
                for chunk in self._stream_first_api_call(
                    args_with_tools, session_id
                ):
                    if "partial_content" in chunk:
                        partial_content += chunk["partial_content"]
                        new_chunk = partial_content[len(previous_content) :]
                        yield {
                            "role": "assistant",
                            "partial_content": new_chunk,
                        }
                        previous_content = partial_content
                    else:
                        # For tool calls, yield the chunk as is
                        yield chunk
            else:
                first_response = self.client.chat.completions.create(
                    **args_with_tools
                )
                message = first_response.choices[0].message
                collected_tool_calls = (
                    getattr(message, "tool_calls", []) or []
                )

            # 2. Process tool calls from both stream and non-stream paths
            for call in collected_tool_calls:
                function_name = call["function_name"]

                try:
                    result = self.tool_map[function_name].execute(
                        **call["arguments"]
                    )

                    tool_response = {
                        "tool_call_id": call.get("tool_call_id"),
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    }
                    self.store.save(tool_response, session_id)
                except Exception as e:
                    error_response = {
                        "error": f"Tool {function_name} failed: {e}"
                    }
                    tool_response["content"] = error_response

            # 3. Second API call (without tools), streamed
            args_without_tools: Dict[str, Any] = {
                "messages": self.store.get_all(session_id),
                "model": self.model,
            }

            # Merge extra params into second call as well
            if params:
                args_without_tools.update(params)

            if stream:
                previous_content = ""
                for chunk in self.client.chat.completions.create(
                    **args_without_tools, stream=True
                ):
                    delta_content = getattr(
                        chunk.choices[0].delta, "content", ""
                    )
                    if delta_content:
                        partial_content += delta_content
                        new_chunk = partial_content[len(previous_content) :]
                        yield {
                            "role": "assistant",
                            "partial_content": new_chunk,
                        }
                        previous_content = partial_content
            else:
                second_response = self.client.chat.completions.create(
                    **args_without_tools
                )
                final_msg = {
                    "role": "assistant",
                    **second_response.choices[0].message.model_dump(),
                }
                yield final_msg

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            raise EmptyPayload(f"API error: {e}")
