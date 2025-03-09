"""
This module provides an Agent class for interacting with the OpenAI API to
generate responses based on chat history.

Classes:
    Agent: A class to interact with the OpenAI API for
    generating AI responses.
"""

import json
from typing import Any, Dict, Optional
import base64

import openai

from fastllm.store import ChatStorageInterface, InMemoryChatStorage
from fastllm.decorators import retry


class EmptyPayload(Exception):
    pass


class Agent:
    """
    Agent class for interacting with the OpenAI API.
    """

    def __init__(
        self,
        model: str = "got4o",
        base_url: str = "https://api.openai.com/v1/",
        api_key: str = "",
        tools: list = None,
        system_prompt: str = "",
        store: ChatStorageInterface = None,
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.tools = None
        self.tool_map = {}
        if tools is not None:
            self.tools = []
            for tool in tools:
                tool_json = tool.tool_json()
                self.tool_map[tool_json["function"]["name"]] = tool
                self.tools.append(tool_json)
        self.system_prompt = system_prompt
        self.store = store
        if store is None:
            self.store = InMemoryChatStorage()

    @retry(max_attempts=5, delay=2)
    def generate(
        self,
        message: str,
        image: bytes = None,
        session_id: str = "default",
        neasted_tool: bool = False,
        params: Dict[str, Any] = {},
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a response from the OpenAI API based
         on the chat history and parameters

        """
        if len(self.store.get_all(session_id)) == 0:
            sys = {"role": "system", "content": self.system_prompt}
            self.store.save(sys)
        else:
            sys = self.store.get_message(0, session_id)
            if sys["content"] != self.system_prompt:
                self.store.set_message(
                    index=0,
                    message={"role": "system", "content": self.system_prompt},
                    session_id=session_id
                )
        if image:
            b64: str = base64.b64encode(image).decode("utf-8")
            item = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": message},
                ],
            }
            self.store.save(message=item, session_id=session_id)
        else:
            self.store.save(
                message={"role": "user", "content": message},
                session_id=session_id)

        args = {
            "messages": self.store.get_all(session_id),
            "model": self.model,
            **params,
        }
        if self.tools is not None:
            args["tool_choice"] = "auto"
            args["tools"] = self.tools

        response_message = self.client.chat.completions.create(**args)
        response_message = response_message.choices[0].message
        tool_calls = response_message.tool_calls
        self.store.save(response_message, session_id)
        if tool_calls:
            while tool_calls is not None:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.tool_map[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    function_response = function_to_call.execute(
                        **function_args)
                    self.store.save(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        },
                        session_id,
                    )

                args["messages"] = self.store.get_all(session_id)
                if neasted_tool is False:
                    if "tool_choice" in params:
                        del args["tool_choice"]
                    if "tools" in params:
                        del args["tools"]
                second_response = self.client.chat.completions.create(**args)
                if not second_response.choices:
                    raise EmptyPayload(second_response.error)
                response_message = second_response.choices[0].message
                if neasted_tool is True:
                    tool_calls = response_message.tool_calls
                else:
                    tool_calls = None
                self.store.save(response_message, session_id)
                if response_message.role == "assistant" and tool_calls is None:
                    yield json.loads(response_message.json())
        else:
            yield json.loads(response_message.json())
