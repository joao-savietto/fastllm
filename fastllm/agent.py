"""
This module provides an Agent class for interacting with the OpenAI API to
generate responses based on chat history.

Classes:
    Agent: A class to interact with the OpenAI API for
    generating AI responses.
"""

from typing import List, Optional, Dict, Any
import json
import openai


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

    def generate(
        self,
        chat_history: List[Dict[str, Any]],
        session_id: str = None,
        neasted_tool: bool = False,
        params: Dict[str, Any] = {},
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a response from the OpenAI API based
         on the chat history and parameters

        """
        args = {"messages": chat_history, "model": self.model, **params}
        if self.tools is not None:
            args["tool_choice"] = "auto"
            args["tools"] = self.tools

        response_message = self.client.chat.completions.create(**args)
        response_message = response_message.choices[0].message
        tool_calls = response_message.tool_calls
        chat_history.append(response_message)
        if tool_calls:
            while tool_calls is not None:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.tool_map[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    function_response = function_to_call.execute(**function_args)
                    chat_history.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                args["messages"] = chat_history
                if neasted_tool is False:
                    if "tool_choice" in params:
                        del args["tool_choice"]
                    if "tools" in params:
                        del args["tools"]
                second_response = self.client.chat.completions.create(**args)
                response_message = second_response.choices[0].message
                if neasted_tool is True:
                    tool_calls = response_message.tool_calls
                else:
                    tool_calls = None
                chat_history.append(response_message)
                if response_message.role == "assistant" and tool_calls is None:
                    yield json.loads(response_message.json())
        else:
            yield json.loads(response_message.json())
