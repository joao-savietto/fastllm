from typing import Union

from .agent import Agent


class Node:
    """
    A class representing a node in a workflow with LLMs.

    Attributes:
        store (dict): Store for additional information. Default is None.
        next_nodes (list of Node): List of nodes that this node can transition to. Default is an empty list.
        messages (list): A list containing the messages to be sent to the LLM. Default is an empty list.
        agent (Agent): The agent object used for generating responses from the LLM.
        neasted_tool (bool): Indicates whether a nested tool is being used. Default is None.
        message_to_send (str or dict): The message to be sent to the LLM. Can be either a string or a dictionary.
        on_generate (callable): A callback function to be called after generating messages.
        before_generate (callable): A callback function to be called before generating messages.
    """  # noqa: E501
    def __init__(
        self,
        store: dict = None,
        next_nodes: "Node" = None,
        messages: dict = None,
        agent: Agent = None,
        neasted_tool: bool = None,
        message_to_send: Union[str, dict] = None,
        on_generate: callable = None,
        before_generate: callable = None,
    ):
        self.store = store if store is not None else {}
        self.next_nodes = next_nodes if next_nodes is not None else []
        self.messages = messages if messages is not None else []
        self.agent = agent
        self.neasted_tool = neasted_tool
        self.message_to_send = message_to_send
        self.on_generate = on_generate
        self.before_generate = before_generate

    def run(self):
        """
        Executes the node's workflow, including sending messages to the LLM and handling responses.

        This method will send the current message to the LLM using the agent, optionally call a callback before generating,
        and then handle the responses by transitioning to the next nodes with the generated responses.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        if isinstance(self.message_to_send, str):
            self.messages.append({
                "role": "user", "content": self.message_to_send})
        elif isinstance(self.message_to_send, dict):
            self.messages.append(self.message_to_send)
        if self.before_generate is not None:
            self.before_generate(self)
        responses = []
        for response in self.agent.generate(
            self.messages, neasted_tool=self.neasted_tool
        ):
            responses.append(response)
        if self.on_generate is not None:
            self.on_generate(self, response)
        for next_node in self.next_nodes:
            next_node.messages = responses
            next_node.run()
        return responses[-1]  # Return the last generated response
