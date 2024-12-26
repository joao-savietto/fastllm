from typing import Union, List

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
        if self.message_to_send:
            if self.before_generate is not None:
                self.before_generate(self)
            if isinstance(self.message_to_send, str):
                self.messages.append({
                    "role": "user",
                    "content": self.message_to_send})
            elif isinstance(self.message_to_send, dict):
                self.messages.append(self.message_to_send)
            responses = []
            for response in self.agent.generate(
                self.messages, neasted_tool=self.neasted_tool
            ):
                responses.append(response)
            if self.on_generate is not None:
                self.on_generate(self)
        for next_node in self.next_nodes:
            next_node.messages = responses
            next_node.run()

    def connect_to(self, node):
        """Connect this node to another node."""
        self.next_nodes.append(node)


class BooleanNode:
    """
    A class representing a conditional node in a workflow with LLMs.

    Attributes:
        store (dict): Store for additional information. Default is an empty dictionary.
        messages (list): List of messages to be sent to the LLM. Default is an empty list.
        agent (Agent): The agent object used for generating responses from the LLM.
        neasted_tool (bool): Indicates whether a nested tool is being used. Default is None.
        message_to_send (Union[str, dict]): The message to be sent to the LLM. Can be either a string or a dictionary.
        on_generate (callable): A callback function to be called after generating messages.
        before_generate (callable): A callback function to be called before generating messages.
        condition (callable): A callable that determines which path to take based on the result of its execution.
        true_nodes (List["Node"]): List of nodes to transition to if the condition is True. Default is an empty list.
        false_nodes (List["Node"]): List of nodes to transition to if the condition is False. Default is an empty list.
    """  # noqa: E501
    def __init__(
        self,
        store: dict = None,
        messages: dict = None,
        agent: Agent = None,
        neasted_tool: bool = None,
        message_to_send: Union[str, dict] = None,
        on_generate: callable = None,
        before_generate: callable = None,
        condition: callable = None,
        true_nodes: List["Node"] = [],
        false_nodes: List["Node"] = [],
    ):
        self.store = store if store is not None else {}
        self.messages = messages if messages is not None else []
        self.agent = agent
        self.neasted_tool = neasted_tool
        self.message_to_send = message_to_send
        self.on_generate = on_generate
        self.before_generate = before_generate
        self.condition = condition
        self.true_nodes = true_nodes
        self.false_nodes = false_nodes

    def run(self):
        """
        Executes the node's workflow, including sending messages to the LLM and handling responses based on a conditional check.

        This method will execute the condition before proceeding with either the true or false nodes, depending on whether the condition is True or False.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        self.before_generate(self)
        nodes = self.true_nodes if self.condition(self) else self.false_nodes
        for next_node in nodes:
            next_node.messages = self.messages
            next_node.store = self.store
            next_node.run()
        self.on_generate(self)

    def connect_to_false(self, node):
        """Connect this node to another node if the condition is False."""
        self.false_nodes.append(node)

    def connect_to_true(self, node):
        """Connect this node to another node if the condition is True."""
        self.true_nodes.append(node)
