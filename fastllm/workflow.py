"""
Workflow module for creating and managing LLM-based workflows.

This module provides classes for building workflow graphs with nodes that can execute
LLM operations and make conditional decisions based on the results of those operations.
"""

from typing import List, Callable
from .agent import Agent


class Node:
    """
    A class representing a node in a workflow with LLMs.

    Attributes:
        ctx (dict): Context dictionary for additional information. Default is None.
        instructions (str): The instructions to be sent to the LLM. Default is an empty None.
                           You can also pass it on the "run" method.
        agent (Agent): The agent object used for generating responses from the LLM.
        instruction (str or dict): The instruction to be sent to the LLM. Can be either a string or a dictionary.
        after_generation (callable): A callback function called after generating instructions.
            Expected parameters:
            - node: The current Node instance
            - session_id: String identifier for the session
            - response: Generated response content
        before_generation (callable): A callback function called before generating instructions.
            Expected parameters:
            - node: The current Node instance
            - session_id: String identifier for the session
    """  # noqa: E501

    def __init__(
        self,
        instruction: str = None,
        ctx: dict = None,
        agent: Agent = None,
        after_generation: callable = None,
        before_generation: callable = None,
        temperature: float = None,
        propagate_storage: bool = True,
        streaming: bool = False,
        tools: List[Callable] = None,
    ):
        self.type = "Node"
        self.instruction = instruction
        self.ctx = ctx if ctx is not None else {}
        self.next_nodes = []
        self.agent = agent
        self.after_generation = after_generation
        self.before_generation = before_generation
        self.temperature = temperature
        self.propagate_storage = propagate_storage
        self.streaming = streaming
        self.tools = tools

    def run(
        self,
        instruction: str = None,
        image: bytes = None,
        session_id: str = "default",
    ):
        """
        Executes the node's workflow, including sending instructions to the LLM and handling responses.

        This method will send the current instruction to the LLM using the agent, optionally call a callback before generating,
        and then handle the responses by transitioning to the next nodes with the generated responses.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        if self.before_generation is not None:
            self.before_generation(self, session_id)
        message = self.instruction or instruction
        if self.agent:
            generate_kwargs = {
                "message": message,
                "image": image,
                "session_id": session_id,
                "stream": self.streaming,
                "params": {"temperature": self.temperature} if self.temperature else {},
                "tools": self.tools or None
            }

            if self.streaming:
                for chunk in self.agent.generate(**generate_kwargs):
                    if self.after_generation:
                        self.after_generation(self, session_id, chunk)
            else:
                generated = self.agent.generate(**generate_kwargs)
                if self.after_generation:
                    self.after_generation(
                        self, session_id, generated["content"]
                    )

            for next_node in self.next_nodes:
                next_node.ctx[session_id] = self.ctx.get(session_id, {})
                if next_node.type == "BooleanNode":
                    if self.propagate_storage:
                        next_node.storage = self.agent.store
                        next_node.propagate_storage = True
                    next_node.run(session_id=session_id)

                else:
                    if self.propagate_storage:
                        next_node.agent.store = self.agent.store
                    next_node.run(session_id=session_id, instruction=message)

    def connect_to(self, node):
        """Connect this node to another node."""
        self.next_nodes.append(node)

    def get_history(self, session_id: str = "default"):
        """Returns the message history of the agent"""
        return self.agent.store.get_all(session_id)


class BooleanNode:
    """
    A class representing a conditional node in a workflow with LLMs.

    Attributes:
        ctx (dict): Context dictionary for additional information. Default is an empty dictionary.
        instructions (list): List of instructions to be sent to the LLM. Default is an empty list.
        agent (Agent): The agent object used for generating responses from the LLM.
        instruction (Union[str, dict]): The instruction to be sent to the LLM. Can be either a string or a dictionary.
        after_generation (callable): A callback function called after generating instructions.
            Expected parameters:
            - node: The current Node instance
            - session_id: String identifier for the session
            - response: Generated response content
        before_generation (callable): A callback function called before generating instructions.
            Expected parameters:
            - node: The current Node instance
            - session_id: String identifier for the session
        condition (callable): A callable that determines which path to take based on its execution.
            Expected parameters:
            - node: The current BooleanNode instance
            - session_id: String identifier for the session
            - last_message: The last generated message by the model. It comes as a dict with keys "content" and "role"
            Returns: Boolean indicating whether condition is True or False
        true_nodes (List["Node"]): List of nodes to transition to if the condition is True. Default is an empty list.
        false_nodes (List["Node"]): List of nodes to transition to if the condition is False. Default is an empty list.
    """  # noqa: E501

    def __init__(
        self,
        ctx: dict = None,
        condition: callable = None,
        instruction_true: str = "",
        instruction_false: str = "",
        storage: any = None,
        propagate_storage: bool = True,
    ):
        self.type = "BooleanNode"
        self.ctx = ctx if ctx is not None else {}
        self.condition = condition
        self.true_nodes = []
        self.false_nodes = []
        self.instruction_true = instruction_true
        self.instruction_false = instruction_false
        self.propagate_storage = propagate_storage
        self.storage = None

    def run(self, session_id: str = "default"):
        """
        Executes the node's workflow, including sending instructions to the LLM and handling responses based on a conditional check.

        This method will execute the condition before proceeding with either the true or false nodes, depending on whether the condition is True or False.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        history = self.storage.get_all(session_id)
        cond = self.condition(self, session_id, history[-1])
        nodes = self.true_nodes if cond else self.false_nodes
        for next_node in nodes:
            next_node.propagate_storage = self.propagate_storage
            next_node.ctx[session_id] = self.ctx.get(session_id, {})
            if type(next_node).__name__ == "BooleanNode":
                if self.propagate_storage:
                    next_node.storage = self.storage
                    next_node.run(session_id)
            else:
                if self.propagate_storage:
                    next_node.agent.store = self.storage
                next_node.run(
                    instruction=(
                        self.instruction_true
                        if cond
                        else self.instruction_false
                    ),
                    session_id=session_id,
                )

    def connect_to_false(self, node):
        """Connect this node to another node if the condition is False."""
        self.false_nodes.append(node)

    def connect_to_true(self, node):
        """Connect this node to another node if the condition is True."""
        self.true_nodes.append(node)

    def get_history(self, session_id: str = "default"):
        """Returns the message history of the agent"""
        return self.storage.get_all(session_id)
