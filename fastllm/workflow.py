from .agent import Agent


class Node:
    """
    A class representing a node in a workflow with LLMs.

    Attributes:
        ctx (dict): ctx for additional information. Default is None.
        messages (list): A list containing the messages to be sent to the LLM. Default is an empty list.
        agent (Agent): The agent object used for generating responses from the LLM.
        neasted_tool (bool): Indicates whether a nested tool is being used. Default is None.
        message_to_send (str or dict): The message to be sent to the LLM. Can be either a string or a dictionary.
        after_generation (callable): A callback function to be called after generating messages.
        before_generation (callable): A callback function to be called before generating messages.
    """  # noqa: E501

    def __init__(
        self,
        ctx: dict = None,
        agent: Agent = None,
        neasted_tool: bool = None,
        after_generation: callable = None,
        before_generation: callable = None,
        temperature: float = 0.7,
    ):
        self.ctx = ctx if ctx is not None else {}
        self.next_nodes = []
        self.agent = agent
        self.neasted_tool = neasted_tool
        self.after_generation = after_generation
        self.before_generation = before_generation
        self.temperature = temperature

    def run(self, message_to_send: str, session_id: str = "default"):
        """
        Executes the node's workflow, including sending messages to the LLM and handling responses.

        This method will send the current message to the LLM using the agent, optionally call a callback before generating,
        and then handle the responses by transitioning to the next nodes with the generated responses.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        if self.before_generation is not None:
            self.before_generation(self, session_id)
        if self.agent:
            for _ in self.agent.generate(
                message_to_send, session_id=session_id,
                neasted_tool=self.neasted_tool,
                params={"temperature": self.temperature}
            ):
                pass
            if self.after_generation is not None:
                self.after_generation(self, session_id)
            for next_node in self.next_nodes:
                next_node.ctx[session_id] = self.ctx.get(session_id, {})
                if type(next_node).__name__ == "BooleanNode":
                    next_node.run(session_id)
                else:
                    next_node.run(message_to_send, session_id)

    def connect_to(self, node):
        """Connect this node to another node."""
        self.next_nodes.append(node)


class BooleanNode:
    """
    A class representing a conditional node in a workflow with LLMs.

    Attributes:
        ctx (dict): ctx for additional information. Default is an empty dictionary.
        messages (list): List of messages to be sent to the LLM. Default is an empty list.
        agent (Agent): The agent object used for generating responses from the LLM.
        neasted_tool (bool): Indicates whether a nested tool is being used. Default is None.
        message_to_send (Union[str, dict]): The message to be sent to the LLM. Can be either a string or a dictionary.
        after_generation (callable): A callback function to be called after generating messages.
        before_generation (callable): A callback function to be called before generating messages.
        condition (callable): A callable that determines which path to take based on the result of its execution.
        true_nodes (List["Node"]): List of nodes to transition to if the condition is True. Default is an empty list.
        false_nodes (List["Node"]): List of nodes to transition to if the condition is False. Default is an empty list.
    """  # noqa: E501
    def __init__(
        self,
        ctx: dict = None,
        condition: callable = None,
        message_case_true: str = "",
        message_case_false: str = ""
    ):
        self.ctx = ctx if ctx is not None else {}
        self.condition = condition
        self.true_nodes = []
        self.false_nodes = []
        self.message_case_true = message_case_true
        self.message_case_false = message_case_false

    def run(self, session_id: str = "default"):
        """
        Executes the node's workflow, including sending messages to the LLM and handling responses based on a conditional check.

        This method will execute the condition before proceeding with either the true or false nodes, depending on whether the condition is True or False.

        Returns:
            The final response from the LLM after all transitions are completed.
        """  # noqa: E501
        cond = self.condition(self, session_id)
        nodes = self.true_nodes if cond else self.false_nodes
        for next_node in nodes:
            next_node.ctx[session_id] = self.ctx.get(session_id, {})
            if type(next_node).__name__ == "BooleanNode":
                next_node.run(session_id)
            else:
                next_node.run(self.message_case_true if cond
                              else self.message_case_false,
                              session_id)

    def connect_to_false(self, node):
        """Connect this node to another node if the condition is False."""
        self.false_nodes.append(node)

    def connect_to_true(self, node):
        """Connect this node to another node if the condition is True."""
        self.true_nodes.append(node)
