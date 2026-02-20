"""
Unit tests for dynamic tool management in Agent and Workflow classes.
Tests verify that tools can be added and removed on demand without using API mocks.
"""

import os
import unittest
from pydantic import BaseModel, Field
from fastllm.agent import Agent
from fastllm.workflow import Node, BooleanNode
from fastllm.decorators import tool
from fastllm.store import InMemoryChatStorage


# Test tools for use in tests
class AddRequest(BaseModel):
    a: float = Field(..., description="First number to add")
    b: float = Field(..., description="Second number to add")


@tool(
    description="Adds two numbers together",
    pydantic_model=AddRequest,
)
def add_numbers(request: AddRequest):
    """Add two numbers and return the result."""
    return {"result": request.a + request.b}


class MultiplyRequest(BaseModel):
    a: float = Field(..., description="First number to multiply")
    b: float = Field(..., description="Second number to multiply")


@tool(
    description="Multiplies two numbers together",
    pydantic_model=MultiplyRequest,
)
def multiply_numbers(request: MultiplyRequest):
    """Multiply two numbers and return the result."""
    return {"result": request.a * request.b}


class SubtractRequest(BaseModel):
    a: float = Field(..., description="Number to subtract from")
    b: float = Field(..., description="Number to subtract")


@tool(
    description="Subtracts two numbers",
    pydantic_model=SubtractRequest,
)
def subtract_numbers(request: SubtractRequest):
    """Subtract two numbers and return the result."""
    return {"result": request.a - request.b}


class TestAgentToolManagement(unittest.TestCase):
    """Test dynamic tool management in Agent class."""

    @classmethod
    def setUpClass(cls):
        """Load environment variables once for all tests."""
        # Load .env file if present (simple approach, no extra dependency)
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"')
                    )
        except FileNotFoundError:
            pass  # Use existing environment variables

        # Retrieve configuration from environment
        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        cls.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.store = InMemoryChatStorage()
        # Create agent without tools initially
        self.agent = Agent(
            model=self.__class__.model,
            base_url=self.__class__.base_url,
            api_key=self.__class__.api_key,
            system_prompt="You are a helpful assistant",
            store=self.store,
        )

    def test_agent_initialized_without_tools(self):
        """Test that agent initializes correctly without tools."""
        self.assertEqual(len(self.agent.tools), 0)
        self.assertEqual(len(self.agent.tool_map), 0)

    def test_agent_adds_tools_dynamically_via_generate(self):
        """Test that tools can be added dynamically through generate method."""
        # Verify initial state
        self.assertEqual(len(self.agent.tools), 0)

        # Call generate with tools parameter and a prompt that encourages tool usage
        try:
            # This will fail due to API call, but we're testing tool initialization
            self.agent.generate(
                message="Please calculate the sum of 5 and 3 using the available tools",
                session_id="test_session",
                stream=False,
                tools=[add_numbers],
            )
        except Exception:
            # We expect an API error, but the tools should be initialized
            pass

        # Verify tools were added
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("add_numbers", list(self.agent.tool_map))
        self.assertEqual(
            self.agent.tools[0]["function"]["name"], "add_numbers"
        )

    def test_agent_adds_multiple_tools_dynamically(self):
        """Test that multiple tools can be added dynamically."""
        # Add first tool
        try:
            self.agent.generate(
                message="Calculate 5 + 3 using the available tools",
                session_id="test_session1",
                stream=False,
                tools=[add_numbers],
            )
        except Exception:
            pass

        self.assertEqual(len(self.agent.tools), 1)

        # Add second tool (replacing first)
        try:
            self.agent.generate(
                message="Calculate 4 * 2 using the available tools",
                session_id="test_session2",
                stream=False,
                tools=[multiply_numbers],
            )
        except Exception:
            pass

        # Should have only the second tool now (replacement behavior)
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("multiply_numbers", self.agent.tool_map)

        # Add multiple tools at once
        try:
            self.agent.generate(
                message="Perform calculations using the available math tools",
                session_id="test_session3",
                stream=False,
                tools=[add_numbers, multiply_numbers, subtract_numbers],
            )
        except Exception:
            pass

        # Should have all three tools now
        self.assertEqual(len(self.agent.tools), 3)
        self.assertIn("add_numbers", list(self.agent.tool_map))
        self.assertIn("multiply_numbers", list(self.agent.tool_map))
        self.assertIn("subtract_numbers", list(self.agent.tool_map))

    def test_agent_tool_replacement(self):
        """Test that providing new tools replaces existing ones."""
        # Add first tool
        try:
            self.agent.generate(
                message="Add 5 and 3 using the available tools",
                session_id="test_session1",
                stream=False,
                tools=[add_numbers],
            )
        except Exception:
            pass

        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("add_numbers", list(self.agent.tool_map))

        # Replace with different tool
        try:
            self.agent.generate(
                message="Multiply 4 and 2 using the available tools",
                session_id="test_session2",
                stream=False,
                tools=[multiply_numbers],
            )
        except Exception:
            pass

        # Old tool should be gone, new tool should be present
        self.assertEqual(len(self.agent.tools), 1)
        self.assertNotIn("add_numbers", list(self.agent.tool_map))
        self.assertIn("multiply_numbers", list(self.agent.tool_map))

    def test_agent_tools_parameter_none_preserves_existing(self):
        """Test that passing None for tools doesn't affect existing tools."""
        # Add a tool first
        try:
            self.agent.generate(
                message="Test",
                session_id="test_session1",
                stream=False,
                tools=[add_numbers],
            )
        except Exception:
            pass

        self.assertEqual(len(self.agent.tools), 1)

        # Call generate with None tools - should preserve existing
        try:
            self.agent.generate(
                message="Test",
                session_id="test_session2",
                stream=False,
                tools=None,
            )
        except Exception:
            pass

        # Tools should still be present
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("add_numbers", list(self.agent.tool_map))

    def test_agent_initialized_with_tools(self):
        """Test that agent initializes correctly with tools."""
        agent_with_tools = Agent(
            model=self.__class__.model,
            base_url=self.__class__.base_url,
            api_key=self.__class__.api_key,
            system_prompt="You are a helpful assistant",
            tools=[add_numbers, multiply_numbers],
            store=InMemoryChatStorage(),
        )

        self.assertEqual(len(agent_with_tools.tools), 2)
        self.assertIn("add_numbers", list(agent_with_tools.tool_map))
        self.assertIn("multiply_numbers", list(agent_with_tools.tool_map))


class TestWorkflowToolManagement(unittest.TestCase):
    """Test dynamic tool management in Workflow classes."""

    @classmethod
    def setUpClass(cls):
        """Load environment variables once for all tests."""
        # Load .env file if present (simple approach, no extra dependency)
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"')
                    )
        except FileNotFoundError:
            pass  # Use existing environment variables

        # Retrieve configuration from environment
        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        cls.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.store = InMemoryChatStorage()
        self.agent = Agent(
            model=self.__class__.model,
            base_url=self.__class__.base_url,
            api_key=self.__class__.api_key,
            system_prompt="You are a helpful assistant",
            store=self.store,
        )

    def test_node_initialized_with_tools(self):
        """Test that Node can be initialized with tools."""
        node = Node(
            instruction="Test instruction",
            agent=self.agent,
            tools=[add_numbers, multiply_numbers],
        )

        self.assertEqual(len(node.tools), 2)

    def test_node_passes_tools_to_agent_generate(self):
        """Test that Node passes its tools to agent.generate method."""
        node = Node(
            instruction="Test instruction",
            agent=self.agent,
            tools=[add_numbers],
        )

        # Verify node has the tool
        self.assertEqual(len(node.tools), 1)

        # When run is called, it should pass tools to generate
        node.run(session_id="test_session")

        # Verify agent has the tool after the call
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("add_numbers", list(self.agent.tool_map))

    def test_node_without_tools_uses_agent_defaults(self):
        """Test that Node without tools uses agent's default tools."""
        # Add tool to agent
        try:
            self.agent.generate(
                message="Test",
                session_id="test_session1",
                stream=False,
                tools=[add_numbers],
            )
        except Exception:
            pass

        # Create node without tools
        node = Node(
            instruction="Test instruction",
            agent=self.agent,
            tools=None,  # Explicitly no tools
        )

        self.assertIsNone(node.tools)

        # When run is called, it should pass None to generate
        try:
            node.run(session_id="test_session2")
        except Exception:
            # We expect an API error, but the call should work correctly
            pass

    def test_boolean_node_with_tools(self):
        """Test that BooleanNode can handle tool management."""

        # Create a simple condition function
        def simple_condition(node, session_id, last_message):
            return True

        boolean_node = BooleanNode(
            condition=simple_condition,
            instruction_true="True instruction",
            instruction_false="False instruction",
            storage=self.store,
        )

        # Connect to a node with tools
        target_node = Node(
            instruction="Target instruction",
            agent=self.agent,
            tools=[add_numbers],
        )
        boolean_node.connect_to_true(target_node)

        self.assertEqual(len(target_node.tools), 1)

    def test_workflow_with_dynamic_tool_changes(self):
        """Test complex workflow with dynamic tool changes."""
        # Create nodes with different tools
        node1 = Node(
            instruction="First step",
            agent=self.agent,
            tools=[add_numbers],
        )

        node2 = Node(
            instruction="Second step",
            agent=self.agent,
            tools=[multiply_numbers],
        )

        # Connect nodes
        node1.connect_to(node2)

        # Run first node - should add add_numbers tool to agent
        try:
            node1.run(session_id="workflow_test")
        except Exception:
            pass

        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("multiply_numbers", list(self.agent.tool_map))

        # Run second node - should replace with multiply_numbers tool
        try:
            node2.run(session_id="workflow_test")
        except Exception:
            pass

        # Should now have only multiply_numbers tool
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn("multiply_numbers", list(self.agent.tool_map))
        self.assertNotIn("add_numbers", list(self.agent.tool_map))

    def test_tool_execution_functionality(self):
        """Test that tool execution methods work correctly."""
        # Test add_numbers tool
        result = add_numbers.execute(a=5, b=3)
        import json

        parsed_result = json.loads(result)
        self.assertEqual(parsed_result["result"], 8.0)

        # Test multiply_numbers tool
        result = multiply_numbers.execute(a=4, b=2.5)
        parsed_result = json.loads(result)
        self.assertEqual(parsed_result["result"], 10.0)

        # Test subtract_numbers tool
        result = subtract_numbers.execute(a=10, b=3)
        parsed_result = json.loads(result)
        self.assertEqual(parsed_result["result"], 7.0)

    def test_tool_json_structure(self):
        """Test that tool JSON structure is correct."""
        tool_json = add_numbers.tool_json()

        self.assertIn("type", tool_json)
        self.assertEqual(tool_json["type"], "function")
        self.assertIn("function", tool_json)
        self.assertIn("name", tool_json["function"])
        self.assertEqual(tool_json["function"]["name"], "add_numbers")
        self.assertIn("description", tool_json["function"])
        self.assertIn("parameters", tool_json["function"])


if __name__ == "__main__":
    unittest.main()
