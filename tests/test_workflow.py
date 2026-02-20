# test_workflow.py
"""
Integration tests for the workflow module using real OpenAI API calls.
These tests replace mocked API interactions with actual calls, pulling
configuration from environment variables (e.g., .env file).
"""

import os
import unittest
from fastllm.agent import Agent
from fastllm.store import InMemoryChatStorage
from fastllm.workflow import Node


class TestRealAPICalls(unittest.TestCase):
    """Integration test suite for real API usage."""

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
        # Create a real Agent instance with environment configuration
        self.agent = Agent(
            model=self.__class__.model,
            base_url=self.__class__.base_url,
            api_key=self.__class__.api_key,
            system_prompt="You are a helpful assistant",
        )
        # Use an in‑memory store for session persistence within the test
        self.store = InMemoryChatStorage()
        self.agent.store = self.store

    def test_generate_real_api_response(self):
        """Test that agent.generate makes a real API call and returns content."""
        # Prepare a minimal chat history
        self.store.save(
            {"role": "system", "content": "You are a helpful assistant"},
            session_id="test",
        )
        self.store.save(
            {"role": "user", "content": "Say hello in a friendly way."},
            session_id="test",
        )

        # Call the generate method (non‑streaming)
        response = self.agent.generate(
            message="Say hello in a friendly way.",
            session_id="test",
            stream=False,
        )
        # Verify that we got at least one response dictionary
        self.assertIsNotNone(response)

        # Check that the response contains expected fields
        self.assertIn("role", response)
        self.assertEqual(response["role"], "assistant")
        self.assertIn("content", response)
        self.assertTrue(len(response["content"]) > 0)

    def test_agent_initialization_with_env(self):
        """Verify that Agent initializes correctly using env variables."""
        self.assertIsNotNone(self.__class__.api_key)
        self.assertIsNotNone(self.__class__.base_url)
        self.assertIsNotNone(self.__class__.model)

        # Ensure the agent attributes are set properly
        self.assertEqual(self.agent.model, self.__class__.model)
        self.assertEqual(self.agent.base_url, self.__class__.base_url)
        self.assertEqual(self.agent.api_key, self.__class__.api_key)


class TestNodeFeatures(unittest.TestCase):
    """Test suite for Node features."""

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
        # Create a real Agent instance with environment configuration
        self.agent = Agent(
            model=self.__class__.model,
            base_url=self.__class__.base_url,
            api_key=self.__class__.api_key,
            system_prompt="You are a helpful assistant",
        )
        # Use an in‑memory store for session persistence within the test
        self.store = InMemoryChatStorage()
        self.agent.store = self.store

    def test_node_creation_and_properties(self):
        """Test that Node objects can be created with correct properties."""
        node = Node(
            instruction="Test instruction", agent=self.agent, temperature=0.7
        )

        # Check basic properties
        self.assertEqual(node.instruction, "Test instruction")
        self.assertEqual(node.agent, self.agent)
        self.assertEqual(node.temperature, 0.7)
        self.assertEqual(node.next_nodes, [])

    def test_node_connection(self):
        """Test that nodes can be connected properly."""
        node1 = Node(instruction="First instruction", agent=self.agent)
        node2 = Node(instruction="Second instruction", agent=self.agent)

        # Connect the nodes
        node1.connect_to(node2)

        # Verify connection
        self.assertEqual(len(node1.next_nodes), 1)
        self.assertEqual(node1.next_nodes[0], node2)

    def test_node_run_basic(self):
        """Test basic Node execution with a simple instruction."""
        node = Node(
            instruction="Explain what a computer is in one sentence.",
            agent=self.agent,
            temperature=0.3,
        )

        # Run the node - it should not raise an exception and should store messages in history
        node.run(session_id="test_session")

        # Check that we can retrieve the history (indicating successful execution)
        history = node.get_history("test_session")
        self.assertGreater(
            len(history), 0
        )  # Should have at least system + user messages

    def test_node_sequential_execution(self):
        """Test sequential execution of connected nodes."""
        # Create first node with instruction
        node1 = Node(
            instruction="Explain what a computer is in one sentence.",
            agent=self.agent,
            temperature=0.3,
        )

        # Create second node that will use the context from the first
        node2 = Node(
            instruction="Summarize the previous explanation in one bullet point.",
            agent=self.agent,
            temperature=0.3,
        )

        # Connect nodes
        node1.connect_to(node2)

        # Run the workflow starting with node1
        output = node1.run(session_id="test_workflow")

        # Check that both nodes have generated responses by checking session history
        history1 = node1.get_history("test_workflow")
        history2 = node2.get_history("test_workflow")

        # Both should have messages in their history
        self.assertGreater(len(history1), 4)
        self.assertGreater(len(history2), 4)

        # The second node's message history should include the first node's response
        # (we can't easily check this without more complex assertions but at least we know it ran)


if __name__ == "__main__":
    unittest.main()
