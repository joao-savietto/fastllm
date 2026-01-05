# test_bmi_workflow_integration.py
"""
Integration tests for the workflow module using real OpenAI API calls.
These tests replace mocked API interactions with actual calls, pulling
configuration from environment variables (e.g., .env file).
"""

import os
import unittest
from fastllm.agent import Agent
from fastllm.store import InMemoryChatStorage


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
                    os.environ.setdefault(key.strip(), value.strip().strip('"'))
        except FileNotFoundError:
            pass  # Use existing environment variables

        # Retrieve configuration from environment
        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
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
            {"role": "system", "content": "You are a helpful assistant"}, session_id="test"
        )
        self.store.save(
            {"role": "user", "content": "Say hello in a friendly way."}, session_id="test"
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


if __name__ == "__main__":
    unittest.main()