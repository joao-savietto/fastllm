"""
Unit tests for formatted outputs in Agent class.

Tests verify that agents can generate responses in specific Pydantic models formats.
"""

import os
import unittest
from pydantic import BaseModel, Field
from fastllm.agent import Agent
from fastllm.store import InMemoryChatStorage


# Test models for response formatting
class UserResponse(BaseModel):
    name: str = Field(..., description="The user's name")
    age: int = Field(..., description="The user's age")
    email: str = Field(..., description="The user's email address")


class ProductReview(BaseModel):
    product_name: str = Field(..., description="Name of the product reviewed")
    rating: float = Field(..., description="Rating from 1.0 to 5.0")
    review_text: str = Field(..., description="Text of the review")


class ProductList(BaseModel):
    """A list of product reviews"""

    products: list[ProductReview] = Field(
        ..., description="List of product reviews"
    )


class TestFormattedOutputs(unittest.TestCase):
    """Test formatted outputs in Agent class."""

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

    def test_agent_generate_with_response_format(self):
        """Test that agent can generate responses with a specific format."""
        # Test with UserResponse model
        result = self.agent.generate(
            message="Create a user profile for Alice who is 30 years old and has email alice@example.com",
            session_id="test_session",
            stream=False,
            response_format=UserResponse,
        )

        # Check that we got a structured response
        self.assertIn("role", result)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)

        # The content should be JSON parsable and match the model structure
        import json

        parsed_content = json.loads(result["content"])
        self.assertIn("name", parsed_content)
        self.assertIn("age", parsed_content)
        self.assertIn("email", parsed_content)

        # Validate values are correct
        self.assertEqual(parsed_content["name"], "Alice")
        self.assertEqual(parsed_content["age"], 30)
        self.assertEqual(parsed_content["email"], "alice@example.com")

    def test_agent_generate_with_different_response_format(self):
        """Test that agent can generate responses with different formats."""
        # Test with ProductReview model
        result = self.agent.generate(
            message="Write a review for 'Smartphone X' with 4.5 stars and the text: 'Great phone, very reliable!'",
            session_id="test_session2",
            stream=False,
            response_format=ProductReview,
        )

        # Check that we got a structured response
        self.assertIn("role", result)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)

        # Validate content structure is correct
        import json

        parsed_content = json.loads(result["content"])
        self.assertIn("product_name", parsed_content)
        self.assertIn("rating", parsed_content)
        self.assertIn("review_text", parsed_content)

        # Validate values are correct
        self.assertEqual(parsed_content["product_name"], "Smartphone X")
        self.assertEqual(parsed_content["rating"], 4.5)

    def test_agent_generate_stream_with_response_format(self):
        """Test that agent can stream responses with a specific format."""
        # Test streaming with UserResponse model
        chunks = list(
            self.agent.generate(
                message="Create a user profile for Bob who is 25 years old and has email bob@example.com",
                session_id="test_session3",
                stream=True,
                response_format=UserResponse,
            )
        )

        # Check that we got some chunks back, including content
        self.assertGreater(len(chunks), 0)

        # Check for partial_content in at least one chunk
        has_partial = any("partial_content" in chunk for chunk in chunks)
        self.assertTrue(
            has_partial, "Should have partial content chunks when streaming"
        )

    def test_agent_generate_with_none_response_format(self):
        """Test that agent works normally without specifying a response format."""
        result = self.agent.generate(
            message="What's your name?",
            session_id="test_session4",
            stream=False,
            response_format=None,  # Explicitly set to None
        )

        # Should still work like normal generation
        self.assertIn("role", result)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)

    def test_response_format_parameter_propagation(self):
        """Test that the response_format parameter properly propagates through agent methods."""
        # This should not raise an exception in parameter handling
        self.agent.generate(
            message="Test",
            session_id="test_session5",
            stream=False,
            response_format=UserResponse,  # Valid Pydantic model
        )

        # If we get here without error, that means our parameter processing works

    def test_agent_generate_with_nested_response_format(self):
        """Test that agent can generate responses with nested Pydantic models."""
        result = self.agent.generate(
            message="Create reviews for 3 products: Smartphone X (4.5 stars), Laptop Y (3.0 stars), and Tablet Z (4.8 stars). Each review should include the product name, rating, and review text.",
            session_id="test_session6",
            stream=False,
            response_format=ProductList,
        )

        # Check that we got a structured response with nested objects
        self.assertIn("role", result)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)

        # Validate content structure is correct - should have products list
        import json

        parsed_content = json.loads(result["content"])
        self.assertIn("products", parsed_content)

        # Should have 3 product reviews in the list
        self.assertEqual(len(parsed_content["products"]), 3)

        # Validate each product has required fields
        for product in parsed_content["products"]:
            self.assertIn("product_name", product)
            self.assertIn("rating", product)
            self.assertIn("review_text", product)


if __name__ == "__main__":
    unittest.main()
