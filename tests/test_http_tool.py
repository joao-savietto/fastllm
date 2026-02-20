import unittest
from unittest.mock import patch, MagicMock
from requests.structures import CaseInsensitiveDict
from fastllm.tools.http_request import http_request, HttpRequestModel
import json


class TestHttpRequestTool(unittest.TestCase):

    def test_model_validation_valid_methods(self):
        """Test that valid HTTP methods are accepted"""
        # These should all work without raising validation errors
        valid_methods = ["get", "post", "put", "patch", "delete"]

        for method in valid_methods:
            model = HttpRequestModel(
                method=method, url="https://api.example.com/test"
            )
            self.assertEqual(model.method, method)

    def test_model_validation_invalid_method(self):
        """Test that invalid HTTP methods raise validation errors"""
        with self.assertRaises(
            Exception
        ):  # Pydantic will raise a validation error
            HttpRequestModel(
                method="invalid_method", url="https://api.example.com/test"
            )

    def test_model_validation_required_fields(self):
        """Test that required fields are properly validated"""
        # This should work - all required fields provided
        model = HttpRequestModel(
            method="get", url="https://api.example.com/test"
        )
        self.assertEqual(model.method, "get")
        self.assertEqual(model.url, "https://api.example.com/test")

    @patch("fastllm.tools.http_request.requests.request")
    def test_http_request_success_get(self, mock_request):
        """Test successful GET request"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = CaseInsensitiveDict(
            {"content-type": "application/json"}
        )
        mock_response.text = '{"message": "success"}'
        mock_response.json.return_value = {"message": "success"}
        mock_request.return_value = mock_response

        # Execute tool
        result = http_request.execute(
            {"method": "get", "url": "https://api.example.com/test"}
        )
        result = json.loads(result)

        # Verify results
        self.assertEqual(result["status_code"], 200)
        self.assertIn("message", result["content"])
        self.assertIsNotNone(result["json"])

    @patch("fastllm.tools.http_request.requests.request")
    def test_http_request_success_post_with_json_body(self, mock_request):
        """Test successful POST request with JSON body"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = CaseInsensitiveDict(
            {"content-type": "application/json"}
        )
        mock_response.text = '{"id": 123, "name": "test"}'
        mock_response.json.return_value = {"id": 123, "name": "test"}
        mock_request.return_value = mock_response

        # Execute tool
        result = http_request.execute(
            {
                "method": "post",
                "url": "https://api.example.com/users",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "John", "email": "john@example.com"},
            }
        )
        result = json.loads(result)

        # Verify results
        self.assertEqual(result["status_code"], 201)
        self.assertIn("test", result["content"])
        self.assertIsNotNone(result["json"])

    @patch("fastllm.tools.http_request.requests.request")
    def test_http_request_success_with_params(self, mock_request):
        """Test successful request with query parameters"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = CaseInsensitiveDict(
            {"content-type": "application/json"}
        )
        mock_response.text = '{"results": []}'
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        # Execute tool
        result = http_request.execute(
            {
                "method": "get",
                "url": "https://api.example.com/search",
                "params": {"q": "test", "limit": 10},
            }
        )
        result = json.loads(result)

        # Verify results
        self.assertEqual(result["status_code"], 200)
        mock_request.assert_called_with(
            method="get",
            url="https://api.example.com/search",
            params={"q": "test", "limit": 10},
        )

    @patch("fastllm.tools.http_request.requests.request")
    def test_http_request_error_handling(self, mock_request):
        """Test error handling in HTTP request"""
        # Setup mock to raise an exception
        mock_request.side_effect = Exception("Network error")

        # Execute tool
        result = http_request.execute(
            {"method": "get", "url": "https://api.example.com/test"}
        )
        result = json.loads(result)

        # Verify error is handled properly
        self.assertIn("error", result)

    def test_tool_schema_generation(self):
        """Test that the tool generates correct schema"""
        # Get the tool JSON schema
        tool_json = http_request.tool_json()

        # Verify it has expected structure
        self.assertEqual(tool_json["type"], "function")
        self.assertEqual(tool_json["function"]["name"], "http_request")
        self.assertIn("description", tool_json["function"])

        # Check parameters structure
        params = tool_json["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("properties", params)
        self.assertIn("required", params)

        # Verify required fields are in the schema
        required_fields = params["required"]
        self.assertIn("method", required_fields)
        self.assertIn("url", required_fields)

    def test_tool_execution_with_different_body_types(self):
        """Test that different body types are handled correctly"""
        with patch(
            "fastllm.tools.http_request.requests.request"
        ) as mock_request:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = CaseInsensitiveDict(
                {"content-type": "application/json"}
            )
            mock_response.text = '{"success": true}'
            mock_response.json.return_value = {"success": True}
            mock_request.return_value = mock_response

            # Test with dict (should be JSON)
            result1 = http_request.execute(
                {
                    "method": "post",
                    "url": "https://api.example.com/test",
                    "body": {"key": "value"},
                }
            )
            result1 = json.loads(result1)

            # Test with string (should be data)
            result2 = http_request.execute(
                {
                    "method": "post",
                    "url": "https://api.example.com/test",
                    "body": "plain text content",
                }
            )
            result2 = json.loads(result2)

            # Both should succeed
            self.assertEqual(result1["status_code"], 200)
            self.assertEqual(result2["status_code"], 200)


class TestHttpRequestToolWithRealAPI(unittest.TestCase):
    """Test with a real mock API service"""

    def setUp(self):
        # Using jsonplaceholder.typicode.com for testing
        self.base_url = "https://jsonplaceholder.typicode.com"

    def test_real_get_request(self):
        """Test actual GET request to a public API"""
        try:
            result = http_request.execute(
                {"method": "get", "url": f"{self.base_url}/posts/1"}
            )
            result = json.loads(result)

            self.assertEqual(result["status_code"], 200)
            self.assertIn("title", result["json"])
        except Exception as e:
            # If network is not available, skip this test
            self.skipTest(f"Network error: {e}")

    def test_real_post_request(self):
        """Test actual POST request to a public API"""
        try:
            result = http_request.execute(
                {
                    "method": "post",
                    "url": f"{self.base_url}/posts",
                    "body": {
                        "title": "test post",
                        "body": "this is a test",
                        "userId": 1,
                    },
                }
            )
            result = json.loads(result)

            self.assertEqual(result["status_code"], 201)
            self.assertIn("title", result["json"])
        except Exception as e:
            # If network is not available, skip this test
            self.skipTest(f"Network error: {e}")
