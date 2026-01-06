import unittest
from unittest.mock import patch, MagicMock

from pydantic import ValidationError

# Import the tool and its request model
from fastllm.tools.bash_tool import run_bash, BashCommandModel
import subprocess


class TestBashTool(unittest.TestCase):
    """Tests for the `run_bash` terminal‑execution tool."""

    @patch("fastllm.tools.bash_tool.subprocess.run")
    def test_successful_command(self, mock_run):
        """
        The tool should return stdout, stderr and returncode when subprocess.run succeeds.
        """
        # Arrange – fake a successful subprocess result
        mocked_result = MagicMock()
        mocked_result.stdout = "file1\nfile2\n"
        mocked_result.stderr = ""
        mocked_result.returncode = 0
        mock_run.return_value = mocked_result

        # Act – call the tool with a valid request model
        req = BashCommandModel(command="ls -1", cwd="/tmp", timeout=5)
        response = run_bash(req)

        # Assert – verify the shape of the returned dict
        self.assertEqual(response["stdout"], "file1\nfile2\n")
        self.assertEqual(response["stderr"], "")
        self.assertEqual(response["returncode"], 0)

        # Also ensure subprocess.run received the correct arguments
        mock_run.assert_called_once_with(
            "ls -1",
            shell=True,
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=5,
        )

    @patch("fastllm.tools.bash_tool.subprocess.run")
    def test_command_error_propagates(self, mock_run):
        """
        If subprocess.run raises an exception (e.g., timeout), the tool should return
        a dict with an ``error`` key containing the stringified exception.
        """
        # Arrange – make subprocess.run raise a TimeoutExpired error
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 10", timeout=1)

        req = BashCommandModel(command="sleep 10", timeout=1)
        response = run_bash(req)

        self.assertIn("error", response)
        self.assertIsInstance(response["error"], str)
        self.assertIn("TimeoutExpired", response["error"])

    def test_validation_rejects_empty_command(self):
        """
        The Pydantic model must reject an empty or whitespace‑only command.
        """
        with self.assertRaises(ValidationError) as ctx:
            BashCommandModel(command="   ")
        # Optional: check that the error message mentions our validator
        self.assertIn("Command must not be empty", str(ctx.exception))

    def test_default_values(self):
        """
        Verify that optional fields get sensible defaults when omitted.
        """
        req = BashCommandModel(command="echo hello")
        # cwd should default to None and timeout to 30 (as defined in the model)
        self.assertIsNone(req.cwd)
        self.assertEqual(req.timeout, 30)


if __name__ == "__main__":
    unittest.main()
