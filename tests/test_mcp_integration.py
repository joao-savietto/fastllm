import json
import os
import sys
import time
import pytest
from fastllm import Agent

# Path to the echo server script
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_echo_server.py")

@pytest.fixture
def mcp_config(tmp_path):
    config_path = tmp_path / "mcp_config.json"
    config_content = {
        "mcpServers": {
            "echo-server": {
                "command": sys.executable,
                "args": [SERVER_SCRIPT],
                "env": {"PYTHONUNBUFFERED": "1"}
            }
        }
    }
    with open(config_path, "w") as f:
        json.dump(config_content, f)
    return str(config_path)

def test_mcp_agent_integration(mcp_config):
    # Initialize Agent with MCP config
    agent = Agent(mcp_config_path=mcp_config)
    
    # Wait a bit for connection and tool listing (it's async in background)
    # The Agent constructor starts the client but doesn't block until tools are ready?
    # Wait, in my implementation:
    # Agent.__init__ calls self.mcp_client.start()
    # MCPClient.start() waits for ready_event.
    # So it should be synchronous from Agent's perspective.
    
    try:
        assert agent.mcp_client is not None
        
        # Verify tool exists
        assert "echo" in agent.tool_map
        
        # Verify tool execution
        tool = agent.tool_map["echo"]
        result = tool.execute(message="Hello MCP")
        
        # Result should be the raw string from the tool
        assert result == "Echo: Hello MCP"

    finally:
        agent.shutdown()
