import asyncio
import json
import threading
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

class MCPToolWrapper:
    def __init__(self, client: "MCPClient", tool_model: Tool):
        self.client = client
        self.tool_model = tool_model
        self.__name__ = tool_model.name # For debug/logging

    def tool_json(self):
        return {
            "type": "function",
            "function": {
                "name": self.tool_model.name,
                "description": self.tool_model.description,
                "parameters": self.tool_model.inputSchema
            }
        }

    def execute(self, **kwargs):
        # We need to make sure we are calling the correct tool on the correct server
        # The wrapper knows its own name, so we can pass it.
        # However, call_tool expects arguments dict.
        return self.client.call_tool(self.tool_model.name, kwargs)


class MCPClient:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.ready_event = threading.Event()
        self.stop_event = asyncio.Event()
        self.tools: List[Tool] = []
        # Map tool name to server name. 
        # Note: If multiple servers have tools with same name, last one wins.
        # Ideally we should namespace them, but for now simplistic approach.
        self._tool_server_map: Dict[str, str] = {} 

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_task())

    async def _main_task(self):
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            mcp_servers = config.get("mcpServers", {})

            async with self.exit_stack:
                for name, server_config in mcp_servers.items():
                    command = server_config.get("command")
                    args = server_config.get("args", [])
                    env = server_config.get("env", {})

                    server_params = StdioServerParameters(
                        command=command,
                        args=args,
                        env=env
                    )

                    # Connect to server
                    read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    
                    self.sessions[name] = session

                    # List tools
                    result = await session.list_tools()
                    for tool in result.tools:
                        self.tools.append(tool)
                        self._tool_server_map[tool.name] = name

                self.ready_event.set()
                # Wait until stopped
                await self.stop_event.wait()
                
        except Exception as e:
            print(f"Error in MCP client loop: {e}")
            # Ensure we unblock start() if it failed
            self.ready_event.set() 

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
        self.ready_event.wait()

    def get_tools(self) -> List[Any]:
        # Return wrappers that match FastLLM tool interface
        wrappers = []
        for tool in self.tools:
            wrappers.append(MCPToolWrapper(self, tool))
        return wrappers

    def call_tool(self, name: str, arguments: dict) -> Any:
        if name not in self._tool_server_map:
            raise ValueError(f"Tool {name} not found")
        
        server_name = self._tool_server_map[name]
        session = self.sessions[server_name]
        
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(name, arguments),
            self.loop
        )
        try:
            result = future.result(timeout=30) # Add timeout to avoid hanging indefinitely
        except Exception as e:
            return f"Error executing tool {name}: {str(e)}"
        
        # Format result
        # result is CallToolResult
        # result.content is list of TextContent or ImageContent or EmbeddedResource
        output = []
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    output.append(content.text)
                elif hasattr(content, 'data'):
                     output.append("[Image Data]") 
                elif hasattr(content, 'resource'):
                     output.append(f"[Resource: {content.resource.uri}]")
                else:
                    output.append(str(content))
        else:
            output.append(str(result))

        return "\n".join(output)

    def stop(self):
        def _stop():
            self.stop_event.set()
        
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(_stop)
        self.thread.join(timeout=2)
