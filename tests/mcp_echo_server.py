from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo Server")

@mcp.tool()
def echo(message: str) -> str:
    """Echoes back the input message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    mcp.run()
