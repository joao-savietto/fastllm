from fastllm import Agent

agent = Agent(
    model="qwen_qwen3-vl-30b-a3b-instruct",
    base_url="http://localhost:1234/v1",
    mcp_config_path="mcp.json"
)

res = agent.generate(
    message="What is the latest Minecraft version in Fenurary 2026?",
    stream=False
)
print(res)

agent.shutdown()