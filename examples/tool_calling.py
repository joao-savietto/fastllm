from fastllm import Agent, tool
from pydantic import BaseModel, Field


class SumRequest(BaseModel):
    num1: int = Field(..., description="The first number to be added")
    num2: int = Field(..., description="The second number to be added")


@tool(
    description="Sums two numbers and returns the result",
    pydantic_model=SumRequest,
)
def sum_numbers(inputs: SumRequest):
    print("Params:", inputs.num1, "+", inputs.num2)
    return {"result": inputs.num1 + inputs.num2}


agent = Agent(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="http://localhost:1234/v1",
    api_key="ollama",
    tools=[sum_numbers],
    system_prompt="You are a helpful assistant",
)

for chunk in agent.generate("Calculate 1900 + 191 using your tool sum_numbers", stream=True):
    print(chunk["partial_content"], end="", flush=True)
