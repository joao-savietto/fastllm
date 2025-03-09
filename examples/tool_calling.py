from fastllm import Agent, tool
from pydantic import BaseModel, Field


class SumRequest(BaseModel):
    num1: int = Field(..., description="The first number to be added")
    num2: int = Field(..., description="The second number to be added")


@tool(
    description="Sums two numbers and returns the result",
    pydantic_model=SumRequest
)
def sum_numbers(inputs: SumRequest):
    print("Params:", inputs.num1, "+", inputs.num2)
    return {"result": inputs.num1 + inputs.num2}


agent = Agent(
               model="qwen2.5:14b-instruct-q6_K",
               base_url="http://localhost:11434/v1",
               api_key="ollama",
               tools=[sum_numbers],
               system_prompt="You are a helpful assistant"
            )

for message in agent.generate("Calculate 1900 + 191"):
    print(message)
