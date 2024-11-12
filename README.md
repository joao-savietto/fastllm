# FastLLM
=====================================

FastLLM is a lightweight Python library designed to help build LLM (Large Language Model) based applications. It provides a simple interface for interacting with OpenAI compatible providers, including open-source models APIs like Ollama and LM Studio.

## Features

*   Support for OpenAI compatible providers: FastLLM integrates seamlessly with popular OpenAI APIs, allowing you to leverage the power of LLMs in your applications.
*   Easy integration: FastLLM provides a simple, Pythonic API that makes it easy to incorporate LLM functionality into your projects.
*   Customizability: With FastLLM, you can easily add custom tools and functions to extend the capabilities of your LLM-based application.

## Getting Started

### Installation

To get started with FastLLM, install the library using pip:

```bash
python setup.py sdist bdist_wheel
pip install .
```

### Usage

Here's an example of how to use FastLLM in your Python script:

```python
from fastllm import Agent, tool
from pydantic import BaseModel, Field


class SumRequest(BaseModel):
    num1: int = Field(..., description="The first number to be added")
    num2: int = Field(..., description="The second number to be added")

@tool(
    description="Sums two numbers and returns the result",
    pydantic_model=SumRequest
)
def sum_numbers(num1, num2):
    print("Params:", num1, "+", num2)
    return {"result": num1 + num2}

agent = Agent(
               model="qwen2.5-coder:7b-instruct-q8_0",
               base_url="http://localhost:11434/v1",
               api_key="ollama",
               tools=[sum_numbers],
            )

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Calculate 1900 + 191"
    }
]

for message in agent.generate(messages):
    print(message)
```

## Contributing

We welcome contributions to FastLLM! If you have any ideas or suggestions for improving the library, please don't hesitate to reach out.

## License

FastLLM is released under the MIT License. See the LICENSE file for more information.

## Contact Us

For any questions or concerns about FastLLM, feel free to contact us at [your email address].

We hope you find FastLLM helpful in your LLM-based project endeavors!