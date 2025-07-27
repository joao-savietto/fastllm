from rich.console import Console
from rich.markdown import Markdown

from fastllm.agent import Agent
from fastllm.workflow import Node


def print_before_generation(node: Node, session_id: str):
    print(f"\nStarting generation for node: {node.instruction}")


def print_after_generation(node: Node, session_id: str):
    messages = node.get_history(session_id)
    m = Markdown(messages[-1]["content"])
    Console().print(m)


# Initialize the agent
agent = Agent(
    model="devstral-small-2505",
    base_url="http://localhost:1234/v1",
    api_key="ollama",
    system_prompt=(
        "You are Qwen, a harmless and useful " "assistant created by Alibaba"
    ),
)

# Create nodes in the workflow
node1 = Node(
    instruction=(
        "Write a rich and detailed explanation "
        "of the events of World War II."
    ),
    agent=agent,
    before_generation=print_before_generation,
    after_generation=print_after_generation,
    temperature=0.8,  # Higher temperature for more creative/diverse output
)

node2 = Node(
    instruction=(
        "Summarize the history into "
        "10 bullet points, focusing on key events and outcomes."
    ),
    agent=agent,
    before_generation=print_before_generation,
    after_generation=print_after_generation,
    temperature=0.3,  # Lower temperature for more focused/structured output
)

# Connect nodes in sequence
node1.connect_to(node2)


def main():
    # Run the workflow starting from node1
    print("\nStarting World War II explanation and summarization workflow")

    # First node will generate detailed history
    node1.run(
        instruction=(
            "Write a comprehensive overview " "of World War II events."
        )
    )


if __name__ == "__main__":
    main()
