from rich.console import Console
from rich.markdown import Markdown

from pydantic import BaseModel, Field

from fastllm.agent import Agent
from fastllm.workflow import BooleanNode, Node
from fastllm.decorators import tool


# Create a Pydantic model for the BMI calculation request
class BMICalculationRequest(BaseModel):
    weight_kg: float = Field(..., description="Weight in kilograms")
    height_m: float = Field(..., description="Height in meters")


@tool(
    description="Calculates Body Mass Index (BMI) based on weight and height",
    pydantic_model=BMICalculationRequest,
)
def calculate_bmi(request: BMICalculationRequest):
    """Calculate BMI using the formula: BMI = weight / (height^2)"""
    print("Params:", request.weight_kg, "kg,", request.height_m, "m")

    # Calculate BMI
    bmi = request.weight_kg / (request.height_m**2)

    # Classify BMI category
    if bmi < 18.5:
        classification = "underweight"
    elif 18.5 <= bmi < 25:
        classification = "normal weight"
    else:
        classification = "overweight"

    return {"bmi": round(bmi, 2), "classification": classification}


def print_response(node: Node, session_id: str):
    messages = node.get_history(session_id)
    m = Markdown(messages[-1]["content"])
    Console().print(m)


# Create an agent with the BMI calculation tool
agent = Agent(
    model="qwen2.5:32b-instruct-q5_K_M",
    base_url="http://10.147.17.172:11434/v1",
    api_key="ollama",
    tools=[calculate_bmi],
    system_prompt="You are Qwen, a helpful and harmful assistant. You are acting as health assistant",
)

# Create nodes in the workflow
main_node = Node(
    instruction=("Calculate BMI based on weight (90kg) and height (1.75m)"),
    agent=agent,
    before_generation=lambda n, s: print("Calculating BMI..."),
    after_generation=lambda n, s: print("Done calculating BMI!"),
    temperature=0.3,
)

bmi_check = BooleanNode(
    condition=lambda self, session_id: (
        "overweight" in self.get_history(session_id)[-1]["content"]
    ),
    instruction_true="Provide a detailed weight loss plan for someone who is overweight",
    instruction_false="Provide a healthy lifestyle recommendation based on BMI classification",
)

# Create nodes for different BMI categories
overweight_node = Node(
    instruction="Create a personalized weight loss program including diet and exercise recommendations",
    agent=agent,
    before_generation=lambda n, s: print("\nGenerating weight loss plan..."),
    after_generation=print_response,
    temperature=0.8,
)

normal_weight_node = Node(
    instruction="Provide tips for maintaining a healthy lifestyle and preventing weight gain",
    agent=agent,
    before_generation=lambda n, s: print(
        "\nGenerating healthy lifestyle tips..."
    ),
    after_generation=print_response,
    temperature=0.7,
)

underweight_node = Node(
    instruction="Suggest ways to gain weight healthily through diet and exercise",
    agent=agent,
    before_generation=lambda n, s: print(
        "Generating healthy weight gain plan..."
    ),
    after_generation=lambda n, s: print(
        "Weight gain recommendations complete!\n"
    ),
    temperature=0.6,
)

# Connect nodes in the workflow
main_node.connect_to(bmi_check)
bmi_check.connect_to_true(overweight_node)
bmi_check.connect_to_false(normal_weight_node)
normal_weight_node.connect_to(underweight_node)


def main():
    print("\nStarting BMI calculator and recommendations workflow...\n")

    # Run the workflow with specific weight and height
    main_node.run(
        instruction="Calculate BMI for someone who weighs 70 kg and is 1.8 meters tall"
    )


if __name__ == "__main__":
    main()
