
---

# 🚀 FastLLM


FastLLM é uma biblioteca Python com ënfase em leveza e simplicidade de uso, com o objetivo de ajudar a desenvolver projetos baseados em LLMs (Large Language Models). Ele fornece uma interface simples para interagir com provedores compatíveis com OpenAI, incluindo APIs de modelos open-source como Ollama e LM Studio.
O nome desta biblioteca foi inspirado no FastAPI, devido a sua sintaxe limpa e a facilidade de se prototipar uma nova API.

**Importante:** FastLLM é uma biblioteca experimental e em desenvolvimento, e pode não ser adequada para uso em projetos reais.

## 🌟 Funcionalidades

*   💻 Suporte a provedores compatíveis com OpenAI: FastLLM se integra perfeitamente com APIs populares do OpenAI, permitindo que você aproveite o poder dos LLMs em seus aplicativos.
*   🔧 Integração fácil: FastLLM fornece uma API Pythonica que torna fácil incorporar a funcionalidade de LLM nos seus projetos.
*   🛠️ Personalização: Com FastLLM, você pode adicionar facilmente ferramentas e funções (tool calling) personalizadas para estender as capacidades do seu aplicativo baseado em LLM. Utilizamos Pydantic para representar e validar inputs para as suas funções de forma simples e familiar.
*   🔄 Workflows: Crie e gerencie sequências de tarefas ou operações, permitindo interações e automações complexas em seus aplicativos.
*   🧠 **Reflection Agent**: Utilize agentes capazes de planejar, executar, refletir e refinar suas ações (padrão ReAct/Reflexion) para resolver problemas difíceis.
*   🔌 **Model Context Protocol (MCP)**: Transforme seu agente em um cliente MCP, conectando-se a servidores de ferramentas externos via configuração simples.

---

## 📥 Primeiros passos

### 🛠️ Instalação

Para começar com FastLLM, instale a biblioteca usando pip:

```bash
pip install git+https://github.com/joao-savietto/fastllm
```

### 💡 Uso

Veja alguns exemplo de como usar FastLLM em seu script Python:

### Agent

```python
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
    api_key="api-key",
    tools=[sum_numbers],
    system_prompt="You are a helpful assistant",
)

for chunk in agent.generate("Calculate 1900 + 191 using your tool sum_numbers", stream=True):
    print(chunk["partial_content"], end="", flush=True)

```

### Workflow

- Workflows permitem que você crie uma fluxo de prompts que é executado sequencialmente.
- Você pode definit tools diferentes para cada etapa do fluxo.
- Um **Node** representa uma etapa do fluxo.

```python
from fastllm import Agent, Node

agent = Agent(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="http://localhost:1234/v1",
    api_key="api-key",
    tools=[sum_numbers],
    system_prompt="You are a helpful assistant",
)

step1 = Node(
    instruction="Write a outline for a blog post about AI.",
    agent=agent
)

step2 = Node(
    instruction="Write the full blog post based on the outline.",
    agent=agent,
)

step1.connect_to(step2)

step1.run()
```

### BooleanNode
- O **BooleanNode** permite fazer um desvio no fluxo caso uma condição específica seja atendida. Veja o exemplo abaixo:

```python
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from fastllm.agent import Agent
from fastllm.decorators import tool
from fastllm.workflow import BooleanNode, Node


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


def print_response(node: Node, session_id: str, message: str = None):
    m = Markdown(message)
    Console().print(m)


# Create an agent with the BMI calculation tool
agent = Agent(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="http://localhost:1234/v1",
    api_key="ollama",
    tools=[calculate_bmi],
    system_prompt="You are Qwen, a helpful and harmless assistant. You are acting as a virtual fitness instructor",
)

# Create nodes in the workflow
main_node = Node(
    instruction=("Calculate BMI based on weight (90kg) and height (1.75m)"),
    agent=agent,
    before_generation=lambda n, s: print("Calculating BMI..."),
    after_generation=lambda n, s, x: print("Done calculating BMI!"),
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
    after_generation=lambda n, s, x: print(
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

```

### Integração com MCPs

FastLLM tem suporte ao **Model Context Protocol (MCP)**, permitindo que você conecte um agente a servidores MCP externos e suas ferramentas.

Para usar o MCP, crie um arquivo de configuração JSON (ex: `mcp_config.json`) com a seguinte estrutura:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {}
    }
  }
}
```

**Observação:** Essa é a mesma estrutura usada por apps como Claude Desktop, LM Studio, Gemini CLI e outros.

<p>Você pode ver mais exemplos na pasta "examples" do repositório! </p>

## 📄 Licença

FastLLM é liberado sob a Licença MIT. Consulte o arquivo LICENSE para mais informações.

## 💖 Agradecimentos!

Esperamos que você ache FastLLM útil em seus projetos baseados em LLM!

---
