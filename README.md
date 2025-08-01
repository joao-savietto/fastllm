
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

## 📥 Primeiros passos

### 🛠️ Instalação

Para começar com FastLLM, instale a biblioteca usando pip:

```bash
python setup.py sdist bdist_wheel
pip install .
```

### 💡 Uso

Aqui está um exemplo de como usar FastLLM em seu script Python:

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
    api_key="ollama",
    tools=[sum_numbers],
    system_prompt="You are a helpful assistant",
)

for chunk in agent.generate("Calculate 1900 + 191 using your tool sum_numbers", stream=True):
    print(chunk["partial_content"], end="", flush=True)

```

<p>Você pode ver mais exemplos na pasta "examples" do repositório! </p>

## 📄 Licença

FastLLM é liberado sob a Licença MIT. Consulte o arquivo LICENSE para mais informações.

## 💖 Agradecimentos!

Esperamos que você ache FastLLM útil em seus projetos baseados em LLM!

---
