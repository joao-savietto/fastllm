
---

# 🚀 FastLLM


FastLLM é uma biblioteca Python com ënfase em leveza e simplicidade de uso, com o objetivo de ajudar a desenvolver projetos baseados em LLMs (Large Language Models). Ele fornece uma interface simples para interagir com provedores compatíveis com OpenAI, incluindo APIs de modelos open-source como Ollama e LM Studio.
O nome desta biblioteca foi inspirado no FastAPI, devido a sua sintaxe limpa e a facilidade de se prototipar uma nova API.

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
    num1: int = Field(..., description="O primeiro número a ser adicionado")
    num2: int = Field(..., description="O segundo número a ser adicionado")

@tool(
    description="Soma dois números e retorna o resultado",
    pydantic_model=SumRequest
)
def sum_numbers(inputs: SumRequest):
    print("Parâmetros:", inputs.num1, "+", inputs.num2)
    return {"result": inputs.num1 + inputs.num2}

agent = Agent(
               model="qwen2.5:14b-instruct-q6_K",
               base_url="http://localhost:11434/v1",
               api_key="ollama",
               tools=[sum_numbers],
               system_prompt="Você é um assistente útil"
            )

for message in agent.generate("Calcular 1900 + 191"):
    print(message)
```

<p>Você pode ver mais exemplos na pasta "examples" do repositório! </p>

## 📄 Licença

FastLLM é liberado sob a Licença MIT. Consulte o arquivo LICENSE para mais informações.

## 💖 Agradecimentos!

Esperamos que você encontre FastLLM útil em seus projetos baseados em LLM!

---

Let me know if you'd like any adjustments!