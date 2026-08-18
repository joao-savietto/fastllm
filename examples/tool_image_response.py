"""Example: a tool that returns an image to the model.

fastllm tools can send images back to the model in tool responses. A tool
signals images either by:
  - returning raw bytes / bytearray (encoded as a PNG data URI), or
  - returning a dict with an "image" key (bytes, base64 string, data URI
    or http(s) URL).

The Agent then sends the tool message with content parts::

    [{"type": "text", "text": ...},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

This requires a vision-capable model and a server that accepts image parts
in tool messages (Ollama / LM Studio handle them; plain text models will
ignore or reject them).

Usage:
    python examples/tool_image_response.py

Optionally, pass extra per-request parameters (JSON) via FASTLLM_EXTRA_PARAMS,
e.g. for servers that need chat_template_kwargs:
    FASTLLM_EXTRA_PARAMS='{"chat_template_kwargs": {"reasoning_effort": "xhigh"}}' \
        python examples/tool_image_response.py
"""

import base64
import json
import os

from pydantic import BaseModel, Field

from fastllm import Agent, tool

# 1x1 red PNG (demonstration payload; swap for your own image bytes)
TINY_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TINY_RED_PNG = base64.b64decode(TINY_RED_PNG_B64)


def _load_env(path: str) -> None:
    """Minimal .env loader (the repo has no python-dotenv dependency)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError as err:
        print(f"Could not read {path}: {err}")


class ShowImageRequest(BaseModel):
    """Request to show an image."""

    reason: str = Field(description="Why the image is being requested")


@tool("Returns a small image to inspect.", ShowImageRequest)
def show_image(request: ShowImageRequest):
    return {"image": TINY_RED_PNG, "note": "a small red image"}


def main() -> None:
    _load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))

    agent = Agent(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
    )

    extra_params = {}
    raw_params = os.getenv("FASTLLM_EXTRA_PARAMS", "").strip()
    if raw_params:
        try:
            extra_params = json.loads(raw_params)
        except json.JSONDecodeError as err:
            print(f"Invalid FASTLLM_EXTRA_PARAMS JSON: {err}")

    events = agent.generate(
        message="Use the show_image tool to get the image, then tell me its color.",
        session_id="example-image-tool",
        tools=[show_image],
        params=extra_params or None,
    )
    for event in events:
        if "tool_call" in event:
            print("tool_call ->", event["tool_calls"])
        elif event.get("role") == "assistant":
            print(event.get("content", event.get("partial_content", "")), end="")
    print()
    agent.shutdown()


if __name__ == "__main__":
    main()
