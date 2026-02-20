import json

import pytest
from pydantic import BaseModel

from fastllm.decorators import (pydantic_to_openai_schema, retry,
                                run_in_thread, streamable_response, tool)


class DummyModel(BaseModel):
    name: str
    age: int


def dummy_func(model: DummyModel) -> dict:
    return {"name": model.name, "age": model.age}


def test_tool_creates_attributes():
    decorated = tool("My description", DummyModel)(dummy_func)
    assert hasattr(decorated, "tool_json")
    assert hasattr(decorated, "execute")


def test_execute_returns_json():
    decorated = tool("Desc", DummyModel)(dummy_func)
    result = decorated.execute(name="test", age=42)
    parsed = json.loads(result)
    assert parsed == {"name": "test", "age": 42}


def testpydantic_to_openai_schema():
    params = pydantic_to_openai_schema(DummyModel)
    assert set(params["properties"].keys()) == {"name", "age"}
    assert params["properties"]["name"]["type"] == "string"
    assert params["required"] == ["name", "age"]


def test_streamable_response_non_stream():
    def gen():
        yield "first"
        yield "second"

    decorated = streamable_response(gen)
    result = decorated()  # invokes without stream kwarg, defaults to False
    assert result == "first"


def test_run_in_thread_starts_thread():
    result = {"executed": False}

    def target():
        result["executed"] = True

    threaded_func = run_in_thread(target)
    threaded_func()
    import time as t

    t.sleep(0.1)  # give thread a moment to execute
    assert result["executed"] is True
