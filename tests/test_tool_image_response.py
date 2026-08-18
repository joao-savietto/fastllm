"""
Offline tests for the openai 3.x SDK upgrade and image support in tool
responses.

No real LLM endpoint is required: the OpenAI client is replaced by fakes
that build responses from the real openai SDK response models.  Building
those models from the openai 3.x type paths doubles as an upgrade
compatibility check.
"""

import base64
import json
import unittest
from collections.abc import Generator
from types import SimpleNamespace
from unittest import mock

import httpx2
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from pydantic import BaseModel, Field

from fastllm.agent import Agent
from fastllm.decorators import build_image_data_uri, serialize_tool_result, tool
from fastllm.exceptions import EmptyPayload
from fastllm.mcp_client import MCPClient
from fastllm.store import InMemoryChatStorage

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG).decode()
TINY_PNG_URI = f"data:image/png;base64,{TINY_PNG_B64}"


class GetImageRequest(BaseModel):
    path: str = Field(..., description="Path of the image to return")


@tool("Returns a tiny red PNG image", GetImageRequest)
def get_image(request: GetImageRequest):
    """Return image bytes plus a caption (the feature under test)."""
    return {"image": TINY_PNG, "caption": "a red pixel"}


class TestBuildImageDataUri(unittest.TestCase):
    def test_bytes_are_encoded_as_data_uri(self):
        self.assertEqual(build_image_data_uri(TINY_PNG), TINY_PNG_URI)

    def test_bytes_with_custom_mime(self):
        uri = build_image_data_uri(TINY_PNG, mime="image/jpeg")
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_bytearray_is_encoded(self):
        self.assertEqual(build_image_data_uri(bytearray(TINY_PNG)), TINY_PNG_URI)

    def test_data_uri_passes_through(self):
        self.assertEqual(build_image_data_uri(TINY_PNG_URI), TINY_PNG_URI)

    def test_url_passes_through(self):
        url = "https://example.com/a.png"
        self.assertEqual(build_image_data_uri(url), url)

    def test_bare_base64_is_wrapped(self):
        self.assertEqual(build_image_data_uri(TINY_PNG_B64), TINY_PNG_URI)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            build_image_data_uri(12345)


class TestSerializeToolResult(unittest.TestCase):
    def test_raw_bytes_become_image_dict(self):
        self.assertEqual(
            json.loads(serialize_tool_result(TINY_PNG)),
            {"image": TINY_PNG_URI},
        )

    def test_dict_with_bytes_image_is_encoded(self):
        parsed = json.loads(
            serialize_tool_result({"image": TINY_PNG, "caption": "red"})
        )
        self.assertEqual(parsed["image"], TINY_PNG_URI)
        self.assertEqual(parsed["caption"], "red")

    def test_plain_dict_unchanged(self):
        result = serialize_tool_result({"a": 1, "b": [2, 3]})
        self.assertEqual(json.loads(result), {"a": 1, "b": [2, 3]})

    def test_tool_execute_with_bytes_image(self):
        @tool("Raw image tool", GetImageRequest)
        def raw_image(request: GetImageRequest):
            return TINY_PNG

        result = raw_image.execute(path="x.png")  # pyright: ignore[reportFunctionMemberAccess]
        self.assertEqual(json.loads(result), {"image": TINY_PNG_URI})

    def test_tool_execute_with_dict_image(self):
        result = get_image.execute(path="x.png")  # pyright: ignore[reportFunctionMemberAccess]
        self.assertEqual(json.loads(result)["image"], TINY_PNG_URI)


class TestAgentToolMessageContent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(
            model="test-model",
            base_url="http://localhost:9/v1",
            api_key="test-key",
            store=InMemoryChatStorage(),
        )

    def test_plain_string_unchanged(self):
        self.assertEqual(self.agent._build_tool_message_content("hello", "t"), "hello")

    def test_non_json_string_unchanged(self):
        self.assertEqual(
            self.agent._build_tool_message_content("not json", "t"), "not json"
        )

    def test_json_string_without_image_unchanged(self):
        payload = json.dumps({"a": 1})
        self.assertEqual(self.agent._build_tool_message_content(payload, "t"), payload)

    def test_json_string_with_data_uri_image(self):
        payload = json.dumps({"image": TINY_PNG_URI, "caption": "red"})
        parts = self.agent._build_tool_message_content(payload, "t")
        self.assertEqual(
            parts[0], {"type": "text", "text": json.dumps({"caption": "red"})}
        )
        self.assertEqual(
            parts[1], {"type": "image_url", "image_url": {"url": TINY_PNG_URI}}
        )

    def test_bare_string_image_is_ambiguous_and_kept_as_text(self):
        payload = json.dumps({"image": "some ambiguous text"})
        self.assertEqual(self.agent._build_tool_message_content(payload, "t"), payload)

    def test_dict_with_bytes_image(self):
        parts = self.agent._build_tool_message_content(
            {"image": TINY_PNG, "note": "x"}, "t"
        )
        self.assertEqual(
            parts[1], {"type": "image_url", "image_url": {"url": TINY_PNG_URI}}
        )

    def test_dict_with_multiple_images(self):
        parts = self.agent._build_tool_message_content(
            {"image": [TINY_PNG, TINY_PNG_URI], "note": "x"}, "t"
        )
        image_parts = [p for p in parts if p["type"] == "image_url"]
        self.assertEqual(len(image_parts), 2)

    def test_url_image(self):
        parts = self.agent._build_tool_message_content(
            {"image": "https://example.com/a.png"}, "t"
        )
        self.assertEqual(
            parts[1],
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        )

    def test_raw_bytes_result(self):
        parts = self.agent._build_tool_message_content(TINY_PNG, "t")
        self.assertEqual(parts[0]["text"], "[tool 't' returned an image]")
        self.assertEqual(
            parts[1], {"type": "image_url", "image_url": {"url": TINY_PNG_URI}}
        )

    def test_plain_dict_result_stays_json(self):
        result = self.agent._build_tool_message_content({"a": 1}, "t")
        self.assertEqual(json.loads(result), {"a": 1})


class _FakeCompletions:
    """Stands in for ``client.chat.completions.create`` in offline tests.

    Builds responses from the real openai 3.x response models. The
    signature is deliberately strict (keyword-only, no ``**kwargs``),
    mirroring openai >= 3's strict create() validation, so that
    ``Agent._merge_extra_params`` routes unknown params into extra_body.
    """

    _UNSET = object()

    def __init__(self, mode):
        self.mode = mode  # "stream" or "nonstream"
        self.calls = []

    def create(
        self,
        *,
        messages,
        model,
        tools=_UNSET,
        stream: bool = False,
        temperature: float | None = None,
        extra_body: dict | None = None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": None if tools is self._UNSET else tools,
                "stream": stream,
                "temperature": temperature,
                "extra_body": extra_body,
            }
        )
        # the agentic loop sends `tools` on every round, so rounds are
        # distinguished by call index instead of the tools kwarg
        is_first_call = len(self.calls) == 1
        if self.mode == "stream":
            return list(self._stream(is_first_call))
        return self._nonstream(is_first_call)

    def _stream(self, is_first_call):
        if is_first_call:
            yield ChatCompletionChunk(
                id="cmpl-1",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=ChoiceDeltaToolCallFunction(
                                        name="get_image",
                                        arguments='{"path": "x.png"}',
                                    ),
                                )
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
                created=1,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield ChatCompletionChunk(
                id="cmpl-1",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(), finish_reason="tool_calls"
                    )
                ],
                created=1,
                model="test-model",
                object="chat.completion.chunk",
            )
        else:
            for piece in ("final ", "answer"):
                yield ChatCompletionChunk(
                    id="cmpl-2",
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(content=piece),
                            finish_reason=None,
                        )
                    ],
                    created=1,
                    model="test-model",
                    object="chat.completion.chunk",
                )
            yield ChatCompletionChunk(
                id="cmpl-2",
                choices=[
                    ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")
                ],
                created=1,
                model="test-model",
                object="chat.completion.chunk",
            )

    def _nonstream(self, is_first_call):
        if is_first_call:
            message = ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id="call_1",
                        type="function",
                        function=Function(
                            name="get_image", arguments='{"path": "x.png"}'
                        ),
                    )
                ],
            )
        else:
            message = ChatCompletionMessage(role="assistant", content="final answer")
        return ChatCompletion(
            id="cmpl-x",
            choices=[Choice(index=0, message=message, finish_reason="stop")],
            created=1,
            model="test-model",
            object="chat.completion",
        )


def _make_agent():
    return Agent(
        model="test-model",
        base_url="http://localhost:9/v1",
        api_key="test-key",
        store=InMemoryChatStorage(),
    )


def _patch_client(agent, fake):
    agent.client.chat.completions = SimpleNamespace(create=fake.create)


class TestAgentImageToolFlow(unittest.TestCase):
    """Full generate() flow with a mocked client: the image a tool returns
    must reach the second API call as an ``image_url`` content part."""

    def _second_call_tool_content(self, fake):
        self.assertEqual(len(fake.calls), 2)
        second_messages = fake.calls[1]["messages"]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        return tool_msgs[0]["content"]

    def test_streaming_tool_call_sends_image_to_model(self):
        agent = _make_agent()
        fake = _FakeCompletions("stream")
        _patch_client(agent, fake)

        events = list(
            agent.generate(
                message="show me the image",
                session_id="img-session",
                stream=True,
                tools=[get_image],
            )
        )

        self._second_call_tool_content(fake)
        content = self._second_call_tool_content(fake)
        self.assertIsInstance(content, list)
        self.assertEqual(
            content[0],
            {"type": "text", "text": json.dumps({"caption": "a red pixel"})},
        )
        self.assertEqual(
            content[1],
            {"type": "image_url", "image_url": {"url": TINY_PNG_URI}},
        )

        # the tool call was announced and the final answer streamed
        self.assertTrue(events[0].get("tool_call"))
        self.assertEqual(events[-1], {"role": "assistant", "partial_content": "answer"})

        # the multimodal tool message was persisted in the store
        stored = agent.store.get_all("img-session")
        stored_tool = [m for m in stored if m.get("role") == "tool"]
        self.assertEqual(stored_tool[0]["content"], content)
        self.assertEqual(stored[-1], {"role": "assistant", "content": "final answer"})

    def test_nonstream_tool_call_sends_image_to_model(self):
        agent = _make_agent()
        fake = _FakeCompletions("nonstream")
        _patch_client(agent, fake)

        # stream=False: the @streamable_response wrapper returns the single
        # yielded dict directly
        result = agent.generate(
            message="show me the image",
            session_id="img-session-2",
            stream=False,
            tools=[get_image],
        )

        content = self._second_call_tool_content(fake)
        self.assertIsInstance(content, list)
        self.assertEqual(content[1]["image_url"]["url"], TINY_PNG_URI)

        self.assertEqual(result["content"], "final answer")
        stored = agent.store.get_all("img-session-2")
        stored_tool = [m for m in stored if m.get("role") == "tool"]
        self.assertEqual(stored_tool[0]["content"], content)


class _TextContent:
    def __init__(self, text):
        self.text = text


class _ImageContent:
    def __init__(self, data, mime="image/png"):
        self.data = data
        self.mimeType = mime


class _EmbeddedResource:
    def __init__(self, uri):
        self.resource = SimpleNamespace(uri=uri)


class _CallToolResult:
    def __init__(self, content):
        self.content = content


class TestMCPFormatCallToolResult(unittest.TestCase):
    def test_text_only_returns_string(self):
        result = MCPClient.format_call_tool_result(
            _CallToolResult([_TextContent("a"), _TextContent("b")])
        )
        self.assertEqual(result, "a\nb")

    def test_image_returns_dict_with_data_uri(self):
        result = MCPClient.format_call_tool_result(
            _CallToolResult([_ImageContent(TINY_PNG_B64)])
        )
        self.assertEqual(result, {"image": TINY_PNG_URI})

    def test_text_and_image(self):
        result = MCPClient.format_call_tool_result(
            _CallToolResult(
                [_TextContent("caption"), _ImageContent(TINY_PNG_B64, "image/jpeg")]
            )
        )
        self.assertEqual(result["text"], "caption")
        self.assertTrue(result["image"].startswith("data:image/jpeg;base64,"))

    def test_multiple_images(self):
        result = MCPClient.format_call_tool_result(
            _CallToolResult([_ImageContent(TINY_PNG_B64), _ImageContent(TINY_PNG_B64)])
        )
        self.assertEqual(result["image"], [TINY_PNG_URI, TINY_PNG_URI])

    def test_resource_only(self):
        result = MCPClient.format_call_tool_result(
            _CallToolResult([_EmbeddedResource("file://x")])
        )
        self.assertEqual(result, "[Resource: file://x]")

    def test_image_dict_flows_into_agent_content_parts(self):
        agent = _make_agent()
        formatted = MCPClient.format_call_tool_result(
            _CallToolResult([_TextContent("hi"), _ImageContent(TINY_PNG_B64)])
        )
        parts = agent._build_tool_message_content(formatted, "mcp_tool")
        self.assertEqual(parts[0]["text"], json.dumps({"text": "hi"}))
        self.assertEqual(
            parts[1], {"type": "image_url", "image_url": {"url": TINY_PNG_URI}}
        )


class _RequestAborted(Exception):
    pass


class TestParamsBridge(unittest.TestCase):
    """openai>=3 strictly validates create() kwargs; provider-specific
    params must be routed into extra_body to reach the server."""

    def _first_call_kwargs(self, fake, params):
        agent = _make_agent()
        _patch_client(agent, fake)
        # omitting `stream` returns the full generator (generate's signature
        # default); consume it so the API calls actually happen
        list(
            agent.generate(
                message="hi",
                session_id="params-session",
                tools=[get_image],
                params=params,
            )
        )
        return fake.calls[0]

    def test_known_params_pass_through(self):
        fake = _FakeCompletions("stream")
        kwargs = self._first_call_kwargs(fake, {"temperature": 0.4})
        self.assertEqual(kwargs.get("temperature"), 0.4)
        # no extra_body built when all params are known create() kwargs
        self.assertIsNone(kwargs.get("extra_body"))

    def test_unknown_params_routed_to_extra_body(self):
        fake = _FakeCompletions("stream")
        kwargs = self._first_call_kwargs(
            fake, {"chat_template_kwargs": {"reasoning_effort": "low"}}
        )
        self.assertNotIn("chat_template_kwargs", kwargs)
        self.assertEqual(
            kwargs.get("extra_body"),
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
        )

    def test_mixed_params_merge_into_existing_extra_body(self):
        fake = _FakeCompletions("stream")
        kwargs = self._first_call_kwargs(
            fake,
            {
                "temperature": 0.2,
                "extra_body": {"existing": 1},
                "server_only": "x",
            },
        )
        self.assertEqual(kwargs.get("temperature"), 0.2)
        self.assertEqual(kwargs.get("extra_body"), {"existing": 1, "server_only": "x"})


class TestStreamableResponseDefault(unittest.TestCase):
    """streamable_response must honor the wrapped function's stream default.

    Agent.generate declares stream=True as its signature default, so a call
    that omits `stream` must return the full generator (not just the first
    event, which would silently drop the rest of the response).
    """

    def test_generate_without_stream_kwarg_returns_full_generator(self):
        agent = _make_agent()
        fake = _FakeCompletions("stream")
        _patch_client(agent, fake)

        events = agent.generate(
            message="hi", session_id="stream-default", tools=[get_image]
        )

        self.assertIsInstance(events, Generator)
        event_list = list(events)
        # full stream: the tool_call event AND the final partial content
        self.assertGreater(len(event_list), 1)
        self.assertTrue(any(e.get("tool_call") for e in event_list))
        self.assertTrue(
            any(
                e.get("role") == "assistant" and e.get("partial_content")
                for e in event_list
            )
        )

    def test_generate_explicit_stream_false_returns_first_value(self):
        agent = _make_agent()
        fake = _FakeCompletions("nonstream")
        _patch_client(agent, fake)

        result = agent.generate(
            message="hi",
            session_id="stream-false",
            stream=False,
            tools=[get_image],
        )

        # non-stream contract: a single dict (the first yielded event)
        self.assertIsInstance(result, dict)


class _FakeTransport:
    """Captures requests at the SDK transport layer, proving that fastllm's
    exact call pattern (including ``tools=None``) serializes cleanly through
    the openai 3.x SDK."""

    def __init__(self):
        self.requests = []

    def build_request(self, *args, **kwargs):
        request = httpx2.Request(
            kwargs["method"],
            kwargs["url"],
            content=kwargs.get("content"),
            headers=kwargs.get("headers"),
        )
        self.requests.append(request)
        return request

    def send(self, request, **kwargs):
        raise _RequestAborted()


class TestOpenAISDKUpgrade(unittest.TestCase):
    def test_openai_sdk_version_is_3_x(self):
        self.assertTrue(
            openai.__version__.startswith("3."),
            f"openai version is {openai.__version__}",
        )

    def test_core_api_surface(self):
        client = openai.OpenAI(base_url="http://localhost:9/v1", api_key="k")
        self.assertTrue(callable(client.chat.completions.create))
        self.assertTrue(issubclass(openai.NotFoundError, Exception))
        self.assertTrue(issubclass(openai.APIConnectionError, Exception))

    def test_agent_request_construction_with_tools_none(self):
        agent = _make_agent()
        agent.client.max_retries = 0
        transport = _FakeTransport()

        with (
            mock.patch.object(agent.client, "_client", transport),
            self.assertRaises(EmptyPayload),
        ):
            list(agent.generate(message="hi", session_id="req", stream=False))

        self.assertEqual(len(transport.requests), 1)
        body = json.loads(transport.requests[0].content)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"][0]["role"], "system")
        # the agentic loop omits `tools` entirely when the agent has no
        # tools (cleaner than the legacy explicit null)
        self.assertNotIn("tools", body)


if __name__ == "__main__":
    unittest.main()
