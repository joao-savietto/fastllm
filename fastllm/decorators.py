"""FastLLM decorator utilities.

This module provides helper decorators and functions used throughout the FastLLM library. The primary purpose is to expose a convenient way for user code to declare OpenAI function calls via ``@tool`` while automatically handling schema conversion, threading helpers, retry logic, and streaming response adaptation.
"""

import base64
import inspect
import json
import threading
import time
import traceback
from collections.abc import Callable, Generator
from functools import wraps
from typing import Any

import openai

from fastllm.exceptions import EmptyPayload


def build_image_data_uri(data: Any, mime: str = "image/png") -> str:
    """Build an ``image_url``-compatible data URI from image data.

    Parameters
    ----------
    data : bytes or str
        Raw image bytes (or ``bytearray``), a plain base64 string, an
        already-formed ``data:`` URI, or an ``http(s)://`` URL.
    mime : str, optional
        MIME type used when encoding raw bytes or bare base64 strings.
        Defaults to ``image/png``.

    Returns
    -------
    str
        A data URI (or the original URL) usable in an OpenAI-compatible
        ``image_url`` content part.

    """
    if isinstance(data, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(data)).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    if isinstance(data, str):
        if data.startswith(("data:", "http://", "https://")):
            return data
        return f"data:{mime};base64,{data}"
    raise TypeError(f"Unsupported image data type: {type(data).__name__}")


def serialize_tool_result(result: Any) -> str:
    """Serialize a tool result to a JSON string, encoding image data.

    Tools may return image data as the whole result (``bytes``/``bytearray``)
    or under the ``image`` key (``bytes``/``bytearray``, base64 string, data
    URI, or URL).  Binary values are encoded as ``data:<mime>;base64,``
    strings so the payload stays JSON-safe; :class:`fastllm.agent.Agent`
    turns such payloads back into multimodal tool messages so the images are
    actually sent to the model.

    """
    if isinstance(result, (bytes, bytearray)):
        return json.dumps({"image": build_image_data_uri(bytes(result))})
    if isinstance(result, dict) and any(
        isinstance(value, (bytes, bytearray)) for value in result.values()
    ):
        result = {
            key: (
                build_image_data_uri(bytes(value))
                if isinstance(value, (bytes, bytearray))
                else value
            )
            for key, value in result.items()
        }
    return json.dumps(result)


def tool(description: str, pydantic_model: type):
    """Decorator that registers a function as an OpenAI tool.

    Parameters
    ----------
    description : str
        Human‑readable description for the function – shown in the function call metadata.
    pydantic_model : type
        A Pydantic ``BaseModel`` subclass describing the expected arguments.  The schema is automatically converted into the JSON format required by the OpenAI API.

    Returns
    -------
    Callable
        The original function wrapped with two additional attributes:
        ``tool_json`` – returns the OpenAI *function* schema and ``execute`` – serialises the call arguments, invokes the original function, and returns a JSON string.

    """

    def decorator(func):
        # Convert the Pydantic model schema to OpenAI parameters format
        openapi_parameters = pydantic_to_openai_schema(pydantic_model)
        openai_format_schema = {
            "name": func.__name__,
            "description": description,
            "parameters": openapi_parameters,
        }

        def tool_json():
            schema = {"type": "function", "function": openai_format_schema}
            return schema

        def execute(*args, **kwargs):
            if args:
                kwargs.update(args[0])

            model = pydantic_model(**kwargs)
            result = func(model)
            return serialize_tool_result(result)

        func.tool_json = tool_json
        func.execute = execute
        return func

    return decorator


def run_in_thread(func):
    """Run ``func`` asynchronously in a new thread.

    The wrapper starts the thread immediately and returns nothing; it is intended for fire‑and‑forget side effects such as logging or background cleanup.

    """

    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()

    return wrapper


def pydantic_to_openai_schema(pydantic_model: type) -> dict:
    """Convert a Pydantic model into OpenAI function‑parameter schema.

    The conversion handles nested references (``$ref``) and array items.  It produces a dictionary compatible with ``openai.FunctionSchema`` used in tool calls.

    """
    # Get pydantic schema including definitions
    pydantic_schema = pydantic_model.model_json_schema()

    def resolve_reference(ref_dict, all_defs):
        """Recursively resolve a reference to its actual definition"""
        if not isinstance(ref_dict, dict) or "$ref" not in ref_dict:
            return ref_dict

        ref_path = ref_dict["$ref"]
        # Extract the definition name from path like '#/$defs/ProductReview'
        ref_name = ref_path.split("/")[-1]

        if ref_name in all_defs:
            # Recursively resolve the reference
            resolved_def = resolve_reference(all_defs[ref_name], all_defs)
            return resolved_def

        return ref_dict

    def convert_property_details(prop_details, all_defs):
        """Convert property details to OpenAI format handling references properly"""

        # Handle direct object references (e.g., field type is a referenced model)
        if "$ref" in prop_details:
            try:
                resolved_schema = resolve_reference(prop_details, all_defs)

                result = {
                    "type": resolved_schema.get("type", "object"),
                    "description": prop_details.get("description", ""),
                }

                # If it's a nested object with properties
                if "properties" in resolved_schema:
                    result["properties"] = {}
                    for (
                        inner_prop_name,
                        inner_prop_details,
                    ) in resolved_schema["properties"].items():
                        result["properties"][inner_prop_name] = {
                            "type": inner_prop_details.get("type", "string"),
                            "description": inner_prop_details.get("description", ""),
                        }
                return result
            except Exception:
                # Fallback for resolution errors
                return {
                    "type": "object",
                    "description": prop_details.get("description", ""),
                }

        # Handle array items with references (the main problem case)
        elif (
            "items" in prop_details
            and isinstance(prop_details["items"], dict)
            and "$ref" in prop_details["items"]
        ):
            try:
                resolved_items = resolve_reference(prop_details["items"], all_defs)

                result = {
                    "type": "array",
                    "description": prop_details.get("description", ""),
                }

                # Handle the items properly based on their resolved type
                if (
                    "properties" in resolved_items
                    and resolved_items.get("type") == "object"
                ):
                    # Nested object in array - preserve all properties
                    result["items"] = {"type": "object", "properties": {}}

                    for inner_prop_name, inner_prop_details in resolved_items[
                        "properties"
                    ].items():
                        result["items"]["properties"][inner_prop_name] = {
                            "type": inner_prop_details.get("type", "string"),
                            "description": inner_prop_details.get("description", ""),
                        }
                else:
                    # Simple type or primitive
                    result["items"] = {"type": resolved_items.get("type", "string")}

                return result
            except Exception:
                # Fallback for resolution errors
                return {
                    "type": "array",
                    "description": prop_details.get("description", ""),
                    "items": {"type": "object"},
                }

        # Handle regular properties
        else:
            try:
                result = {
                    "type": prop_details.get("type", "string"),
                    "description": prop_details.get("description", ""),
                }

                # For complex nested objects (non-references)
                if "properties" in prop_details and prop_details["type"] == "object":
                    result["properties"] = {}
                    for inner_prop_name, inner_prop_details in prop_details[
                        "properties"
                    ].items():
                        # Recursive handling of nested properties
                        result["properties"][inner_prop_name] = (
                            convert_property_details(inner_prop_details, all_defs)
                        )

                return result
            except Exception:
                # Fallback
                return {
                    "type": "object",
                    "description": prop_details.get("description", ""),
                }

    # Main conversion logic
    defs = pydantic_schema.get("$defs", {})

    openai_format_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Convert all properties with proper reference resolution
    for prop_name, prop_details in pydantic_schema["properties"].items():
        converted_prop = convert_property_details(prop_details, defs)
        openai_format_schema["properties"][prop_name] = converted_prop

    # Add required fields
    if "required" in pydantic_schema:
        openai_format_schema["required"] = pydantic_schema["required"]

    return openai_format_schema


def retry(max_attempts=5, delay=2):
    """Retry decorator for transient OpenAI errors.

    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts before giving up.  Defaults to 5.
    delay : int, optional
        Seconds to wait between attempts.  Defaults to 2.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except openai.NotFoundError:
                    attempts += 1
                    print(
                        f"Attempt {attempts} failed with 404. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                except Exception as err:
                    raise Exception(
                        f"Function {func.__name__} failed after {max_attempts} attempts."
                    ) from err

        return wrapper

    return decorator


def streamable_response(
    func: Callable[..., Generator],
) -> Callable[..., Any]:
    """Decorator that adapts a generator to a simple response interface.

    The wrapped function can be called with ``stream=True`` to receive the raw generator, or with ``stream=False`` to get only the first yielded value.  When ``stream`` is omitted, the wrapped function's own signature default is used (e.g. ``Agent.generate`` defaults to ``stream=True``).  When the underlying function returns a plain dict, it is passed through unchanged.

    """
    try:
        stream_default = inspect.signature(func).parameters["stream"].default
        if stream_default is inspect.Parameter.empty:
            stream_default = False
    except (KeyError, ValueError):
        stream_default = False

    def wrapper(*args, **kwargs):
        stream = kwargs.get("stream", stream_default)
        gen = func(*args, **kwargs)
        if isinstance(gen, dict):
            return gen
        if not stream:
            try:
                # Get the first (and only) value from generator
                return next(gen)
            except StopIteration as err:
                print(traceback.format_exc())
                raise EmptyPayload("No response generated") from err
        else:
            return gen

    return wrapper
