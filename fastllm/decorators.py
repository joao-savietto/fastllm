import json
import threading
import time
from fastllm.exceptions import EmptyPayload
from functools import wraps
from typing import Generator, Any, Callable
import traceback

import openai


def tool(description: str, pydantic_model: type):
    def decorator(func):
        # Generate the Pydantic model schema
        pydantic_schema = pydantic_model.schema()
        # Convert the Pydantic schema to the OpenAI API format
        openai_format_schema = {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

        for param, details in pydantic_schema["properties"].items():
            openai_format_schema["parameters"]["properties"][param] = {
                "type": details.get("type", "string"),
                "description": details.get("description", ""),
            }

            if details.get("required", True):
                openai_format_schema["parameters"]["required"].append(param)

        def tool_json():
            schema = {"type": "function", "function": openai_format_schema}
            return schema

        def execute(*args, **kwargs):
            if args:
                kwargs.update(args[0])

            model = pydantic_model(**kwargs)
            result = func(model)
            return json.dumps(result)

        func.tool_json = tool_json
        func.execute = execute
        return func

    return decorator


def run_in_thread(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()

    return wrapper


def retry(max_attempts=5, delay=2):
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
                except Exception:
                    raise Exception(
                        f"Function {func.__name__} failed after {max_attempts} attempts."
                    )

        return wrapper

    return decorator


def streamable_response(
    func: Callable[..., Generator],
) -> Callable[..., Any]:
    """
    Decorator to make a generator-returning function behave like:
      - Returns a generator when `stream=True`
      - Returns the first (and only) value when `stream=False`

    Useful for APIs where you want one interface
    that adapts based on stream flag.
    """

    def wrapper(*args, **kwargs):
        stream = kwargs.get("stream", False)
        gen = func(*args, **kwargs)
        if isinstance(gen, dict):
            return gen
        if not stream:
            try:
                # Get the first (and only) value from generator
                return next(gen)
            except StopIteration:
                print(traceback.format_exc())
                raise EmptyPayload("No response generated")
        else:
            return gen

    return wrapper
