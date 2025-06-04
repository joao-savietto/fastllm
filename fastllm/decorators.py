import json
import threading
import time
from functools import wraps

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

            if details.get("required", True):
                openai_format_schema["parameters"]["required"].append(param)

        def tool_json():
            schema = {"type": "function", "function": openai_format_schema}
            return schema

        @wraps(func)
        def execute(**kwargs):
            # Validate and construct the parameters using the Pydantic model
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
