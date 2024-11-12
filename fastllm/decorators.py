import json
from functools import wraps


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

        for param, details in pydantic_schema['properties'].items():
            openai_format_schema["parameters"]['properties'][param] = {
                "type": details.get('type', 'string'),
                "description": details.get('description', '')
            }
            if param in pydantic_schema['required']:
                openai_format_schema["parameters"]['required'].append(param)

        def tool_json():
            schema = {
                "type": "function",
                "function": openai_format_schema
            }
            return schema

        @wraps(func)
        def execute(**kwargs):
            # Validate and construct the parameters using the Pydantic model
            parameters = pydantic_model(**kwargs)
            result = func(**parameters.dict())
            return json.dumps(result)

        func.tool_json = tool_json
        func.execute = execute
        return func

    return decorator
