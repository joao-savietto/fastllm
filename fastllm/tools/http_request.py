from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field, field_validator

from fastllm import tool


class HttpRequestModel(BaseModel):
    method: str = Field(
        ..., description="HTTP method (get, post, put, patch, delete)"
    )
    url: str = Field(..., description="URL to make the request to")
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP headers"
    )
    params: Optional[Dict[str, Any]] = Field(
        None, description="Query parameters"
    )
    body: Optional[Any] = Field(
        None, description="Request body (for POST/PUT/PATCH)"
    )

    @field_validator("method")
    def validate_method(cls, v):
        valid_methods = ["get", "post", "put", "patch", "delete"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v.lower()


@tool(
    description="Makes HTTP requests with customizable method, URL, headers, parameters and body",
    pydantic_model=HttpRequestModel,
)
def http_request(request: HttpRequestModel):
    try:
        kwargs = {}
        if request.headers:
            kwargs["headers"] = request.headers
        if request.params:
            kwargs["params"] = request.params
        if request.body is not None:
            if isinstance(request.body, (dict, list)):
                kwargs["json"] = request.body  # JSON payload
            else:
                kwargs["data"] = str(request.body)

        response = requests.request(
            method=request.method,
            url=request.url,
            **kwargs,
        )

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.text,
            "json": (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else None
            ),
        }

    except Exception as exc:
        # Return a dict that the decorator will turn into JSON:
        return {"error": str(exc)}
