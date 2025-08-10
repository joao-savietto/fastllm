# ----------------------------------------------------------------------
# Bash / terminal execution tool for FastLLM
# ----------------------------------------------------------------------

import subprocess
from typing import Optional

from fastllm import tool
from pydantic import BaseModel, Field, validator


class BashCommandModel(BaseModel):
    """Parameters accepted by the bash‑execution tool."""
    command: str = Field(..., description="The bash command to execute")
    cwd: Optional[str] = Field(
        None,
        description="Working directory in which to run the command (defaults to current process dir)",
    )
    timeout: int = Field(
        30,
        description="Maximum seconds to allow the command to run before terminating",
    )

    @validator("command")
    def _non_empty(cls, v):
        if not v.strip():
            raise ValueError("Command must not be empty")
        return v


@tool(
    description=(
        "Executes a bash/terminal command on the host and returns its output. "
        "Useful for file operations, system queries, or invoking other CLI tools."
    ),
    pydantic_model=BashCommandModel,
)
def run_bash(request: BashCommandModel):
    """Run a shell command safely and return its stdout / stderr."""
    try:
        result = subprocess.run(
            request.command,
            shell=True,
            cwd=request.cwd,
            capture_output=True,
            text=True,
            timeout=request.timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as exc:
        # The decorator will JSON‑encode this dict for the LLM
        return {"error": f"{type(exc).__name__}: {exc}"}
