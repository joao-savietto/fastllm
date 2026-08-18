"""FastLLM - a lightweight library for building LLM apps on top of any
OpenAI-compatible provider (OpenAI, Ollama, LM Studio, ...).
"""

from fastllm.reflection_agent import ReflectionAgent
from fastllm.store import (
    ChatStorageInterface,
    InMemoryChatStorage,
    JSONChatStorage,
    RedisChatStorage,
)

from .agent import Agent
from .decorators import (
    build_image_data_uri,
    pydantic_to_openai_schema,
    serialize_tool_result,
    tool,
)
from .knowledge_base import Chroma, FullTextSearchBase, KnowledgeBaseInterface
from .tools import (
    BashCommandModel,
    FileNameWithContent,
    FolderNameModel,
    HttpRequestModel,
    MoveModel,
    PathModel,
    create_file,
    create_folder,
    delete_file,
    delete_folder,
    find_files,
    http_request,
    move_file,
    move_folder,
    read_file,
    run_bash,
)
from .workflow import BooleanNode, Node

__all__ = [
    "Agent",
    "BashCommandModel",
    "BooleanNode",
    "ChatStorageInterface",
    "Chroma",
    "FileNameWithContent",
    "FolderNameModel",
    "FullTextSearchBase",
    "HttpRequestModel",
    "InMemoryChatStorage",
    "JSONChatStorage",
    "KnowledgeBaseInterface",
    "MoveModel",
    "Node",
    "PathModel",
    "RedisChatStorage",
    "ReflectionAgent",
    "build_image_data_uri",
    "create_file",
    "create_folder",
    "delete_file",
    "delete_folder",
    "find_files",
    "http_request",
    "move_file",
    "move_folder",
    "pydantic_to_openai_schema",
    "read_file",
    "run_bash",
    "serialize_tool_result",
    "tool",
]
