"""Tool-call healing utilities.

Some models (especially smaller local ones served via llama.cpp, vLLM, Ollama,
etc.) frequently emit tool calls as *plain text* inside the assistant ``content``
or ``reasoning_content`` field instead of using the structured ``tool_calls``
field of the OpenAI API. When that happens the agent loop never sees a tool call
and stops early.

This module recovers ("heals") those malformed tool calls by scanning the text
for the most common encodings and converting them back into the canonical
OpenAI tool-call structure::

    {
        "id": "...",
        "type": "function",
        "function": {"name": "...", "arguments": "<json string>"},
    }

Only calls whose function name is a *known* tool (present in ``tool_map``) are
recovered, to avoid turning ordinary prose into spurious tool calls.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

__all__ = ["heal_tool_calls", "strip_tool_call_markup"]


def _new_id() -> str:
    """Generate a tool-call id compatible with OpenAI-style ids."""
    return "call_" + uuid.uuid4().hex[:24]


def _try_json(text: str) -> Optional[Any]:
    """Best-effort JSON parse. Returns None on failure."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _make_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    """Build a canonical OpenAI tool-call dict from a name and arguments."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": _new_id(),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_xml_parameters(block: str) -> Dict[str, Any]:
    """Parse ``<parameter=key>value</parameter>`` pairs into a dict.

    Values that look like JSON (numbers, booleans, objects, arrays, quoted
    strings) are coerced to the corresponding Python type; otherwise the raw
    string is kept.
    """
    args: Dict[str, Any] = {}
    for m in re.finditer(
        r"<parameter\s*=\s*([^>]+?)\s*>(.*?)</parameter>", block, re.DOTALL
    ):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        parsed = _try_json(raw)
        args[key] = parsed if parsed is not None else raw
    return args


def _parse_function_blocks(
    text: str, valid_names: set
) -> List[Dict[str, Any]]:
    """Parse Qwen/Hermes XML ``<function=NAME> ... </function>`` blocks.

    The body may contain a JSON object of arguments, a set of
    ``<parameter=...>`` pairs, or nothing at all (no-arg call).
    """
    out: List[Dict[str, Any]] = []
    for m in re.finditer(
        r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", text, re.DOTALL
    ):
        name = m.group(1).strip()
        if name not in valid_names:
            continue
        body = m.group(2).strip()
        args: Any = {}
        if body:
            parsed = _try_json(body)
            if isinstance(parsed, dict):
                args = parsed
            elif "<parameter" in body:
                args = _parse_xml_parameters(body)
        out.append(_make_tool_call(name, args))
    return out


def _extract_call_from_obj(
    obj: Any, valid_names: set
) -> Optional[Dict[str, Any]]:
    """Build a tool call from a parsed dict like {"name":.., "arguments":..}."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    if isinstance(name, dict):  # nested {"function": {"name": ..}}
        name = name.get("name")
    if not name or name not in valid_names:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", {})
    return _make_tool_call(name, args)


def heal_tool_calls(
    content: Optional[str],
    reasoning_content: Optional[str],
    tool_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Recover text-encoded tool calls from model output.

    Args:
        content: The assistant message ``content`` field.
        reasoning_content: The assistant ``reasoning_content`` field (if the
            server separates chain-of-thought from the answer). Tool calls
            leaked into reasoning are recovered too.
        tool_map: Mapping of known tool name -> tool. Only these names are
            recovered.

    Returns:
        A list of canonical OpenAI tool-call dicts (possibly empty).
    """
    if not tool_map:
        return []

    text = "\n".join(p for p in (content, reasoning_content) if p)
    if not text:
        return []

    # Cheap early-out: nothing that looks like a textual tool call.
    if "<tool_call" not in text and "<function" not in text and \
            '"name"' not in text:
        return []

    valid_names = set(tool_map.keys())
    healed: List[Dict[str, Any]] = []

    # 1) Hermes-style: <tool_call> ... </tool_call>
    tool_call_blocks = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
    )
    for inner in tool_call_blocks:
        parsed = _try_json(inner)
        call = _extract_call_from_obj(parsed, valid_names)
        if call:
            healed.append(call)
            continue
        # JSON might be wrapped/concatenated; try to find an object in it.
        obj_match = re.search(r"\{.*\}", inner, re.DOTALL)
        if obj_match:
            call = _extract_call_from_obj(
                _try_json(obj_match.group(0)), valid_names
            )
            if call:
                healed.append(call)
                continue
        # Otherwise it may contain an XML <function=..> block.
        healed.extend(_parse_function_blocks(inner, valid_names))

    # 2) Bare Qwen XML outside <tool_call> wrappers.
    if not healed:
        healed.extend(_parse_function_blocks(text, valid_names))

    # 3) Fenced JSON code block: ```json {"name":.., "arguments":..} ```
    if not healed:
        for m in re.finditer(
            r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", text, re.DOTALL
        ):
            call = _extract_call_from_obj(
                _try_json(m.group(1)), valid_names
            )
            if call:
                healed.append(call)

    return healed


def strip_tool_call_markup(text: Optional[str]) -> str:
    """Remove text-based tool-call markup so it is not shown to the user."""
    if not text:
        return ""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<function\s*=\s*[^>]+>.*?</function>", "", text, flags=re.DOTALL
    )
    return text.strip()
