"""Prompt rendering + tool-call extraction for FunctionGemma 270M.

- `render_prompt()`: uses HuggingFace `tokenizer.apply_chat_template(...)` so the
  byte-sequence matches training exactly. Any drift here (e.g. Ollama's Go
  template) is catastrophic for a 270M model.
- `extract_tool_calls()`: regex-based parser for the custom
  `<start_function_call>call:{name}{k:v,...}<end_function_call>` format the
  model emits. Lifted verbatim from
  `FunctionGemma_(270M)_Multi_Turn_Tool_Calling.ipynb`.

The helpers in this module are backend-agnostic: they don't care whether the
raw generation comes from Ollama, llama.cpp, or HuggingFace transformers.
"""

from __future__ import annotations

import json
import re
from typing import Any


# FunctionGemma tool-call grammar (recursive — supports nested dict/list args):
#   call        := `<start_function_call>call:` NAME `{` pairs `}` `<end_function_call>`
#   pairs       := (KEY `:` value (`,` KEY `:` value)*)?
#   value       := escape_string | object | array | number | bool | null | bareword
#   escape_str  := `<escape>` ... `<escape>`
#   object      := `{` pairs `}`
#   array       := `[` value (`,` value)* `]`
_CALL_BOUNDARY = re.compile(r"<start_function_call>call:(\w+)(\{)")


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _parse_escape_string(s: str, i: int) -> tuple[str, int]:
    """i points at first char after opening <escape>. Returns (text, next_i)."""
    end = s.find("<escape>", i)
    if end < 0:
        return s[i:], len(s)
    return s[i:end], end + len("<escape>")


def _parse_value(s: str, i: int) -> tuple[Any, int]:
    i = _skip_ws(s, i)
    if s.startswith("<escape>", i):
        return _parse_escape_string(s, i + len("<escape>"))
    c = s[i]
    if c == "{":
        return _parse_object(s, i)
    if c == "[":
        return _parse_array(s, i)
    # bareword: up to ,  }  ] (respecting nested escape-strings isn't needed here)
    j = i
    while j < len(s) and s[j] not in ",}]":
        j += 1
    raw = s[i:j].strip()
    if raw == "":
        return "", j
    if raw.lower() == "true":
        return True, j
    if raw.lower() == "false":
        return False, j
    if raw.lower() in ("null", "none"):
        return None, j
    try:
        return int(raw), j
    except ValueError:
        pass
    try:
        return float(raw), j
    except ValueError:
        pass
    return raw.strip("'\""), j


def _parse_key(s: str, i: int) -> tuple[str, int]:
    i = _skip_ws(s, i)
    j = i
    while j < len(s) and (s[j].isalnum() or s[j] == "_"):
        j += 1
    return s[i:j], j


def _parse_object(s: str, i: int) -> tuple[dict, int]:
    # s[i] == '{'
    i += 1
    out: dict[str, Any] = {}
    while True:
        i = _skip_ws(s, i)
        if i >= len(s):
            return out, len(s)
        if s[i] == "}":
            return out, i + 1
        key, i = _parse_key(s, i)
        i = _skip_ws(s, i)
        if i >= len(s) or s[i] != ":":
            # malformed; bail
            return out, i
        i += 1
        value, i = _parse_value(s, i)
        out[key] = value
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i += 1
            continue
        if i < len(s) and s[i] == "}":
            return out, i + 1
        return out, i


def _parse_array(s: str, i: int) -> tuple[list, int]:
    # s[i] == '['
    i += 1
    out: list[Any] = []
    while True:
        i = _skip_ws(s, i)
        if i >= len(s):
            return out, len(s)
        if s[i] == "]":
            return out, i + 1
        value, i = _parse_value(s, i)
        out.append(value)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i += 1
            continue
        if i < len(s) and s[i] == "]":
            return out, i + 1
        return out, i


def extract_tool_calls(text: str) -> list[dict]:
    """Parse every `<start_function_call>…<end_function_call>` block.

    Handles nested dicts (e.g. `extra_args:{nprocs:64}`), lists, numbers,
    bools, and `<escape>`-wrapped strings. Stops each call at the matching
    `}<end_function_call>` by tracking brace depth so nested `}` don't
    prematurely terminate the call.
    """
    calls: list[dict] = []
    for m in _CALL_BOUNDARY.finditer(text):
        name = m.group(1)
        start = m.end() - 1  # position of the opening `{`
        # Skip escape-wrapped strings so their `{` / `}` don't count toward depth.
        depth = 0
        i = start
        end = None
        while i < len(text):
            if text.startswith("<escape>", i):
                nxt = text.find("<escape>", i + len("<escape>"))
                if nxt < 0:
                    i = len(text)
                    break
                i = nxt + len("<escape>")
                continue
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end is None:
            continue
        obj, _ = _parse_object(text, start)
        calls.append({"name": name, "arguments": obj})
    return calls


def mcp_tools_to_hf_schema(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP `tools/list` entries into HF chat-template shape."""
    out = []
    for t in mcp_tools:
        schema = t.get("input_schema") or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        # normalise — HF template expects `parameters`
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return out


def render_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    """Render with `add_generation_prompt=True`. Strips the `<bos>` that
    Unsloth's training loop also strips (`removeprefix('<bos>')`)."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    ).removeprefix("<bos>")


SYSTEM_PROMPT = (
    "You are a Jarvis-CD HPC workflow assistant. Use the provided tools to "
    "create and manage pipelines, attach and configure packages, and operate "
    "the JarvisManager. Think briefly before each tool call. Call one tool "
    "at a time unless the user asks for multiple actions."
)


def initial_messages(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def append_tool_call(
    messages: list[dict], thought: str, calls: list[dict]
) -> list[dict]:
    """Append an assistant turn with `think` + tool_calls, matching training shape."""
    messages.append(
        {
            "role": "assistant",
            "content": f"<think>{thought}</think>" if thought else "",
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}_{i}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for i, call in enumerate(calls)
            ],
        }
    )
    return messages


def append_tool_result(messages: list[dict], name: str, content: str) -> list[dict]:
    messages.append({"role": "tool", "name": name, "content": content})
    return messages


def split_think_and_calls(raw_output: str) -> tuple[str, list[dict], str]:
    """Parse raw model output into (thought, calls, remainder).

    `remainder` is whatever natural-language text sits after the last
    `<end_function_call>` — useful for the final-summary turn.
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""
    calls = extract_tool_calls(raw_output)
    # strip think + all tool-call tags to get trailing natural-language text
    trailing = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    trailing = re.sub(
        r"<start_function_call>.*?<end_function_call>", "", trailing, flags=re.DOTALL
    )
    trailing = re.sub(
        r"<start_function_response>.*?<end_function_response>",
        "",
        trailing,
        flags=re.DOTALL,
    )
    return thought, calls, trailing.strip()
