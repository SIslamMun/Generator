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


# same regex FunctionGemma's multi-turn inference notebook uses
_CALL_RE = re.compile(
    r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", re.DOTALL
)
_ARG_RE = re.compile(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))")


def _cast(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value.strip("'\"")


def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for name, args_blob in _CALL_RE.findall(text):
        arguments: dict[str, Any] = {}
        for key, v_esc, v_plain in _ARG_RE.findall(args_blob):
            raw = v_esc if v_esc or v_esc == "" else v_plain
            arguments[key] = _cast(raw.strip())
        calls.append({"name": name, "arguments": arguments})
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
