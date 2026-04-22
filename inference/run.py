"""End-to-end inference: real Jarvis MCP + trained FunctionGemma.

Flow per user turn:
    1. Render prompt with HF tokenizer + MCP tool schemas.
    2. Ask Ollama (or HF transformers) to generate.
    3. Extract `<start_function_call>...<end_function_call>` calls.
    4. If there are calls: invoke them on the real Jarvis MCP server, append
       the `tool` role responses to the conversation, loop back to step 1.
    5. If there are no calls: that generation is the final answer — print it
       and return to the REPL.

Multi-turn ceiling: `--max-iters` caps the tool-call → generate loop to
prevent runaways. Default 6, which comfortably covers chain_first (up to 5
calls) plus one summary turn.

Usage:
    uv run inference/run.py --model jarvis-v7 --prompt "List all pipelines"
    uv run inference/run.py --model jarvis-v7 --repl         # interactive
    uv run inference/run.py --model jarvis-v7 --dry-run      # stub MCP (test harness)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# support both `python -m inference.run` and `python inference/run.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mcp_client import JarvisMCP
    from ollama_backend import OllamaBackend
    from render_and_parse import (
        SYSTEM_PROMPT,
        mcp_tools_to_hf_schema,
        render_prompt,
        split_think_and_calls,
        append_tool_call,
        append_tool_result,
        initial_messages,
    )
else:
    from .mcp_client import JarvisMCP
    from .ollama_backend import OllamaBackend
    from .render_and_parse import (
        SYSTEM_PROMPT,
        mcp_tools_to_hf_schema,
        render_prompt,
        split_think_and_calls,
        append_tool_call,
        append_tool_result,
        initial_messages,
    )


class StubMCP:
    """Stand-in for Jarvis MCP when `--dry-run`. Returns a canned shape."""

    def list_tools(self):
        return [
            {
                "name": "jm_list_pipelines",
                "description": "List all Jarvis pipelines",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "create_pipeline",
                "description": "Create a new pipeline",
                "input_schema": {
                    "type": "object",
                    "properties": {"pipeline_id": {"type": "string"}},
                    "required": ["pipeline_id"],
                },
            },
        ]

    def call_tool(self, name, arguments=None):
        return json.dumps({"stubbed": True, "tool": name, "args": arguments or {}})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def run_once(
    tokenizer,
    backend,
    mcp,
    user_prompt: str,
    max_iters: int = 6,
    verbose: bool = True,
) -> str:
    tools_raw = mcp.list_tools()
    tools = mcp_tools_to_hf_schema(tools_raw)
    messages = initial_messages(user_prompt)

    final_text = ""
    for step in range(max_iters):
        rendered = render_prompt(tokenizer, messages, tools)
        if verbose:
            print(f"\n--- step {step + 1} / prompt rendered ({len(rendered)} chars) ---")
        raw = backend.generate(rendered)
        if verbose:
            print(f"--- raw output ---\n{raw}\n--- end raw ---")

        thought, calls, trailing = split_think_and_calls(raw)

        if not calls:
            final_text = trailing or raw.strip()
            messages.append({"role": "assistant", "content": raw})
            break

        if verbose:
            print(f"[step {step + 1}] thought: {thought[:120]}")
            for c in calls:
                print(f"    call: {c['name']}({c['arguments']})")

        append_tool_call(messages, thought, calls)
        for call in calls:
            result = mcp.call_tool(call["name"], call["arguments"])
            if verbose:
                preview = result if len(result) < 200 else result[:200] + "…"
                print(f"    ← {call['name']}: {preview}")
            append_tool_result(messages, call["name"], result)

    else:
        final_text = (
            f"(stopped after {max_iters} tool-call iterations without a final answer)"
        )

    return final_text


def build_parser():
    p = argparse.ArgumentParser(description="FunctionGemma + Jarvis MCP inference")
    p.add_argument("--model", default="jarvis-v6-official",
                   help="Ollama model tag (default: jarvis-v6-official)")
    p.add_argument("--tokenizer", default="unsloth/functiongemma-270m-it",
                   help="HF tokenizer for prompt rendering")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--prompt", default=None, help="One-shot user prompt")
    p.add_argument("--repl", action="store_true", help="Interactive REPL")
    p.add_argument("--dry-run", action="store_true",
                   help="Use stub MCP instead of starting the real Jarvis server")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer  # deferred import
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    backend = OllamaBackend(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    mcp_cls = StubMCP if args.dry_run else JarvisMCP
    verbose = not args.quiet

    with mcp_cls() as mcp:
        if args.prompt:
            answer = run_once(tokenizer, backend, mcp, args.prompt,
                              max_iters=args.max_iters, verbose=verbose)
            print("\n=== answer ===")
            print(answer)
            return

        if args.repl:
            print("Jarvis FunctionGemma REPL. Ctrl-C or 'quit' to exit.")
            while True:
                try:
                    user = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user or user.lower() in ("quit", "exit"):
                    break
                try:
                    answer = run_once(tokenizer, backend, mcp, user,
                                      max_iters=args.max_iters, verbose=verbose)
                    print("\n=== answer ===")
                    print(answer)
                except Exception as e:
                    print(f"ERROR: {e}")
            return

        print("Provide --prompt or --repl", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
