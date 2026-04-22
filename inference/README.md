# Inference: FunctionGemma 270M + real Jarvis MCP

End-to-end inference harness for the fine-tuned FunctionGemma 270M model:

- **`mcp_client.py`** — stdio MCP client that speaks JSON-RPC to the real
  Jarvis MCP server (`jarvis-env/clio-kit-mcp-servers/jarvis/src/server.py`).
  Paginates `tools/list` so the full 29-tool catalog is returned.
- **`render_and_parse.py`** — prompt rendering with HF `apply_chat_template`
  (byte-parity with training) + the `<start_function_call>…<end_function_call>`
  regex from Unsloth's multi-turn inference notebook.
- **`ollama_backend.py`** — Ollama `/api/generate raw=true` with Unsloth's
  published sampling defaults (temperature 1.0, top_p 0.95, top_k 64).
- **`hf_backend.py`** — HF `transformers` backend, for Colab/GPU runs directly
  after training (no Ollama export required).
- **`run.py`** — CLI that wires MCP + backend + multi-turn tool-call loop.

## Prerequisites

- `ollama serve` running with the trained model registered (e.g.
  `jarvis-v6-official` or your freshly-exported `jarvis-v7`).
- `transformers` installed (the tokenizer only — no torch required for
  inference that uses the Ollama backend).
- The Jarvis MCP server is launched automatically by `mcp_client.py` via
  the `jarvis-mcp` console script from
  `/home/shazzadul/Illinois_Tech/Spring26/RA/clio-kit/clio-kit-mcp-servers/jarvis/.venv/`.
  Override via the `server_cmd` and `cwd` kwargs of `JarvisMCP(...)` if
  your paths differ.

## Quick start

One-shot prompt through the real MCP server:

```bash
cd /home/shazzadul/Illinois_Tech/Spring26/RA/Training/context/Generator
python inference/run.py --model jarvis-v6-official --prompt "List all my pipelines"
```

Interactive REPL:

```bash
python inference/run.py --model jarvis-v6-official --repl
```

Dry-run (stubbed MCP — useful for smoke-testing the model without touching
Jarvis):

```bash
python inference/run.py --model jarvis-v6-official --dry-run \
    --prompt "Create a pipeline called demo"
```

Quiet mode (only prints the final answer):

```bash
python inference/run.py --model jarvis-v6-official --quiet \
    --prompt "List all my pipelines"
```

## How a single turn executes

```
user prompt ─► render_prompt(tokenizer, messages, tools) ─► Ollama /api/generate raw=true
                                                                     │
                                                                     ▼
              split_think_and_calls(raw)  ─►  <think>…</think> + [{name, arguments}, …]
                                                                     │
                   ┌─────────────────────────────────────────────────┘
                   ▼
          for each call:  mcp.call_tool(name, arguments)  ─►  JSON result
                                                                     │
                                                                     ▼
              append to messages as role=tool ──► loop back to render_prompt
```

The loop terminates when the model produces a response with no tool calls —
that final generation is the natural-language answer shown to the user.

## Sampling knobs

Defaults match Unsloth's published FunctionGemma recipe. Override via:

```bash
python inference/run.py --model … --temperature 0.7 --top-p 0.9 --top-k 40
```

At 270M, straying far from `(1.0, 0.95, 64)` noticeably hurts structured-token
emission (see `docs/functiongemma_multi_tool_research.md`).

## Library use (inside the Colab notebook)

```python
from inference.hf_backend import HFBackend
from inference.render_and_parse import render_prompt, split_think_and_calls, initial_messages, mcp_tools_to_hf_schema
from inference.run import run_once, StubMCP

backend = HFBackend(model, tokenizer)
with StubMCP() as mcp:
    answer = run_once(tokenizer, backend, mcp, "Create a pipeline called demo")
print(answer)
```

Swap `StubMCP` for `JarvisMCP` on a machine that has the real Jarvis server
installed.
