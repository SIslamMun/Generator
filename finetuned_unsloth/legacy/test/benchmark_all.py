"""Paper-grade cross-model / cross-temperature benchmark on the real Jarvis MCP.

Design goals:
  * PROVIDER-PLURAL  — Ollama (FunctionGemma & OpenAI-tools), Anthropic, vLLM
                       all plug in via the Provider protocol. Adding a new
                       backend = one new class.
  * TEMPERATURE SWEEP — each provider runs every case at several temperatures.
                       The "sweet spot for tool-calling" is whatever T maxes
                       task_success; we surface it explicitly in the report.
  * VARIANCE-AWARE    — non-zero T runs each case N_REPEATS times (for CI).
                       Greedy (T=0.0) runs once.
  * FAIR TOOL GRAMMAR — each model uses its native format:
                        - FunctionGemma <start_function_call>…<end_function_call>
                        - OpenAI-style tool_calls via Ollama /api/chat
                        - Anthropic tool_use via Messages API + MCP connector
                       Backends differ, but all hit the SAME MCP server.
  * METRICS           — tool_ok, arg_ok, task_success, answer_correctness,
                        hallucination, mcp_errors, latency_gen, latency_total.
  * OUTPUTS           — per-run JSONL, per-(model, T) aggregate with 95% CI
                        (bootstrap), Markdown + LaTeX tables.

Usage:
    python finetuned_unsloth/test/benchmark_all.py               # full sweep
    python finetuned_unsloth/test/benchmark_all.py --models jarvis-v8
    python finetuned_unsloth/test/benchmark_all.py --temperatures 0.0 0.3
    python finetuned_unsloth/test/benchmark_all.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

sys.path.insert(0, "/u/sislam3/Generator")

from inference.mcp_client import JarvisMCP
from inference.render_and_parse import (
    SYSTEM_PROMPT, mcp_tools_to_hf_schema, render_prompt,
    split_think_and_calls, append_tool_call, append_tool_result, initial_messages,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
HARDWARE_LOCAL = "NVIDIA GH200 120GB (Delta-AI, aarch64)"


# ────────────────────────── test cases ──────────────────────────

CASES: list[dict] = [
    # SINGLE
    {"id": "s1", "cat": "single", "prompt": "Create a pipeline named demo_pipeline.",
     "expect_tools": ["create_pipeline"],
     "expect_args":  [{"pipeline_id": "demo_pipeline"}],
     "expect_answer": ["demo_pipeline", "created"]},
    {"id": "s2", "cat": "single", "prompt": "List every Jarvis pipeline I currently have.",
     "expect_tools": ["jm_list_pipelines"], "expect_args": [{}],
     "expect_answer": ["pipeline"]},
    {"id": "s3", "cat": "single", "prompt": "Bootstrap my Jarvis setup for the summit machine.",
     "expect_tools": ["jm_bootstrap_from"],
     "expect_args":  [{"machine": "summit"}],
     "expect_answer": ["summit", "bootstrapped"]},
    {"id": "s4", "cat": "single", "prompt": "Reset the whole Jarvis system.",
     "expect_tools": ["jm_reset"], "expect_args": [{}],
     "expect_answer": ["reset"]},
    {"id": "s5", "cat": "single", "prompt": "Set my current pipeline to gpu_training.",
     "expect_tools": ["jm_cd"], "expect_args": [{"pipeline_id": "gpu_training"}],
     "expect_answer": ["gpu_training", "current"]},
    {"id": "s6", "cat": "single", "prompt": "Build the resource graph with a half-second sleep between operations.",
     "expect_tools": ["jm_graph_build"], "expect_args": [{"net_sleep": 0.5}],
     "expect_answer": ["graph", "built"]},

    # MULTI
    {"id": "m1", "cat": "multi", "prompt": "Create a pipeline named bench_a, then destroy the deprecated_test pipeline.",
     "expect_tools": ["create_pipeline", "destroy_pipeline"],
     "expect_args":  [{"pipeline_id": "bench_a"}, {"pipeline_id": "deprecated_test"}],
     "expect_answer": ["bench_a", "deprecated_test"]},
    {"id": "m2", "cat": "multi", "prompt": "List my pipelines, then show me the resource graph.",
     "expect_tools": ["jm_list_pipelines", "jm_graph_show"],
     "expect_args":  [{}, {}],
     "expect_answer": ["pipeline", "graph"]},

    # CHAIN_FIRST
    {"id": "c1", "cat": "chain_first",
     "prompt": "Create a pipeline called bench_v2, switch to it, and attach an IOR package with 16 procs.",
     "expect_tools": ["create_pipeline", "jm_cd", "append_pkg", "configure_pkg"],
     "expect_args":  [
         {"pipeline_id": "bench_v2"},
         {"pipeline_id": "bench_v2"},
         {"pipeline_id": "bench_v2", "pkg_type": "ior"},
         {"pipeline_id": "bench_v2", "pkg_id": "ior", "extra_args": {"nprocs": 16}},
     ],
     "expect_answer": ["bench_v2", "ior"]},
    {"id": "c2", "cat": "chain_first",
     "prompt": "Load the pipeline climate_forecast_2026 and make it my current pipeline.",
     "expect_tools": ["load_pipeline", "jm_cd"],
     "expect_args":  [{"pipeline_id": "climate_forecast_2026"},
                      {"pipeline_id": "climate_forecast_2026"}],
     "expect_answer": ["climate_forecast_2026"]},

    # ERROR_RECOVERY
    {"id": "e1", "cat": "error_recovery",
     "prompt": "Load the pipeline fresh_pipeline so I can use it; if it doesn't exist, create it first.",
     "expect_tools_any_of": [["load_pipeline", "create_pipeline"],
                              ["create_pipeline", "load_pipeline"],
                              ["load_pipeline"]],
     "expect_answer": ["fresh_pipeline"]},
    {"id": "e2", "cat": "error_recovery",
     "prompt": "Append an mdtest package to pipeline io_bench — if the pipeline is missing, create it and then attach.",
     "expect_tools_any_of": [["append_pkg", "create_pipeline", "append_pkg"],
                              ["create_pipeline", "append_pkg"],
                              ["load_pipeline", "create_pipeline", "append_pkg"]],
     "expect_answer": ["io_bench", "mdtest"]},
]


# ────────────────────────── graders ──────────────────────────

def _grade_ordered(expected: list[str], actual: list[str]) -> bool:
    ai = 0
    for name in expected:
        while ai < len(actual) and actual[ai] != name:
            ai += 1
        if ai >= len(actual): return False
        ai += 1
    return True


def _grade_any_of(options: list[list[str]], actual: list[str]) -> bool:
    return any(_grade_ordered(o, actual) for o in options)


def _args_subset(expected: dict, actual: dict) -> bool:
    for k, v in expected.items():
        if k not in actual: return False
        a = actual[k]
        if isinstance(v, dict) and isinstance(a, dict):
            if not _args_subset(v, a): return False
        elif isinstance(v, float) and isinstance(a, (int, float)):
            if abs(float(a) - v) > 1e-6: return False
        else:
            if str(a) != str(v): return False
    return True


def _grade_args(expected_args: list[dict], calls: list[dict]) -> bool:
    idx = 0
    for exp in expected_args:
        found = False
        while idx < len(calls):
            if _args_subset(exp, calls[idx]["arguments"]):
                found = True
                idx += 1
                break
            idx += 1
        if not found: return False
    return True


def _answer_correctness(final: str, expected_words: list[str]) -> bool:
    low = (final or "").lower()
    return all(w.lower() in low for w in expected_words)


_HALLUC_VERBS = ("created", "destroyed", "loaded", "configured", "reset",
                 "bootstrapped", "attached", "appended", "built")


def _hallucination(final: str, mcp_responses: list[str]) -> int:
    joined = " ".join(mcp_responses).lower()
    score = 0
    for verb in _HALLUC_VERBS:
        for m in re.finditer(rf"(\w+) (?:has been|was|is) {verb}", (final or "").lower()):
            ident = m.group(1).strip(" .,'\"")
            if ident and ident not in joined and ident not in {
                "pipeline", "the", "a", "an", "it", "graph", "system",
            }:
                score += 1
    return score


# ────────────────────────── provider protocol ──────────────────────────

@dataclass
class RunResult:
    calls: list[dict] = field(default_factory=list)
    mcp_responses: list[str] = field(default_factory=list)
    mcp_errors: int = 0
    final_answer: str = ""
    latency_gen_s: float = 0.0


class Provider(Protocol):
    name: str
    family: str
    params: str
    provider: str
    hardware: str
    tag: str

    def run_case(self, mcp: JarvisMCP, case: dict, *, temperature: float,
                 seed: int) -> RunResult: ...


# ---- helpers ----

def _ollama_generate_raw(prompt: str, model: str, temperature: float, seed: int) -> tuple[str, float]:
    payload = {
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0 else 0.95,
            "top_k": 1 if temperature == 0 else 64,
            "seed": seed,
            "num_predict": 512, "num_ctx": 8192,
        },
    }
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", ""), time.time() - t0


def _ollama_chat(model: str, messages: list[dict], tools: list[dict],
                 temperature: float, seed: int) -> tuple[dict, float]:
    payload = {
        "model": model, "messages": messages, "tools": tools, "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0 else 0.95,
            "top_k": 1 if temperature == 0 else 64,
            "seed": seed, "num_ctx": 8192,
        },
    }
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    return body, time.time() - t0


# ---- provider 1: Ollama + FunctionGemma grammar (our models) ----

@dataclass
class OllamaFunctionGemmaProvider:
    name: str                 # e.g. "jarvis-v8"
    tag: str                  # ollama tag
    tokenizer_dir: str
    params: str = "270M"
    family: str = "gemma3 (FT)"
    provider: str = "Ollama"
    hardware: str = HARDWARE_LOCAL

    def __post_init__(self):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

    def run_case(self, mcp, case, *, temperature, seed) -> RunResult:
        tools = mcp_tools_to_hf_schema(mcp.list_tools())
        messages = initial_messages(case["prompt"])
        r = RunResult()
        for step in range(6):
            rendered = render_prompt(self._tokenizer, messages, tools)
            raw, dt = _ollama_generate_raw(rendered, self.tag, temperature, seed)
            r.latency_gen_s += dt
            thought, calls, trailing = split_think_and_calls(raw)
            if not calls:
                r.final_answer = trailing or raw.strip()
                return r
            r.calls.extend(calls)
            append_tool_call(messages, thought, calls)
            for call in calls:
                result = mcp.call_tool(call["name"], call["arguments"])
                r.mcp_responses.append(result)
                if '"error"' in result.lower() or '"iserror": true' in result.lower():
                    r.mcp_errors += 1
                append_tool_result(messages, call["name"], result)
        r.final_answer = "(max iterations)"
        return r


# ---- provider 2: Ollama + OpenAI-style tool calls (general instruct models) ----

@dataclass
class OllamaOpenAIToolsProvider:
    name: str
    tag: str
    params: str
    family: str
    tokenizer_dir: str = ""  # unused
    provider: str = "Ollama"
    hardware: str = HARDWARE_LOCAL

    def run_case(self, mcp, case, *, temperature, seed) -> RunResult:
        raw_tools = mcp.list_tools()
        tools = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t.get("description") or "")[:300],
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        } for t in raw_tools]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": case["prompt"]},
        ]
        r = RunResult()
        for step in range(6):
            body, dt = _ollama_chat(self.tag, messages, tools, temperature, seed)
            r.latency_gen_s += dt
            msg = body.get("message", {})
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                r.final_answer = (msg.get("content") or "").strip()
                return r
            calls = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {}
                calls.append({"name": fn.get("name"), "arguments": args})
            r.calls.extend(calls)
            messages.append(msg)
            for call in calls:
                result = mcp.call_tool(call["name"], call["arguments"])
                r.mcp_responses.append(result)
                if '"error"' in result.lower() or '"iserror": true' in result.lower():
                    r.mcp_errors += 1
                messages.append({"role": "tool", "content": result})
        r.final_answer = "(max iterations)"
        return r


# ---- provider 4: Google Gemini via google-adk + MCP ----
# Requires GOOGLE_API_KEY (AI Studio free tier is fine) OR Vertex AI credentials.
# ADK manages its own MCP subprocess — same jarvis-mcp binary as Claude SDK, so
# outcomes are comparable.

@dataclass
class GoogleADKProvider:
    name: str                  # e.g. "gemini-2.5-flash"
    tag: str                   # gemini model id
    params: str = "?"
    family: str = "gemini"
    provider: str = "Google ADK"
    hardware: str = "Google data center (opaque)"
    tokenizer_dir: str = ""

    def __post_init__(self):
        if not (os.environ.get("GOOGLE_API_KEY") or
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")):
            raise RuntimeError("GOOGLE_API_KEY (or GOOGLE_APPLICATION_CREDENTIALS) not set")

    def run_case(self, mcp, case, *, temperature, seed) -> RunResult:
        import asyncio
        return asyncio.run(self._run_async(case, temperature))

    async def _run_async(self, case, temperature) -> RunResult:
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
        from mcp import StdioServerParameters
        from google.genai import types as gen_types

        toolset = MCPToolset(
            connection_params=StdioServerParameters(
                command=JARVIS_MCP_CMD,
                args=[], env=dict(os.environ),
            ),
        )
        agent = Agent(
            model=self.tag,
            name="jarvis_agent",
            instruction=SYSTEM_PROMPT,
            tools=[toolset],
            generate_content_config=gen_types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="bench", user_id="u", session_id=f"{self.name}-{time.time():.0f}",
        )
        runner = Runner(agent=agent, session_service=session_service, app_name="bench")

        r = RunResult()
        t0 = time.time()
        content = gen_types.Content(role="user", parts=[gen_types.Part(text=case["prompt"])])
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=content,
        ):
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    r.calls.append({"name": fc.name, "arguments": dict(fc.args or {})})
                elif getattr(part, "function_response", None):
                    fr = part.function_response
                    text = json.dumps(fr.response) if not isinstance(fr.response, str) else fr.response
                    r.mcp_responses.append(text)
                    if '"error"' in text.lower() or '"iserror": true' in text.lower():
                        r.mcp_errors += 1
                elif getattr(part, "text", None):
                    r.final_answer = (r.final_answer + "\n" + part.text).strip() \
                        if r.final_answer else part.text.strip()
        r.latency_gen_s = time.time() - t0
        # best-effort cleanup of the MCP subprocess
        try:
            await toolset.close()
        except Exception:
            pass
        return r


# ---- provider 3: Claude via claude-agent-sdk (uses Claude Code session auth) ----
# NB: Claude Code SDK drives its own MCP subprocess — our `mcp` arg is ignored
# for the tool-call round-trips, but we point the SDK at the SAME jarvis-mcp
# binary so outcomes are comparable. The SDK controls sampling internally,
# so temperature isn't directly configurable here — we report "default".

JARVIS_MCP_CMD = "/u/sislam3/clio-kit/clio-kit-mcp-servers/jarvis/.venv/bin/jarvis-mcp"
JARVIS_MCP_CWD = "/u/sislam3/clio-kit/clio-kit-mcp-servers/jarvis"


@dataclass
class AnthropicAgentSDKProvider:
    name: str                  # e.g. "claude-sonnet-4-6"
    tag: str                   # model id: "sonnet", "haiku", "opus", or full
    params: str = "?"
    family: str = "claude"
    provider: str = "Claude Code SDK"
    hardware: str = "Anthropic data center (opaque)"
    tokenizer_dir: str = ""
    max_turns: int = 8

    def run_case(self, mcp, case, *, temperature, seed) -> RunResult:
        import asyncio
        return asyncio.run(self._run_async(case))

    async def _run_async(self, case) -> RunResult:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions,
            AssistantMessage, UserMessage, ResultMessage,
            TextBlock, ToolUseBlock, ToolResultBlock,
        )
        opts = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=self.tag,
            mcp_servers={
                "jarvis": {
                    "type": "stdio",
                    "command": JARVIS_MCP_CMD,
                    "args": [],
                    "env": dict(os.environ),
                },
            },
            # Allow every jarvis-mcp tool without prompting. The SDK's
            # allowed_tools is a whitelist, so we list all 29 tool names
            # in the `mcp__jarvis__<tool>` namespace.
            allowed_tools=[f"mcp__jarvis__{t['name']}" for t in [
                {"name":"update_pipeline"},{"name":"build_pipeline_env"},
                {"name":"create_pipeline"},{"name":"load_pipeline"},
                {"name":"get_pkg_config"},{"name":"destroy_pipeline"},
                {"name":"run_pipeline"},{"name":"unlink_pkg"},
                {"name":"remove_pkg"},{"name":"append_pkg"},
                {"name":"configure_pkg"},{"name":"jm_list_pipelines"},
                {"name":"jm_list_repos"},{"name":"jm_reset"},
                {"name":"jm_bootstrap_list"},{"name":"jm_bootstrap_from"},
                {"name":"jm_cd"},{"name":"jm_create_config"},
                {"name":"jm_load_config"},{"name":"jm_save_config"},
                {"name":"jm_set_hostfile"},{"name":"jm_construct_pkg"},
                {"name":"jm_add_repo"},{"name":"jm_remove_repo"},
                {"name":"jm_promote_repo"},{"name":"jm_get_repo"},
                {"name":"jm_graph_show"},{"name":"jm_graph_build"},
                {"name":"jm_graph_modify"}
            ]],
            max_turns=self.max_turns,
            permission_mode="bypassPermissions",
            # required empties:
            tools=None,
            betas=(), add_dirs=(), env={}, extra_args={}, plugins=(),
        )
        r = RunResult()
        t0 = time.time()
        async for msg in query(prompt=case["prompt"], options=opts):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        name = block.name
                        if "__" in name:        # strip mcp__jarvis__ prefix
                            name = name.split("__")[-1]
                        r.calls.append({"name": name, "arguments": block.input or {}})
                    elif isinstance(block, TextBlock):
                        # the final text is the LAST TextBlock in the stream
                        r.final_answer = (r.final_answer + "\n" + (block.text or "")).strip() \
                            if r.final_answer else (block.text or "").strip()
            elif isinstance(msg, UserMessage):
                # the SDK feeds back tool results as UserMessage with tool_result content
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            text = str(block.content)
                            r.mcp_responses.append(text)
                            if '"error"' in text.lower() or '"iserror": true' in text.lower() \
                               or getattr(block, "is_error", False):
                                r.mcp_errors += 1
            elif isinstance(msg, ResultMessage):
                # final metadata (tokens, usage, etc.) — nothing to add
                pass
        r.latency_gen_s = time.time() - t0
        return r


# ────────────────────────── model registry ──────────────────────────

def default_providers() -> list[Provider]:
    provs: list[Provider] = []
    provs.append(OllamaFunctionGemmaProvider(
        name="jarvis-v7", tag="jarvis-v7:latest",
        tokenizer_dir="/u/sislam3/Generator/finetuned_unsloth/artifacts/model_merged_16bit",
    ))
    provs.append(OllamaFunctionGemmaProvider(
        name="jarvis-v8", tag="jarvis-v8:latest",
        tokenizer_dir="/u/sislam3/Generator/finetuned_unsloth/artifacts/v8/model_merged_16bit",
    ))
    v9_dir = Path("/u/sislam3/Generator/finetuned_unsloth/artifacts/v9/model_merged_16bit")
    if v9_dir.exists():
        provs.append(OllamaFunctionGemmaProvider(
            name="jarvis-v9", tag="jarvis-v9:latest", tokenizer_dir=str(v9_dir)))
    # v10: tokenizer lives on shared scratch since we don't copy the big model into the repo
    v10_dir = Path("/work/hdd/bekn/sislam3/jarvis_v10_lora/merged_16bit")
    if v10_dir.exists():
        provs.append(OllamaFunctionGemmaProvider(
            name="jarvis-v10", tag="jarvis-v10:latest", tokenizer_dir=str(v10_dir)))
    # Local Ollama instruct models (native OpenAI-style tool calling).
    # Add any tag that has been pulled with `ollama pull <tag>` to the list below.
    for alias, tag, params, family in [
        ("qwen2.5-7b",      "qwen2.5:7b-instruct",   "7B",   "qwen2.5"),
        ("gpt-oss-20b",     "gpt-oss:20b",           "20B",  "gpt-oss"),
        ("llama3.1-8b",     "llama3.1:8b",           "8B",   "llama3.1"),
        ("mistral-7b",      "mistral:7b-instruct",   "7B",   "mistral"),
        ("nemotron-nano-30b",    "nemotron-3-nano:30b",     "30B",  "nemotron-3"),
        ("nemotron-cascade-30b", "nemotron-cascade-2:30b",  "30B",  "nemotron-cascade-2"),
        ("nemotron-super-120b",  "nemotron-3-super:120b",   "120B", "nemotron-3"),
    ]:
        provs.append(OllamaOpenAIToolsProvider(
            name=alias, tag=tag, params=params, family=family,
        ))
    # Claude via Claude Code SDK (uses local `claude` CLI auth; no API key needed).
    # Require `CLAUDE_AGENT_SDK=1` env var to opt-in so the benchmark doesn't
    # accidentally spend session budget. Skip silently otherwise.
    if os.environ.get("CLAUDE_AGENT_SDK") == "1":
        for label, tag in [
            ("claude-haiku-4-5",  "haiku"),
            ("claude-sonnet-4-6", "sonnet"),
            ("claude-opus-4-7",   "opus"),
        ]:
            try:
                provs.append(AnthropicAgentSDKProvider(name=label, tag=tag))
            except Exception as e:
                print(f"[warn] could not init {label}: {e}", file=sys.stderr)
    # Google Gemini via ADK — needs GOOGLE_API_KEY; opt-in via GOOGLE_ADK=1.
    if os.environ.get("GOOGLE_ADK") == "1":
        for label, tag in [
            ("gemini-2.5-flash", "gemini-2.5-flash"),
            ("gemini-2.5-pro",   "gemini-2.5-pro"),
        ]:
            try:
                provs.append(GoogleADKProvider(name=label, tag=tag))
            except Exception as e:
                print(f"[warn] could not init {label}: {e}", file=sys.stderr)
    return provs


# ────────────────────────── scoring / aggregation ──────────────────────────

def score_one(case: dict, r: RunResult, wall_s: float) -> dict:
    actual_names = [c["name"] for c in r.calls]
    if "expect_tools_any_of" in case:
        tool_ok = _grade_any_of(case["expect_tools_any_of"], actual_names)
        arg_ok = tool_ok
    else:
        tool_ok = _grade_ordered(case["expect_tools"], actual_names)
        arg_ok = tool_ok and _grade_args(case["expect_args"], r.calls)
    task_success = bool(tool_ok and arg_ok and r.mcp_errors == 0)
    answer_ok = _answer_correctness(r.final_answer, case.get("expect_answer", []))
    halluc = _hallucination(r.final_answer, r.mcp_responses)
    return {
        "case_id": case["id"], "cat": case["cat"],
        "tool_ok": tool_ok, "arg_ok": arg_ok,
        "task_success": task_success, "answer_ok": answer_ok,
        "hallucinations": halluc, "mcp_errors": r.mcp_errors,
        "n_calls": len(r.calls), "calls": actual_names,
        "answer": (r.final_answer or "")[:400],
        "latency_gen_s": round(r.latency_gen_s, 3),
        "latency_total_s": round(wall_s, 3),
    }


def _bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(0)
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(alpha/2 * n_boot)]
    hi = boots[int((1 - alpha/2) * n_boot)]
    return mean, lo, hi


def aggregate(rows: list[dict]) -> dict:
    """Aggregate rows by (model, temperature)."""
    groups: dict[tuple[str, float], dict] = {}
    for r in rows:
        key = (r["model"], r["temperature"])
        g = groups.setdefault(key, {
            "n_runs": 0, "by_metric": {k: [] for k in
                ["tool_ok","arg_ok","task_success","answer_ok",
                 "hallucinations","mcp_errors","latency_gen_s","latency_total_s"]},
            "meta": {k: r.get(k) for k in ["family","params","provider","hardware"]},
        })
        g["n_runs"] += 1
        for k in g["by_metric"]:
            g["by_metric"][k].append(float(r[k]) if not isinstance(r[k], bool) else (1.0 if r[k] else 0.0))
    out = {}
    for (model, T), g in groups.items():
        rec = {"model": model, "temperature": T, "n_runs": g["n_runs"], **g["meta"]}
        for k, vs in g["by_metric"].items():
            mean, lo, hi = _bootstrap_ci(vs)
            rec[f"{k}_mean"] = round(mean, 4)
            rec[f"{k}_lo"] = round(lo, 4)
            rec[f"{k}_hi"] = round(hi, 4)
        out[(model, T)] = rec
    return out


def best_T_per_model(agg: dict) -> dict:
    best: dict[str, tuple[float, float]] = {}
    for (model, T), rec in agg.items():
        cur = best.get(model)
        score = rec["task_success_mean"]
        # tie-break on faster generation
        if not cur or score > cur[1] or (score == cur[1] and rec["latency_gen_s_mean"] < agg[(model, cur[0])]["latency_gen_s_mean"]):
            best[model] = (T, score)
    return best


# ────────────────────────── reporting ──────────────────────────

def render_md(agg: dict, best: dict, out_md: Path, env_rows: list[dict]):
    lines = [
        "# Cross-model / cross-temperature benchmark on real Jarvis MCP", "",
        "Every provider calls the **real** `jarvis-mcp` stdio server (29 Jarvis-CD tools). "
        "Each row below is the mean across 12 held-out cases (6 single, 2 multi, "
        "2 chain_first, 2 error_recovery) × repeats; 95% CIs are bootstrap intervals.",
        "",
        "## Environment",
        "| model | family | params | provider | hardware |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for e in env_rows:
        key = e["model"]
        if key in seen: continue
        seen.add(key)
        lines.append(f"| `{e['model']}` | {e['family']} | {e['params']} | {e['provider']} | {e['hardware']} |")
    lines.append("")
    lines.append("## Per-(model, temperature) aggregate")
    lines.append("| model | T | task_success | tool_ok | arg_ok | answer_ok | halluc | mcp_err | gen_s | total_s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def pct(rec, k):
        m = rec[f"{k}_mean"]
        lo = rec[f"{k}_lo"]; hi = rec[f"{k}_hi"]
        return f"{100*m:.0f}% [{100*lo:.0f}, {100*hi:.0f}]"

    def num(rec, k):
        m = rec[f"{k}_mean"]
        lo = rec[f"{k}_lo"]; hi = rec[f"{k}_hi"]
        return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"

    for (model, T), rec in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        tag = "**" if best.get(model, (None,))[0] == T else ""
        lines.append(f"| {tag}`{model}`{tag} | {T:.1f} | {pct(rec,'task_success')} | "
                     f"{pct(rec,'tool_ok')} | {pct(rec,'arg_ok')} | {pct(rec,'answer_ok')} | "
                     f"{rec['hallucinations_mean']:.2f} | {rec['mcp_errors_mean']:.2f} | "
                     f"{rec['latency_gen_s_mean']:.2f} | {rec['latency_total_s_mean']:.2f} |")

    lines.append("")
    lines.append("**Bold** = temperature with the highest `task_success_mean` for that model "
                 "(tie-broken by lowest `gen_s_mean`).")
    lines.append("")
    lines.append("## Best temperature per model")
    lines.append("| model | best_T | task_success@T | family | provider |")
    lines.append("|---|---|---|---|---|")
    for model, (T, score) in sorted(best.items()):
        rec = agg[(model, T)]
        lines.append(f"| `{model}` | {T:.1f} | {100*score:.0f}% | {rec['family']} | {rec['provider']} |")
    out_md.write_text("\n".join(lines))
    print(f"Markdown report → {out_md}")


def render_latex(agg: dict, best: dict, out_tex: Path):
    """A compact main-result LaTeX table for a paper."""
    lines = [r"\begin{table*}[t]",
             r"\centering",
             r"\small",
             r"\caption{Real-MCP task success across providers and temperatures. Numbers are mean \% over 12 cases; 95\% CI in brackets.}",
             r"\label{tab:mcp_bench}",
             r"\begin{tabular}{llcccccc}",
             r"\toprule",
             r"Model & T & Task Success & Tool Acc. & Arg Acc. & Ans. Corr. & Halluc. & Lat.\,(s) \\",
             r"\midrule"]
    for (model, T), rec in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        bold_open, bold_close = "", ""
        if best.get(model, (None,))[0] == T:
            bold_open, bold_close = r"\textbf{", r"}"
        lines.append(
            f"{bold_open}{model.replace('_','-')}{bold_close} & {T:.1f} & "
            f"{100*rec['task_success_mean']:.0f} [{100*rec['task_success_lo']:.0f}, {100*rec['task_success_hi']:.0f}] & "
            f"{100*rec['tool_ok_mean']:.0f} & {100*rec['arg_ok_mean']:.0f} & "
            f"{100*rec['answer_ok_mean']:.0f} & {rec['hallucinations_mean']:.1f} & "
            f"{rec['latency_gen_s_mean']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out_tex.write_text("\n".join(lines))
    print(f"LaTeX table → {out_tex}")


# ────────────────────────── driver ──────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of provider names to run (default: all available)")
    ap.add_argument("--temperatures", nargs="*", type=float,
                    default=[0.0, 0.3, 0.7],
                    help="Temperature sweep (default: 0.0 0.3 0.7)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Runs per non-zero temperature (default: 3). T=0 always 1 run.")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--case-timeout", type=int, default=0,
                    help="Per-case wall-clock cap in seconds (0 = off). "
                         "On timeout, record a TIMEOUT row and move on. Useful for "
                         "cloud SDKs that can block indefinitely on a flaky case.")
    args = ap.parse_args()

    # Install a SIGALRM-based per-case watchdog if requested.
    import signal
    class _CaseTimeout(Exception):
        pass
    def _alarm(sig, frame):
        raise _CaseTimeout()
    if args.case_timeout > 0:
        signal.signal(signal.SIGALRM, _alarm)

    provs = default_providers()
    if args.models:
        provs = [p for p in provs if p.name in args.models]
    if not provs:
        print("No providers selected.", file=sys.stderr); sys.exit(2)

    logdir = Path("/u/sislam3/Generator/finetuned_unsloth/logs")
    logdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = logdir / f"benchmark_{args.tag}.jsonl"
    out_md    = logdir / f"benchmark_{args.tag}.md"
    out_tex   = logdir / f"benchmark_{args.tag}.tex"

    all_rows: list[dict] = []
    env_rows: list[dict] = []

    with out_jsonl.open("w") as f, JarvisMCP() as mcp:
        mcp_tools = mcp.list_tools()
        print(f"Connected to real MCP: {len(mcp_tools)} tools")
        for prov in provs:
            env_rows.append({"model": prov.name, "family": prov.family,
                             "params": prov.params, "provider": prov.provider,
                             "hardware": prov.hardware})
            # Claude Code SDK controls sampling itself — run once at T=default.
            # Google ADK supports temperature; sweep it like Ollama providers.
            if isinstance(prov, AnthropicAgentSDKProvider):
                temps = [0.0]   # single "default" bucket
                repeats = 1
            else:
                temps = args.temperatures
                repeats = args.repeats
            for T in temps:
                n = 1 if T == 0.0 else repeats
                for run_idx in range(n):
                    print(f"\n=== {prov.name} | T={T} | run {run_idx+1}/{n} ===")
                    for case in CASES:
                        mcp.call_tool("jm_reset", {})
                        t0 = time.time()
                        if args.case_timeout > 0:
                            signal.alarm(args.case_timeout)
                        try:
                            r = prov.run_case(mcp, case, temperature=T, seed=42 + run_idx)
                        except _CaseTimeout:
                            r = RunResult(final_answer=f"TIMEOUT after {args.case_timeout}s")
                            print(f"  [{case['id']}] TIMEOUT @ {args.case_timeout}s", flush=True)
                        except Exception as e:
                            r = RunResult(final_answer=f"EXCEPTION: {e}")
                            traceback.print_exc(file=sys.stdout)
                        finally:
                            if args.case_timeout > 0:
                                signal.alarm(0)
                        wall = time.time() - t0
                        row = score_one(case, r, wall)
                        row.update({
                            "model": prov.name, "ollama_tag": prov.tag,
                            "family": prov.family, "params": prov.params,
                            "provider": prov.provider, "hardware": prov.hardware,
                            "temperature": T, "run_idx": run_idx,
                        })
                        all_rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        flag = "✓" if row["task_success"] else ("~" if row["tool_ok"] else "✗")
                        print(f"  [{case['id']}] {flag} tool={int(row['tool_ok'])} "
                              f"arg={int(row['arg_ok'])} ans={int(row['answer_ok'])} "
                              f"halluc={row['hallucinations']} mcp_err={row['mcp_errors']} "
                              f"t={row['latency_total_s']:.1f}s")
                        f.flush()

    agg = aggregate(all_rows)
    best = best_T_per_model(agg)
    render_md(agg, best, out_md, env_rows)
    render_latex(agg, best, out_tex)
    print(f"\nJSONL → {out_jsonl}")


if __name__ == "__main__":
    main()
