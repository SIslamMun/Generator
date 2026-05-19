"""End-to-end verification: trained model → tool call → REAL NDP server.

For each probe:
  1. model generates a tool call
  2. parse_tool_call() extracts {name, arguments}
  3. the call is EXECUTED against the live NDP API (http://155.101.6.191:8003)
     — the same endpoints the ndp_mcp server wraps
  4. we report whether the call ran and what NDP returned

This is the real check: does the fine-tuned model emit calls that actually
work against the National Data Platform, end to end?
"""
from __future__ import annotations

# COMPAT shim — mamba_ssm 2.2.5 vs transformers v5 (see test_inference.py)
import transformers.generation as _g, transformers.generation.utils as _gu
for _c in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
    if not hasattr(_g, _c):
        setattr(_g, _c, getattr(_gu, "GenerateDecoderOnlyOutput", _gu.ModelOutput))

import json
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# MERGED_DIR overridable so we can baseline-test an archived run.
MERGED_DIR = Path(os.environ.get("MERGED_DIR", HERE / "artifacts" / "merged_16bit"))
NDP_TOOLS  = HERE.parent.parent.parent / "configs" / "tools" / "ndp_tools.json"
NDP_BASE   = "http://155.101.6.191:8003"

# reuse the parser + catalog loader + probes from test_inference.py
sys.path.insert(0, str(HERE))
from test_inference import parse_tool_call, load_tools_for_chat_template, PROBES  # noqa: E402

ANTI_HALLUCINATION_SYSTEM = (
    "Tool-call discipline:\n"
    "- ONLY include parameters that you are actually setting to a value.\n"
    "- NEVER include parameters whose value would be None, null, empty string, or unset.\n"
    "- NEVER invent parameter names that are not in the tool's schema.\n"
    "- If a parameter is optional and you're not using it, OMIT IT ENTIRELY (do not emit it with a None placeholder)."
)


# ─────────────────────────── NDP execution layer ───────────────────

def _http_get(path: str, params: list[tuple]) -> tuple[int, object]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{NDP_BASE}{path}?{qs}" if qs else f"{NDP_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def _http_post(path: str, params: list[tuple], body: dict) -> tuple[int, object]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{NDP_BASE}{path}?{qs}" if qs else f"{NDP_BASE}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def execute_on_ndp(name: str, args: dict) -> dict:
    """Execute a parsed tool call against the live NDP API.

    Mirrors what ndp_mcp/server.py does internally. Returns a small
    summary dict: {ok, http, summary}.
    """
    server = args.get("server", "global")
    if name == "list_organizations":
        params = [("server", server)]
        if args.get("name_filter"):
            params.append(("name", args["name_filter"]))
        code, body = _http_get("/organization", params)
        ok = code == 200 and isinstance(body, list)
        return {"ok": ok, "http": code,
                "summary": f"{len(body)} orgs" if ok else str(body)[:160]}

    if name == "search_datasets":
        if args.get("search_terms"):
            params = [("server", server)]
            for t in args["search_terms"]:
                params.append(("terms", t))
            if args.get("search_keys"):
                for k in args["search_keys"]:
                    params.append(("keys", k))
            code, body = _http_get("/search", params)
        else:
            # advanced search → POST with field filters
            body_fields = {k: v for k, v in args.items() if k != "server"}
            code, body = _http_post("/search", [("server", server)], body_fields)
        ok = code == 200 and isinstance(body, list)
        return {"ok": ok, "http": code,
                "summary": f"{len(body)} datasets" if ok else str(body)[:160]}

    if name == "get_dataset_details":
        ident = args.get("dataset_identifier", "")
        itype = args.get("identifier_type", "id")
        field = "dataset_name" if itype == "name" else None
        # the MCP server resolves details via advanced search
        if field:
            code, body = _http_post("/search", [("server", server)], {field: ident})
        else:
            # id lookup — advanced search returns all, we filter client-side
            code, body = _http_post("/search", [("server", server)], {})
            if code == 200 and isinstance(body, list):
                body = [d for d in body if d.get("id") == ident]
        ok = code == 200 and isinstance(body, list)
        hit = len(body) if ok else 0
        return {"ok": ok, "http": code,
                "summary": f"{hit} match" if ok else str(body)[:160]}

    return {"ok": False, "http": -1, "summary": f"unknown tool {name!r}"}


# ─────────────────────────── main ──────────────────────────────────

def main():
    if not MERGED_DIR.exists():
        sys.exit(f"ERROR: {MERGED_DIR} missing")

    from unsloth import FastLanguageModel
    print(f"=== loading {MERGED_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_DIR), max_seq_length=4096,
        load_in_4bit=False, load_in_8bit=False, trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)
    tools = load_tools_for_chat_template()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  ready. tools={[t['function']['name'] for t in tools]}\n")

    n_ok, n_total = 0, 0
    for i, query in enumerate(PROBES, 1):
        n_total += 1
        print("=" * 78)
        print(f"[{i}/{len(PROBES)}] {query}")
        messages = [
            {"role": "system", "content": ANTI_HALLUCINATION_SYSTEM},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, use_cache=False,
                eos_token_id=im_end, pad_token_id=tokenizer.eos_token_id,
                stop_strings=["</function>"], tokenizer=tokenizer)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=False)
        print(f"  RAW  : {' '.join(decoded.split())[:400]}")
        call = parse_tool_call(decoded, tools)
        if not call:
            print("  PARSE: ✗ no tool call found")
            continue
        print(f"  CALL : {call['name']}({json.dumps(call['arguments'])})")
        result = execute_on_ndp(call["name"], call["arguments"])
        mark = "✓" if result["ok"] else "✗"
        print(f"  NDP  : {mark} HTTP {result['http']} — {result['summary']}")
        if result["ok"]:
            n_ok += 1

    print("\n" + "=" * 78)
    print(f"RESULT: {n_ok}/{n_total} probes produced a call that ran OK against live NDP")


if __name__ == "__main__":
    main()
