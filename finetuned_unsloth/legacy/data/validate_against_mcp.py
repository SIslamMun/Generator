"""Post-filter raw generator output by REPLAYING each trace against the real
Jarvis MCP server. Any example where a 'success' step returns an error, or
where an 'error' step the model invented doesn't match the actual error shape,
is dropped.

Also enforces:
  - chain dependencies: create_pipeline before any operation on that pipeline
  - jm_reset between every example so state doesn't leak
  - dict extra_args for configure_pkg / append_pkg
  - float net_sleep for jm_graph_{build,modify}

Input:  raw JSON list of examples (v7-shape: {instruction, solution:{method, reasoning_path}})
Output: filtered JSON list + a companion stats JSON

Usage:
    python validate_against_mcp.py --input v10_delta_raw.json \
                                   --output v10_delta_clean.json \
                                   --stats  v10_delta_stats.json
"""

import argparse
import json
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, "/u/sislam3/Generator")
from inference.mcp_client import JarvisMCP


class ReplayTimeout(Exception):
    pass


@contextmanager
def per_example_deadline(seconds: int):
    """SIGALRM-based wall-clock cap for a whole example. Linux-only, good enough here."""
    def _handler(signum, frame):
        raise ReplayTimeout(f"example exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


REQUIRED_DICT_ARGS = {
    "configure_pkg": "extra_args",
    "append_pkg": "extra_args",
}
REQUIRED_FLOAT_ARGS = {
    "jm_graph_build": "net_sleep",
    "jm_graph_modify": "net_sleep",
}


def quick_schema_check(tool: str, args: dict) -> str | None:
    """Cheap pre-check before we spend MCP calls. Returns error string or None."""
    if tool in REQUIRED_DICT_ARGS:
        k = REQUIRED_DICT_ARGS[tool]
        v = args.get(k)
        if v is not None and not isinstance(v, dict):
            return f"{tool}.{k} must be dict, got {type(v).__name__}"
    if tool in REQUIRED_FLOAT_ARGS:
        k = REQUIRED_FLOAT_ARGS[tool]
        v = args.get(k)
        if v is None:
            return f"{tool}.{k} required"
        if not isinstance(v, (int, float)):
            try: float(v)
            except Exception: return f"{tool}.{k} must be number, got {v!r}"
    return None


def replay(mcp, example: dict) -> dict:
    """Replay all reasoning_path steps against the real MCP.

    Returns {ok: bool, reason: str, updated_path: [...]} where updated_path
    has each step annotated with `actual_mcp_result` / `actual_mcp_error`.
    """
    # Reset MCP before every example
    try:
        mcp.call_tool("jm_reset", {})
    except Exception as e:
        return {"ok": False, "reason": f"jm_reset failed: {e}", "updated_path": []}

    updated = []
    reasoning = example.get("solution", {}).get("reasoning_path", [])

    for i, step in enumerate(reasoning):
        tool = step.get("tool")
        args = step.get("args", {}) or {}
        status = step.get("status", "success")

        if not tool:
            # "speak-only" step — nothing to replay
            updated.append(dict(step))
            continue

        # Schema pre-check
        se = quick_schema_check(tool, args)
        if se is not None:
            return {"ok": False, "reason": f"step {i} ({tool}): {se}", "updated_path": updated}

        # Actually call MCP with an aggressive per-call timeout.
        # run_pipeline / build_pipeline_env can hang for minutes; we don't
        # need to wait that long to tell whether the *call shape* is valid.
        try:
            result_text = mcp.call_tool(tool, args, timeout=20.0)
        except Exception as e:
            return {"ok": False, "reason": f"step {i} ({tool}) MCP call raised: {e}", "updated_path": updated}

        is_error = (
            '"error"' in result_text.lower()
            or '"iserror": true' in result_text.lower()
            or result_text.startswith('{"error"')
        )

        new_step = dict(step)
        new_step["actual_mcp_result"] = result_text[:500]  # cap length
        new_step["actual_mcp_is_error"] = is_error

        if status == "success" and is_error:
            return {"ok": False,
                    "reason": f"step {i} ({tool}) expected success but got error: {result_text[:120]}",
                    "updated_path": updated + [new_step]}
        if status == "failure" and not is_error:
            # Model invented a failure step that actually works — drop.
            return {"ok": False,
                    "reason": f"step {i} ({tool}) marked failure but MCP succeeded",
                    "updated_path": updated + [new_step]}

        updated.append(new_step)

    return {"ok": True, "reason": "ok", "updated_path": updated}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--stats",  "-s", default=None)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="Debug: stop after N examples (0 = all)")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    stats_path = Path(args.stats).resolve() if args.stats else dst.with_suffix(".stats.json")
    dst.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text())
    print(f"loaded {len(data)} raw examples from {src}")
    if args.max_examples:
        data = data[: args.max_examples]

    kept, dropped_by_reason = [], {}
    drop_samples = {}
    t0 = time.time()
    with JarvisMCP() as mcp:
        for i, ex in enumerate(data):
            try:
                with per_example_deadline(60):  # whole-example wall-clock cap
                    r = replay(mcp, ex)
            except ReplayTimeout as e:
                r = {"ok": False, "reason": f"timeout: {e}", "updated_path": []}
            except Exception as e:
                r = {"ok": False, "reason": f"fatal: {e}", "updated_path": []}

            if r["ok"]:
                # Attach the replay annotations onto the original solution
                ex["solution"]["reasoning_path"] = r["updated_path"]
                kept.append(ex)
            else:
                bucket = r["reason"].split(":")[0][:40]
                dropped_by_reason[bucket] = dropped_by_reason.get(bucket, 0) + 1
                drop_samples.setdefault(bucket, r["reason"])

            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                keep_rate = 100 * len(kept) / (i + 1)
                print(f"  [{i+1:>5}/{len(data)}]  kept={len(kept)}  "
                      f"({keep_rate:.1f}%)  dropped={i+1-len(kept)}  "
                      f"t={dt:.0f}s", flush=True)

    # Final reset so subsequent test runs start clean
    try:
        with JarvisMCP() as mcp: mcp.call_tool("jm_reset", {})
    except Exception: pass

    dst.write_text(json.dumps(kept, indent=2))
    stats = {
        "input": str(src), "output": str(dst),
        "input_count": len(data),
        "kept_count": len(kept),
        "drop_count": len(data) - len(kept),
        "keep_rate": round(len(kept) / max(len(data), 1), 4),
        "dropped_by_reason": dropped_by_reason,
        "sample_messages": drop_samples,
        "elapsed_s": round(time.time() - t0, 2),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print()
    print(f"kept   : {len(kept):>5} / {len(data)}  ({100*len(kept)/max(len(data),1):.1f}%)")
    print(f"dropped: {len(data)-len(kept):>5}")
    for reason, n in sorted(dropped_by_reason.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {reason}")
    print(f"\nclean → {dst}")
    print(f"stats → {stats_path}")


if __name__ == "__main__":
    main()
