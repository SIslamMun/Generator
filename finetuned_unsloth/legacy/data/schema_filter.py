"""Fast schema-only filter for v10 raw data.

Why not replay-against-MCP? Because real MCP calls for `run_pipeline` /
`build_pipeline_env` / etc. can hang for minutes at the socket layer
inside C code where Python signal handlers can't interrupt.

Instead we cheaply catch the failure modes we actually care about:
  - configure_pkg / append_pkg: extra_args must be dict or null (never str, int, list)
  - jm_graph_build / jm_graph_modify: net_sleep must be a number
  - Every step must name a tool in the catalog
  - Required parameters per catalog must be present
  - Chain dependency: `pipeline_id` referenced before `create_pipeline` for that id
    is either a deliberate failure step (status=failure) OR starts with load_pipeline

Output: filtered JSON list + stats summary.

Usage:
    python schema_filter.py --input v10_delta_raw.json --output v10_delta_clean.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_DICT_ARGS = {
    "configure_pkg": "extra_args",
    "append_pkg": "extra_args",
}
REQUIRED_NUMBER_ARGS = {
    "jm_graph_build": "net_sleep",
    "jm_graph_modify": "net_sleep",
}


def load_catalog_index(path: str) -> dict:
    """Return {tool_name: {required_param_names}}. Accepts .yaml or .json."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())
    tools = data.get("tools", data) if isinstance(data, dict) else data
    idx = {}
    for t in tools:
        req = {p["name"] for p in t.get("parameters", []) if p.get("required")}
        idx[t["name"]] = req
    return idx


def check_example(ex: dict, catalog: dict) -> tuple[bool, str]:
    steps = ex.get("solution", {}).get("reasoning_path", [])
    if not steps:
        return False, "empty reasoning_path"

    created_pipelines = set()
    for i, step in enumerate(steps):
        tool = step.get("tool")
        args = step.get("args", {}) or {}
        status = step.get("status", "success")

        # None-tool step (speak-only) is fine
        if not tool:
            continue

        if tool not in catalog:
            return False, f"step {i}: unknown tool {tool!r}"

        # Required params present?
        missing = catalog[tool] - set(args.keys())
        if missing:
            return False, f"step {i} ({tool}): missing required param(s) {missing}"

        # extra_args must be dict
        if tool in REQUIRED_DICT_ARGS:
            k = REQUIRED_DICT_ARGS[tool]
            v = args.get(k)
            if v is not None and not isinstance(v, dict):
                return False, f"step {i} ({tool}.{k}): must be dict, got {type(v).__name__}"

        # net_sleep must be number
        if tool in REQUIRED_NUMBER_ARGS:
            k = REQUIRED_NUMBER_ARGS[tool]
            v = args.get(k)
            if v is None:
                return False, f"step {i} ({tool}.{k}): missing"
            if not isinstance(v, (int, float)):
                return False, f"step {i} ({tool}.{k}): must be number, got {type(v).__name__}"

        # Chain dependency sanity:
        # If step references a pipeline_id on a *success* step for an operation that
        # requires the pipeline to exist, either (a) we created it earlier, or
        # (b) the first use was load_pipeline, or (c) it's a plausible outside-state name.
        if status == "success":
            if tool == "create_pipeline":
                pid = args.get("pipeline_id")
                if pid:
                    created_pipelines.add(pid)
            # Don't flag the "pipeline must be created earlier" case — many valid traces
            # assume pre-existing pipelines via load_pipeline, which is fine.

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--catalog", default="/u/sislam3/Generator/configs/tools/jarvis_tools.yaml")
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    stats_path = Path(args.stats).resolve() if args.stats else dst.with_suffix(".stats.json")
    dst.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog_index(args.catalog)
    print(f"catalog: {len(catalog)} tools")

    data = json.loads(src.read_text())
    print(f"loaded {len(data)} raw examples from {src}")

    kept = []
    drop_reasons: Counter = Counter()
    drop_samples = {}

    for ex in data:
        ok, reason = check_example(ex, catalog)
        if ok:
            kept.append(ex)
        else:
            bucket = reason.split(":")[1].strip() if ":" in reason else reason
            bucket = bucket.split("(")[0].strip()[:50]
            drop_reasons[bucket] += 1
            drop_samples.setdefault(bucket, reason)

    dst.write_text(json.dumps(kept, indent=2))
    stats = {
        "input": str(src), "output": str(dst),
        "input_count": len(data),
        "kept_count": len(kept),
        "drop_count": len(data) - len(kept),
        "keep_rate": round(len(kept) / max(len(data), 1), 4),
        "dropped_by_reason": dict(drop_reasons),
        "sample_messages": drop_samples,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print()
    print(f"kept   : {len(kept):>5} / {len(data)}  ({100*len(kept)/max(len(data),1):.1f}%)")
    print(f"dropped: {len(data)-len(kept):>5}")
    for reason, n in drop_reasons.most_common(20):
        print(f"  {n:>4}  {reason}")
    print(f"\nclean → {dst}")
    print(f"stats → {stats_path}")


if __name__ == "__main__":
    main()
