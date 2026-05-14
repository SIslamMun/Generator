#!/usr/bin/env python3
"""Audit generated NDP tool-use examples.

Usage: python audit_examples.py <examples.json>
"""
import json, sys, re
from collections import Counter
from pathlib import Path

NDP_TOOLS = {"list_organizations", "search_datasets", "get_dataset_details"}
NDP_SERVERS = {"local", "global", "pre_ckan"}
# Real values that should show up if examples are well-grounded
KNOWN_ORG_SLUGS = {
    "nasa-firms", "nasa-werk", "ibm-nasa-geospatial",
    "national-aeronautics-and-space-administration-nasa",
    "california-landscape-metrics", "cal-oes", "burnpro3d",
    "ai-genomics-at-scale", "aquasteady",
}
KNOWN_DATASET_IDS = {
    "1f29f678-924c-456d-95d4-aa0bc7de7037",
    "3264e7ee-ef6d-42d5-b722-8ad39670cf3d",
}
# Jarvis-vocab signals we want to flag (prompt-bleed risk)
JARVIS_BLEED = re.compile(
    r"\b(pipeline|package|pkg|configure_pkg|jarvis|hermes|gromacs|lammps|hdf5-ng|build_env|jm_create|append_pkg)\b",
    re.IGNORECASE,
)


def main(path):
    data = json.loads(Path(path).read_text())
    n = len(data)
    print(f"\n══ {path}  ({n} examples) ══\n")

    methods = Counter(ex["solution"]["method"] for ex in data)
    tools_used = Counter()
    server_values = Counter()
    invalid_tools = []
    bad_servers = []
    empty_reasoning = 0
    instr_lens = []
    n_grounded_org = 0
    n_grounded_id = 0
    n_jarvis_bleed = 0
    bleed_samples = []
    failures_in_path = 0
    multi_step_lengths = []

    for i, ex in enumerate(data):
        instr = ex.get("instruction", "")
        instr_lens.append(len(instr))
        if JARVIS_BLEED.search(instr):
            n_jarvis_bleed += 1
            if len(bleed_samples) < 3:
                bleed_samples.append(instr)
        sol = ex["solution"]
        path_steps = sol.get("reasoning_path", [])
        if not path_steps:
            empty_reasoning += 1
            continue
        if len(path_steps) > 1:
            multi_step_lengths.append(len(path_steps))
        for s in path_steps:
            tname = s.get("tool", "")
            tools_used[tname] += 1
            if tname not in NDP_TOOLS:
                invalid_tools.append((i, tname))
            args = s.get("args") or {}
            sv = args.get("server")
            if sv is not None:
                server_values[str(sv)] += 1
                if sv not in NDP_SERVERS:
                    bad_servers.append((i, sv))
            # check grounding: are real slugs / ids ever used?
            args_blob = json.dumps(args)
            if any(slug in args_blob for slug in KNOWN_ORG_SLUGS):
                n_grounded_org += 1
            if any(did in args_blob for did in KNOWN_DATASET_IDS):
                n_grounded_id += 1
            if s.get("status") == "failure":
                failures_in_path += 1

    print("── METHOD DISTRIBUTION (target: s=0.45 m=0.30 c=0.10 e=0.15) ──")
    for m, c in methods.most_common():
        pct = 100 * c / n if n else 0
        print(f"  {m:18s} {c:4d}  ({pct:5.1f}%)")

    print("\n── TOOL USAGE ──")
    for t, c in tools_used.most_common():
        marker = "" if t in NDP_TOOLS else "  ← INVALID"
        print(f"  {t:30s} {c:4d}{marker}")

    print("\n── server PARAM ──")
    for v, c in server_values.most_common():
        marker = "" if v in NDP_SERVERS else "  ← INVALID"
        print(f"  {v!r:20s} {c:4d}{marker}")

    print(f"\n── GROUNDING (looking for known real NDP values) ──")
    print(f"  steps using a real org slug:    {n_grounded_org}")
    print(f"  steps using a real dataset UUID:{n_grounded_id}")

    print(f"\n── QUALITY FLAGS ──")
    print(f"  empty reasoning path:           {empty_reasoning}")
    print(f"  invalid tool names (total):     {len(invalid_tools)}")
    print(f"  invalid server values:          {len(bad_servers)}")
    print(f"  failure-status steps:           {failures_in_path}")
    if multi_step_lengths:
        avg = sum(multi_step_lengths) / len(multi_step_lengths)
        print(f"  multi-step avg length:          {avg:.2f}  (n={len(multi_step_lengths)})")
    print(f"  instruction length avg:         {sum(instr_lens)/max(len(instr_lens),1):.0f} chars")

    print(f"\n── JARVIS-VOCAB BLEED IN INSTRUCTIONS ──")
    print(f"  count: {n_jarvis_bleed} / {n}")
    for s in bleed_samples:
        print(f"  └─ {s[:120]}")

    # Show 2 samples from each method
    print("\n── SAMPLES ──")
    seen = set()
    for ex in data:
        m = ex["solution"]["method"]
        if m in seen:
            continue
        seen.add(m)
        print(f"\n  [{m}]")
        print(f"    instruction: {ex['instruction'][:140]}")
        path = ex["solution"]["reasoning_path"]
        for s in path[:2]:
            tool = s.get("tool", "?")
            args = s.get("args", {})
            print(f"    step {s.get('step','?')} → {tool}({json.dumps(args)[:80]})")
            if s.get("status") == "failure":
                print(f"      ✗ status=failure: {str(s.get('actual_result',''))[:100]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: audit_examples.py <examples.json>")
        sys.exit(1)
    main(sys.argv[1])
