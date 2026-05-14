"""Synthesize targeted delta examples for the patterns the v7 model failed on.

The v7 test showed weak chain_first/error_recovery handling for *short* chains
(2-3 steps) and for "create-then-act" and "append-to-missing" patterns. The
existing v7 raw data is heavily skewed toward 5-step chains, so this script
fills the gap with ~500 short, schema-correct examples in the same raw shape
that `convert_to_functiongemma.py` expects.

Output: a JSON list, each entry shaped like the v7 raw entries
  {instruction, solution: {method, reasoning_path: [{tool, args, thought,
  status, actual_result|error_message}, ...], final_answer}}
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).with_name("v7_delta_raw.json")
rng = random.Random(42)

# Realistic HPC pipeline names
PIPELINE_NAMES = [
    "astro_sim", "climate_run", "molec_dyn", "cfd_solve", "gene_assembly",
    "bench_v2", "quantum_chem", "ai_train", "data_analysis", "genome_pipe",
    "weather_fcst", "protein_fold", "cosmology", "fluid_flow", "ml_hyperparam",
    "neutrino_run", "seismic_fwi", "particle_transport", "combustion_sim",
    "ocean_model", "io_bench", "checkpoint_test", "tensor_ops", "sparse_solve",
    "nucleosynthesis", "plasma_phys", "mesh_refine", "mhd_sim", "radiation_transport",
    "fresh_pipeline", "my_workflow", "hpc_demo", "release_bench", "prod_run",
]

PKG_TYPES = ["ior", "mdtest", "hdf5", "orangefs", "darshan", "stream", "osu", "elbencho"]

CONFIG_KEYS = [
    ("nprocs", [4, 8, 16, 32, 64, 128]),
    ("block_size", ["256k", "1m", "4m", "16m"]),
    ("transfer_size", ["4k", "64k", "1m"]),
    ("iterations", [10, 100, 1000]),
    ("read_write_ratio", [0.5, 0.7, 0.9]),
    ("threads", [1, 2, 4, 8]),
]

MACHINES = ["summit", "frontier", "perlmutter", "aurora", "delta", "polaris"]


def pick_config_args() -> dict:
    n = rng.randint(1, 3)
    out = {}
    for k, vs in rng.sample(CONFIG_KEYS, n):
        out[k] = rng.choice(vs)
    return out


def step(tool, args=None, thought="", status="success", actual=None):
    return {
        "tool": tool,
        "args": args or {},
        "thought": thought,
        "status": status,
        "actual_result": actual if actual is not None else {"ok": True},
    }


def err_step(tool, args, thought, err_msg):
    return {
        "tool": tool,
        "args": args,
        "thought": thought,
        "status": "failure",
        "error_message": err_msg,
    }


def pattern_a_load_then_cd():
    """Load existing pipeline → cd to it. (2-step chain)"""
    pid = rng.choice(PIPELINE_NAMES)
    instr = rng.choice([
        f"Load the pipeline {pid} and make it my current pipeline.",
        f"Load pipeline {pid} and switch to it.",
        f"Open the pipeline {pid} and set it as the active one.",
        f"Load {pid} so I can work on it, and make it current.",
        f"Switch me into the existing pipeline {pid}.",
    ])
    path = [
        step("load_pipeline", {"pipeline_id": pid},
             f"The user wants to work with an existing pipeline named {pid}. I'll load it first.",
             actual={"pipeline_id": pid, "status": "loaded"}),
        step("jm_cd", {"pipeline_id": pid},
             f"Now set {pid} as the current pipeline context so subsequent operations target it.",
             actual=f"Current pipeline set to '{pid}'"),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "chain_first",
            "reasoning_path": path,
            "final_answer": f"The pipeline {pid} is loaded and set as the current working pipeline.",
        },
    }


def pattern_b_create_cd_attach():
    """Create pipeline → cd → attach package. (3-step chain)"""
    pid = rng.choice(PIPELINE_NAMES)
    pkg = rng.choice(PKG_TYPES)
    instr = rng.choice([
        f"Create a pipeline called {pid}, switch to it, and attach a {pkg} package.",
        f"Set up a new pipeline {pid}, make it current, and add a {pkg} benchmark.",
        f"Start a new pipeline named {pid}, cd into it, and attach the {pkg} package.",
        f"Create {pid}, switch me in, and hook up a {pkg} package.",
    ])
    path = [
        step("create_pipeline", {"pipeline_id": pid},
             f"First I need to create the pipeline {pid} since it doesn't exist yet.",
             actual={"pipeline_id": pid, "status": "created"}),
        step("jm_cd", {"pipeline_id": pid},
             f"Set {pid} as the current context so the next package is attached to it.",
             actual=f"Current pipeline set to '{pid}'"),
        step("append_pkg", {"pipeline_id": pid, "pkg_type": pkg},
             f"Now attach the {pkg} package to {pid}.",
             actual={"pipeline_id": pid, "appended": pkg}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "chain_first",
            "reasoning_path": path,
            "final_answer": f"Pipeline {pid} has been created, set as current, and the {pkg} package is attached.",
        },
    }


def pattern_c_create_cd_attach_configure():
    """Create → cd → attach → configure (4-step chain with dict extra_args)."""
    pid = rng.choice(PIPELINE_NAMES)
    pkg = rng.choice(PKG_TYPES)
    cfg = pick_config_args()
    cfg_desc = ", ".join(f"{k}={v}" for k, v in cfg.items())
    instr = rng.choice([
        f"Create a pipeline called {pid}, switch to it, attach an {pkg} package, and configure it with {cfg_desc}.",
        f"Set up pipeline {pid}, make it current, add a {pkg} benchmark, and configure {cfg_desc}.",
        f"Create {pid}, cd in, attach {pkg}, then configure with {cfg_desc}.",
    ])
    path = [
        step("create_pipeline", {"pipeline_id": pid},
             f"Create pipeline {pid} since it's new.",
             actual={"pipeline_id": pid, "status": "created"}),
        step("jm_cd", {"pipeline_id": pid},
             f"Set {pid} as current so subsequent package operations target it.",
             actual=f"Current pipeline set to '{pid}'"),
        step("append_pkg", {"pipeline_id": pid, "pkg_type": pkg},
             f"Attach the {pkg} package to {pid}.",
             actual={"pipeline_id": pid, "appended": pkg}),
        step("configure_pkg", {"pipeline_id": pid, "pkg_id": pkg, "extra_args": cfg},
             f"Configure {pkg} with the requested settings.",
             actual={"pipeline_id": pid, "configured": pkg}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "chain_first",
            "reasoning_path": path,
            "final_answer": f"Pipeline {pid} is created with a configured {pkg} package ({cfg_desc}).",
        },
    }


def pattern_d_error_append_missing():
    """Append to missing pipeline → error → create → retry append."""
    pid = rng.choice(PIPELINE_NAMES)
    pkg = rng.choice(PKG_TYPES)
    instr = rng.choice([
        f"Append an {pkg} package to pipeline {pid} — if the pipeline doesn't exist, create it first.",
        f"Add a {pkg} package to {pid}. Create the pipeline first if it's missing.",
        f"Attach {pkg} to pipeline {pid}; create {pid} beforehand if it doesn't already exist.",
    ])
    path = [
        err_step("append_pkg", {"pipeline_id": pid, "pkg_type": pkg},
                 f"Try appending {pkg} to {pid} directly first.",
                 f"Pipeline '{pid}' not found"),
        step("create_pipeline", {"pipeline_id": pid},
             f"The append failed because {pid} doesn't exist. I'll create it first.",
             actual={"pipeline_id": pid, "status": "created"}),
        step("append_pkg", {"pipeline_id": pid, "pkg_type": pkg},
             f"Now that {pid} exists, retry attaching {pkg}.",
             actual={"pipeline_id": pid, "appended": pkg}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "error_recovery",
            "reasoning_path": path,
            "final_answer": f"Pipeline {pid} was missing, so I created it and then attached the {pkg} package.",
        },
    }


def pattern_e_error_load_missing():
    """Load missing → error → create."""
    pid = rng.choice(PIPELINE_NAMES)
    instr = rng.choice([
        f"Load the pipeline {pid} so I can use it; if it doesn't exist, create it first.",
        f"Open pipeline {pid} — if missing, create it and load.",
    ])
    path = [
        err_step("load_pipeline", {"pipeline_id": pid},
                 f"Try to load the existing pipeline {pid}.",
                 f"Pipeline '{pid}' not found"),
        step("create_pipeline", {"pipeline_id": pid},
             f"Load failed because {pid} doesn't exist yet, so I'll create it.",
             actual={"pipeline_id": pid, "status": "created"}),
        step("jm_cd", {"pipeline_id": pid},
             f"Switch into {pid} now that it exists.",
             actual=f"Current pipeline set to '{pid}'"),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "error_recovery",
            "reasoning_path": path,
            "final_answer": f"Pipeline {pid} didn't exist, so I created it and set it as current.",
        },
    }


def pattern_f_configure_single():
    """Single tool: configure_pkg with dict extra_args (teaches dict grammar)."""
    pid = rng.choice(PIPELINE_NAMES)
    pkg = rng.choice(PKG_TYPES)
    cfg = pick_config_args()
    cfg_desc = ", ".join(f"{k}={v}" for k, v in cfg.items())
    instr = rng.choice([
        f"Configure the {pkg} package in {pid} with {cfg_desc}.",
        f"Set {cfg_desc} on the {pkg} package in pipeline {pid}.",
        f"Reconfigure {pkg} in {pid} using {cfg_desc}.",
    ])
    path = [
        step("configure_pkg", {"pipeline_id": pid, "pkg_id": pkg, "extra_args": cfg},
             f"The user wants to configure the {pkg} package in {pid} with specific settings.",
             actual={"pipeline_id": pid, "configured": pkg}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "single",
            "reasoning_path": path,
            "final_answer": f"The {pkg} package in {pid} is now configured ({cfg_desc}).",
        },
    }


def pattern_g_graph_build():
    """Single tool: jm_graph_build with float net_sleep."""
    sleep_s = rng.choice([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    verb = rng.choice(["Build", "Rebuild", "Construct"])
    instr = rng.choice([
        f"{verb} the resource graph with a {sleep_s}-second sleep between operations.",
        f"{verb} the resource graph using {sleep_s} seconds net sleep.",
        f"Run {verb.lower()} on the resource graph, spacing ops by {sleep_s} seconds.",
    ])
    path = [
        step("jm_graph_build", {"net_sleep": float(sleep_s)},
             f"The user wants to build the resource graph with {sleep_s}s sleep between operations.",
             actual="Resource graph built."),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "single",
            "reasoning_path": path,
            "final_answer": f"The resource graph is built using a {sleep_s}-second inter-op sleep.",
        },
    }


def pattern_h_multi_create_destroy():
    """Multi: create one + destroy another."""
    a = rng.choice(PIPELINE_NAMES); b = rng.choice([p for p in PIPELINE_NAMES if p != a])
    instr = rng.choice([
        f"Create pipeline {a} and destroy pipeline {b}.",
        f"Make a new pipeline {a}, then get rid of pipeline {b}.",
        f"Create {a}, and clean up {b} at the same time.",
    ])
    path = [
        step("create_pipeline", {"pipeline_id": a},
             f"First create the new pipeline {a}.",
             actual={"pipeline_id": a, "status": "created"}),
        step("destroy_pipeline", {"pipeline_id": b},
             f"Now destroy the old pipeline {b} as requested.",
             actual={"pipeline_id": b, "status": "destroyed"}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "multi",
            "reasoning_path": path,
            "final_answer": f"Pipeline {a} has been created and pipeline {b} has been destroyed.",
        },
    }


def pattern_i_create_pipeline_only():
    """Single: explicit 'create new' phrasing (disambiguates from 'load')."""
    pid = rng.choice(PIPELINE_NAMES)
    instr = rng.choice([
        f"Start a brand new pipeline called {pid}.",
        f"Create a fresh pipeline {pid}.",
        f"Set up a new empty pipeline named {pid}.",
        f"Initialize a new pipeline called {pid}.",
        f"Make a new pipeline {pid} from scratch.",
    ])
    path = [
        step("create_pipeline", {"pipeline_id": pid},
             f"The user explicitly wants a new pipeline {pid}, so create_pipeline is the right tool (not load_pipeline).",
             actual={"pipeline_id": pid, "status": "created"}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "single",
            "reasoning_path": path,
            "final_answer": f"A new pipeline {pid} has been created.",
        },
    }


def pattern_j_load_pipeline_only():
    """Single: explicit 'load existing' phrasing."""
    pid = rng.choice(PIPELINE_NAMES)
    instr = rng.choice([
        f"Load the existing pipeline {pid}.",
        f"Open the pipeline {pid} that's already set up.",
        f"Reload my pipeline {pid}.",
        f"Bring up pipeline {pid} from saved state.",
    ])
    path = [
        step("load_pipeline", {"pipeline_id": pid},
             f"The user said 'existing'/'load' — use load_pipeline, not create_pipeline.",
             actual={"pipeline_id": pid, "status": "loaded"}),
    ]
    return {
        "instruction": instr,
        "solution": {
            "method": "single",
            "reasoning_path": path,
            "final_answer": f"The pipeline {pid} has been loaded.",
        },
    }


PATTERNS = [
    (pattern_a_load_then_cd, 80),
    (pattern_b_create_cd_attach, 100),
    (pattern_c_create_cd_attach_configure, 100),
    (pattern_d_error_append_missing, 80),
    (pattern_e_error_load_missing, 60),
    (pattern_f_configure_single, 60),
    (pattern_g_graph_build, 40),
    (pattern_h_multi_create_destroy, 40),
    (pattern_i_create_pipeline_only, 60),
    (pattern_j_load_pipeline_only, 40),
]


def main():
    examples = []
    for fn, n in PATTERNS:
        for _ in range(n):
            examples.append(fn())
    rng.shuffle(examples)
    OUT.write_text(json.dumps(examples, indent=2))
    from collections import Counter
    c = Counter(ex["solution"]["method"] for ex in examples)
    print(f"wrote {len(examples)} examples → {OUT}")
    print(f"by method: {dict(c)}")


if __name__ == "__main__":
    main()
