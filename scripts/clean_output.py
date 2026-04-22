#!/usr/bin/env python3
"""
Post-process v7 tool-use outputs to fix quality issues.

Fixes:
1. Drops examples with empty {} expected_result (teaches model to output empty returns)
2. Fixes {"error": "..."} in multi-step → reclassifies as error-recovery pattern
3. Fills missing status values in pipeline tool returns
4. Optional: replaces over-represented pipeline names with diverse alternatives
5. Semantic deduplication via sentence-transformer similarity

Usage:
    python scripts/clean_output.py outputs/v7_10k/jarvis_v7_10000.json \
        -o outputs/v7_10k/jarvis_v7_10000_clean.json \
        --dedupe --diversify-names
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

# Real return shapes from jarvis_handler.py (ground truth)
PIPELINE_SHAPES = {
    'create_pipeline': {'pipeline_id', 'status'},
    'load_pipeline': {'pipeline_id', 'status'},
    'update_pipeline': {'pipeline_id', 'status'},
    'build_pipeline_env': {'pipeline_id', 'status'},
    'run_pipeline': {'pipeline_id', 'status'},
    'destroy_pipeline': {'pipeline_id', 'status'},
    'append_pkg': {'pipeline_id', 'appended'},
    'configure_pkg': {'pipeline_id', 'configured'},
    'get_pkg_config': {'pipeline_id', 'pkg_id', 'config'},
    'unlink_pkg': {'pipeline_id', 'unlinked'},
    'remove_pkg': {'pipeline_id', 'removed'},
}

STATUS_VALUES = {
    'create_pipeline': 'created',
    'load_pipeline': 'loaded',
    'update_pipeline': 'updated',
    'build_pipeline_env': 'environment_built',
    'run_pipeline': 'running',
    'destroy_pipeline': 'destroyed',
}

MANAGER_TOOLS = {
    'jm_create_config', 'jm_load_config', 'jm_save_config',
    'jm_set_hostfile', 'jm_bootstrap_from', 'jm_bootstrap_list',
    'jm_reset', 'jm_list_pipelines', 'jm_cd',
    'jm_list_repos', 'jm_add_repo', 'jm_remove_repo',
    'jm_promote_repo', 'jm_get_repo', 'jm_construct_pkg',
    'jm_graph_show', 'jm_graph_build', 'jm_graph_modify',
}

# Pool of diverse pipeline name templates for diversification
DIVERSE_NAMES_POOL = [
    # Scientific
    'climate_sim', 'ocean_dynamics', 'galaxy_formation', 'protein_folding',
    'molecular_dynamics', 'cfd_solver', 'plasma_physics', 'seismic_analysis',
    'quantum_chem', 'genomics_pipe', 'neural_trainer', 'deep_inference',
    'bioinformatics', 'cosmology_sim', 'fluid_solver', 'astro_pipeline',
    # HPC benchmarks
    'io_bench_a', 'io_bench_b', 'perf_eval', 'throughput_test',
    'latency_probe', 'mem_bandwidth', 'network_stress', 'gpu_compute',
    'hpl_linpack', 'stream_test', 'hpcg_solve', 'osu_latency',
    # ML/AI
    'llm_finetune', 'vision_train', 'rl_experiment', 'gan_pipeline',
    'transformer_ft', 'bert_pretrain', 'diffusion_gen', 'rag_index',
    # Domain-specific
    'weather_forecast', 'market_sim', 'epidemic_model', 'traffic_opt',
    'power_grid', 'logistics_net', 'drug_discovery', 'materials_design',
    # Generic-with-versions
    'analytics_v2', 'pipeline_alpha', 'dev_staging', 'prod_release',
    'experiment_42', 'batch_job_7', 'research_run', 'hotfix_test',
]

# Names that are too common and should be replaced
OVERUSED = {'pipeline_123', 'performance_test', 'demo-pipeline', 'test_pipeline',
            'test-pipeline', 'test_pipe', 'pipeline_001', 'pipeline_01', 'ml_pipeline',
            'data_analysis'}


def fix_shape(step: Dict, method: str) -> bool:
    """
    Try to fix/validate a step's expected_result in place.
    Returns True if step is usable, False if it should trigger example drop.
    """
    tool = step.get('tool', '')
    er = step.get('expected_result')

    # Skip validation for chain-first (uses descriptive strings)
    if method == 'chain_first':
        return True

    # Error-recovery: preserve all structured failure/recovery data
    # The pattern is already well-formed (100% have failure step); be lenient on recovery shapes
    if method == 'error_recovery':
        # Accept all — the structure (failure + recovery) is the training signal,
        # not the exact recovery-step result shape
        return True

    # Empty {} — LLM was lazy, DROP the example
    if er == {} or er is None:
        return False

    # Pipeline tool with error dict → treat as failure, re-tag
    if tool in PIPELINE_SHAPES and isinstance(er, dict):
        if 'error' in er and len(er) <= 2:
            # {"error": "..."} means this step actually failed
            # Move error to actual_result and clear expected_result
            step['actual_result'] = er.get('error', '')
            step['status'] = 'failure'
            step['error_message'] = er.get('error', '')[:200]
            step['expected_result'] = {
                'pipeline_id': step.get('args', {}).get('pipeline_id', '?'),
                'status': STATUS_VALUES.get(tool, 'ok'),
            } if tool in STATUS_VALUES else {}
            return True

        # Check shape match
        expected = PIPELINE_SHAPES[tool]
        actual = set(er.keys())
        if actual != expected:
            return False  # wrong shape, drop

        # Fix status value if wrong
        if tool in STATUS_VALUES:
            correct = STATUS_VALUES[tool]
            if er.get('status') != correct:
                er['status'] = correct
        return True

    # Manager tools should return string or list, not dict
    if tool in MANAGER_TOOLS and isinstance(er, dict):
        return False  # drop — manager tools don't return dicts

    return True


def is_valid_example(example: Dict) -> bool:
    """Final validity check after fixing."""
    if len(example.get('instruction', '')) < 15:
        return False
    if not example['solution']['reasoning_path']:
        return False
    for step in example['solution']['reasoning_path']:
        if not step.get('tool'):
            return False
        if not step.get('thought') or len(step['thought']) < 10:
            return False
    return True


def clean_shapes(data: List[Dict]) -> (List[Dict], Dict):
    """Apply shape fixes and drop unfixable examples."""
    cleaned = []
    stats = {'total': len(data), 'dropped_shape': 0, 'fixed_error_dict': 0, 'dropped_empty': 0, 'kept': 0}

    for ex in data:
        method = ex['solution']['method']
        all_steps_ok = True

        for step in ex['solution']['reasoning_path']:
            er_before = step.get('expected_result')
            ok = fix_shape(step, method)
            if not ok:
                all_steps_ok = False
                if er_before == {} or er_before is None:
                    stats['dropped_empty'] += 1
                else:
                    stats['dropped_shape'] += 1
                break
            # Count fixes
            if isinstance(er_before, dict) and 'error' in er_before and step.get('status') == 'failure':
                stats['fixed_error_dict'] += 1

        if all_steps_ok and is_valid_example(ex):
            cleaned.append(ex)
            stats['kept'] += 1

    return cleaned, stats


def diversify_names(data: List[Dict]) -> int:
    """
    Replace over-represented pipeline names with diverse pool names.
    Uses per-example random mapping so each occurrence of an overused name
    gets replaced consistently within one example but diversely across the dataset.
    """
    # Count current usage
    name_usage = Counter()
    for ex in data:
        for step in ex['solution']['reasoning_path']:
            args = step.get('args', {})
            if 'pipeline_id' in args and isinstance(args['pipeline_id'], str):
                name_usage[args['pipeline_id']] += 1

    # Find overused names (appearing > threshold times)
    threshold = max(30, len(data) // 200)
    overused_names = {n for n, c in name_usage.items() if c > threshold and n in OVERUSED}

    if not overused_names:
        return 0

    replaced = 0
    rng = random.Random(42)  # reproducible

    for ex in data:
        # Each example gets its own fresh mapping from the pool
        ex_mapping = {}
        pool_shuffled = list(DIVERSE_NAMES_POOL)
        rng.shuffle(pool_shuffled)
        pool_iter = iter(pool_shuffled)

        # Find which overused names this example uses
        def get_replacement(old_name):
            if old_name not in ex_mapping:
                try:
                    ex_mapping[old_name] = next(pool_iter)
                except StopIteration:
                    # Refresh pool if exhausted
                    rng.shuffle(pool_shuffled)
                    ex_mapping[old_name] = pool_shuffled[0]
            return ex_mapping[old_name]

        # Replace in instruction, args, thoughts, final_answer
        for old in overused_names:
            if old not in ex['instruction'] and not any(
                old in str(s.get('args', {}).get('pipeline_id', ''))
                for s in ex['solution']['reasoning_path']
            ):
                continue

            new = get_replacement(old)

            if old in ex['instruction']:
                ex['instruction'] = ex['instruction'].replace(old, new)
                replaced += 1
            fa = ex['solution'].get('final_answer', '')
            if old in fa:
                ex['solution']['final_answer'] = fa.replace(old, new)
            for step in ex['solution']['reasoning_path']:
                if step.get('thought') and old in step['thought']:
                    step['thought'] = step['thought'].replace(old, new)
                args = step.get('args', {})
                if args.get('pipeline_id') == old:
                    args['pipeline_id'] = new

    return replaced


def dedupe_by_instruction(data: List[Dict]) -> List[Dict]:
    """Remove exact duplicate instructions (first 80 chars)."""
    seen = set()
    out = []
    for ex in data:
        key = ex['instruction'][:80]
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def semantic_dedupe(data: List[Dict], threshold: float = 0.95) -> List[Dict]:
    """
    Drop semantic near-duplicates within each method (single/multi/chain/error-recovery).
    Dedupe per-method so error-recovery patterns don't compete with chain-first.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("sentence-transformers not available, skipping semantic dedupe")
        return data

    # Group by method
    by_method: Dict[str, List[int]] = {}
    for i, e in enumerate(data):
        m = e['solution']['method']
        by_method.setdefault(m, []).append(i)

    print(f"  Method counts: { {m: len(v) for m, v in by_method.items()} }")
    print(f"  Computing embeddings for {len(data)} instructions...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([e['instruction'] for e in data], show_progress_bar=False, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    keep = [True] * len(data)

    for method, indices in by_method.items():
        for a, i in enumerate(indices):
            if not keep[i]:
                continue
            for j in indices[a+1:]:
                if keep[j] and float(embs[i] @ embs[j]) > threshold:
                    keep[j] = False

    dropped_by_method = {}
    for method, indices in by_method.items():
        dropped_by_method[method] = sum(1 for i in indices if not keep[i])
    print(f"  Dropped by method: {dropped_by_method}")

    return [e for e, k in zip(data, keep) if k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='Input JSON file')
    ap.add_argument('-o', '--output', required=True, help='Output JSON file')
    ap.add_argument('--dedupe', action='store_true', help='Drop exact duplicate instructions')
    ap.add_argument('--semantic-dedupe', action='store_true', help='Drop semantic near-duplicates')
    ap.add_argument('--semantic-threshold', type=float, default=0.95,
                    help='Cosine similarity threshold for semantic dedupe (default 0.95)')
    ap.add_argument('--diversify-names', action='store_true',
                    help='Replace over-represented pipeline names')
    args = ap.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    original_count = len(data)
    print(f"  {original_count} examples loaded")

    # Step 1: Shape fixing
    print("\n[1/4] Fixing shapes...")
    data, shape_stats = clean_shapes(data)
    print(f"  Kept: {shape_stats['kept']}")
    print(f"  Dropped (empty result): {shape_stats['dropped_empty']}")
    print(f"  Dropped (wrong shape): {shape_stats['dropped_shape']}")
    print(f"  Fixed (error dict → failure): {shape_stats['fixed_error_dict']}")

    # Step 2: Deduplication
    if args.dedupe:
        print("\n[2/4] Removing exact duplicates...")
        before = len(data)
        data = dedupe_by_instruction(data)
        print(f"  Dropped: {before - len(data)}, kept: {len(data)}")

    # Step 3: Semantic deduplication
    if args.semantic_dedupe:
        print(f"\n[3/4] Semantic dedupe (threshold={args.semantic_threshold})...")
        before = len(data)
        data = semantic_dedupe(data, args.semantic_threshold)
        print(f"  Dropped: {before - len(data)}, kept: {len(data)}")

    # Step 4: Name diversification
    if args.diversify_names:
        print("\n[4/4] Diversifying pipeline names...")
        n = diversify_names(data)
        print(f"  Replacements made: {n}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n═══ DONE ═══")
    print(f"  {original_count} → {len(data)} examples ({100*len(data)/original_count:.1f}% retained)")
    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    main()
