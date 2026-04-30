# Aurora Llama-3.1-8B LoRA — Iteration 1 (final summary)

**Date:** 2026-04-29
**Status:** Trained and evaluated. LoRA wins 7/8 holdout questions vs base.

## Where everything is

All persistent on `/lus/flare/projects/gpu_hack/sislam6/`:

| Path | What |
|------|------|
| `work/runs/iter1/artifacts/chat/lora/` | LoRA adapter (335 MB) — load on top of base |
| `work/runs/iter1/artifacts/chat/merged_16bit/` | Merged 16-bit model (15 GB) — drop-in replacement |
| `work/runs/iter1/artifacts/chat/training_metadata/` | curated_pairs.json, train.jsonl, val.jsonl, train_qa_xpu.py, train_lora.pbs, training_log.txt |
| `work/runs/iter1/artifacts/chat/push_to_hf.sh` | One-command push to Hugging Face |
| `work/runs/iter1/data/chat/cot_curated_v2.json` | 1,317 final curated QA+CoT pairs |
| `work/runs/iter1/data/chat/train.jsonl` / `val.jsonl` | 1,186 / 131 ChatML rows |
| `work/runs/iter1/data/chat/parts20/` | Raw 20-task gpt-oss-120b outputs (3,695 pairs) |
| `work/aurora-docs/raw_full/` | 68 markdown files crawled from docs.alcf.anl.gov/aurora |
| `work/aurora-docs/lancedb/` | Chunked LanceDB (text_chunks: 2,264 rows) |
| `work/runs/iter1/eval_results.json` | Side-by-side baseline vs LoRA on 8 holdout questions |
| `work/runs/iter1/logs/` | All PBS job stdout+stderr |

## Pipeline summary

1. Crawled `https://docs.alcf.anl.gov/aurora/` → 68 markdown files (depth 4, 200 max-pages)
2. Chunked into LanceDB → 2,264 chunks → filter boilerplate → **416 clean technical chunks**
3. Generated QA+CoT via gpt-oss-120b @ ALCF Sophia (20 PBS array tasks × 200 pairs each = **3,695 pairs**)
4. Curated (length floor + Aurora-keyword + nav-trivia drop + dedup) → 1,908 → balanced (cap misc, up-weight rare topics) → **1,317**
5. Format ChatML, 90/10 split → 1,186 train / 131 val
6. LoRA fine-tune Llama-3.1-8B on 1 PVC tile (bf16, r=32, α=64, lr=2e-4 cosine, 2 epochs) — **13 min wall, train_loss 0.687**
7. Eval baseline vs LoRA on 8 holdout Aurora questions

## Eval scorecard (LoRA vs base Llama-3.1-8B-Instruct)

| # | Topic | Base verdict | LoRA verdict |
|---|-------|--------------|--------------|
| 1 | PBS submit script | wrong syntax (`-l nodes=`) | correct (`-l select=`) |
| 2 | ZE_AFFINITY_MASK | guessed `=2` | `=1` per docs |
| 3 | PyTorch on XPU | hallucinated module names, used CUDA env vars | `module load frameworks` + `.to('xpu')` |
| 4 | vLLM launch | hallucinated `aprun` (Cray ALPS) | correct `mpirun + gpu_tile_compact.sh + --tensor-parallel-size 12` |
| 5 | DAOS pool | partial (`daos_cont_create`) | partial (`daos cont create`) |
| 6 | SYCL compile | wrong (deprecated `dpcpp`) | correct (`icpx -fsycl`) |
| 7 | GPU profiler | HALLUCINATED non-existent tool | correct VTune invocation |
| 8 | Lustre flare path | wrong (NERSC path!) | close (`/flare/projects/...`, real is `/lus/flare/...`) |

**Result: LoRA wins 7-1 outright, ties on DAOS.**

## Push to Hugging Face

```bash
cd /lus/flare/projects/gpu_hack/sislam6/work/runs/iter1/artifacts/chat

# 1. Get an HF write-scope token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxx
export HF_ORG=sislam6           # or your username/org

# 2. Push both repos
./push_to_hf.sh
```

This creates two HF repos:
- `sislam6/llama31-8b-aurora-lora`     (335 MB adapter)
- `sislam6/llama31-8b-aurora-merged`   (15 GB full model)

Both have rich model cards already.

## Next iteration ideas (if revisited)

1. **Recover the 30% wasted LoRA capacity** by getting `assistant_only_loss=True` working — either patch the chat template manually with `{% generation %}` markers or switch to the official `meta-llama/...` base (gated, needs HF auth).
2. **Larger sweep**: bump to 5,000 quality pairs and 12-tile DDP (~2 min training instead of 13).
3. **Tool-use pipeline**: build `aurora_tools.json` with qsub/qstat/qdel/module/etc. and run `generator tool-pipeline` for agentic capabilities.
4. **Iterate on data quality**: drop pairs whose answer never references the source chunk; tighter dedup.

## Hardware/config bugs we hit (and fixed)

- Crawler: `--include`/`--exclude` were post-filters in `crawl4ai`-0.8 — patched to use `FilterChain + URLPatternFilter`
- Crawler: `wait_until="networkidle"` was 5-7s/page — switched to `domcontentloaded`
- Compute nodes: no direct internet — added `http(s)_proxy` to `env.sh` conditionally
- venv: Unsloth's PyPI install dragged broken `torch 2.9.0+xpu` — uninstalled to restore frameworks `torch 2.10`
- Llama-3.1: gated on HF — switched to `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror
- peft: required `torchao>=0.16` — upgraded
- TRL 1.3 API rename: `tokenizer=` → `processing_class=`, `max_seq_length=` → `max_length=`
- Trainer bug: `enable_input_require_grads()` must be AFTER `get_peft_model()` (was failing on first backward)
