# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`generator` is the synthetic-training-data + fine-tuning half of a Jarvis-CD HPC assistant project. It produces QA pairs, CoT reasoning, and tool-use examples from source material, then fine-tunes Gemma3 / FunctionGemma checkpoints on them. It is intended to run on Delta-AI (NVIDIA GH200, SLURM) with Ollama serving the teacher model locally, but everything except the SLURM scripts also runs on a workstation.

## Common commands

Install (editable):
```bash
uv pip install -e .              # local providers only (ollama, vllm)
uv pip install -e ".[all]"       # + cloud providers
uv pip install -e ".[coverage]"  # + sentence-transformers/sklearn (needed for select-coverage / multi-score diversity)
```

Tests & lint:
```bash
uv run pytest tests/ -v                       # full suite
uv run pytest tests/test_tool_use.py -v       # single file
uv run pytest tests/test_tool_use.py::test_name  # single test
uv run ruff check src/
```

The CLI is `generator` (entry point `generator.cli:main`). `generator --help` lists everything; key groups:

- **Data → JSON**: `generate`, `generate-cot`, `enhance-cot`, `enrich`, `curate`, `multi-score`, `select-coverage`, `export`, `pipeline` (full QA pipeline)
- **Tool-use data**: `tool-parse`, `tool-deps`, `tool-generate`, `tool-generate-chain`, `tool-generate-full`, `tool-execute`, `tool-curate`, `tool-evaluate`, `tool-pipeline`
- **Training (universal pipeline)**: `train-init` (wizard → `pipeline.yaml`), `train-chat`, `train-tool`, `train` (both), `ingest` (build a LanceDB from a source directory)

Smoke / scale runs on Delta go through SLURM scripts: `scripts/run_pipeline_smoke.sbatch`, `scripts/run_pipeline_scale.sbatch`, `slurm/gen_*.sbatch`. They start `ollama serve` on the compute node before calling the generator CLI — replicate that pattern for any new SLURM job.

## Architecture: two layers

The repo has **two overlapping CLI surfaces** that share clients/prompts/formatters but are otherwise independent. New work usually plugs into the training pipeline (layer 2); the standalone commands (layer 1) are still used directly for ad-hoc data work.

### Layer 1 — standalone generators (`src/generator/{qa,cot,tool}/`)

Each module owns one data-shape and is callable both from `generator.cli` and from layer 2 stages:

- `qa/` — `qa_generator.py` (Instruction Backtranslation from LanceDB chunks), `curate.py` (LLM-as-Judge), `enrich.py` (response rewriting), `multi_scorer.py` (DEITA 3D scoring), `compare.py`
- `cot/` — `cot_generator.py` (QA+reasoning from scratch), `cot_enhancer.py` (add reasoning to existing QA, with intermediate-checkpoint + resume)
- `tool/` — `tool_schemas.py` (Tool/Parameter dataclasses, `load_tools`, `save_examples`), `tool_parser.py`, `tool_generator.py`, `tool_executor.py`, `tool_curator.py`, `dependency_graph.py` (In-N-Out param graphs), `outcome_evaluator.py` (MCP-AgentBench), `coverage_selector.py` (TOUCAN clustering), `mcp_generator.py` (synthesize tool defs from a topic)

Shared infra:
- `clients/` — provider factory (`get_client(provider, cfg)`): `ollama`, `claude` (Agent SDK), `gemini`/`adk` (Google ADK), `vllm`, `openai`, `anthropic`. Legacy names `claude_sdk`/`adk` still resolve. All take a flat config dict.
- `prompt_loader.py` — loads `configs/prompts/*.yaml` into one dict; tool prompts live in `tool_prompts.yaml`.
- `formatters.py` — ChatML / Alpaca / ShareGPT / JSONL writers.

`configs/config.yaml` is the global config consumed by the layer-1 CLI. The `llm:` block is provider-namespaced (`llm.ollama.model`, `llm.claude.model`, …) and `_extract_llm_config` in `cli.py` flattens whichever provider is active, applying `--provider` / `--model` overrides on top.

### Layer 2 — universal training pipeline (`src/generator/training/`)

Driven by a single `pipeline.yaml` with `chat:` and `tool:` sections (see `examples/pipeline-smoke.yaml`, `examples/pipeline-scale.yaml`). The CLI commands `train-init`, `train-chat`, `train-tool`, `train`, `ingest` are thin wrappers; the real wiring is here:

- `config.py` — `GLOBAL_DEFAULTS`, `CHAT_DEFAULTS`, `TOOL_DEFAULTS`, deep-merge `load_config`, interactive `wizard`. Edits to defaults flow through to every generated pipeline.
- `hardware.py` — `detect()` returns `{has_cuda, gpu_name, gpu_mem_gb, has_slurm, ...}`; pipelines log a summary on start.
- `runner.py` — `Stage(name, description, inputs, outputs, fn, skip_if_disabled_path)` + `run_stages()`. **Idempotency is mtime-based**: a stage is skipped when all `outputs` exist *and* are newer than every `inputs`. `--force` overrides this; `--from <name>` / `--only <name>` slice the DAG. State is journalled to `<output_dir>/logs/{chat,tool}_state.json`.
- `pipelines.py` — composes the stage lists for chat and tool. Chat: `gen-cot → fix-cot → curate → prep-chat → train-chat → eval-chat`. Tool: `gen-tool → schema-filter → prep-tool → train-tool → eval-tool`. `run_both` runs chat then tool serially.
- `stages/` — one file per stage. Most stages are short adapters that call into layer-1 modules or shell out via `_subprocess.py`. `prep_tool.py` converts to the FunctionGemma format; `schema_filter.py` drops rows whose tool-call arg shapes don't match the tool registry.

When adding a stage: declare its inputs/outputs as `Path` objects (the runner uses these for both caching *and* the `--from` skip logic), and expose a `<name>_data_paths(cfg)` or `<name>_artifact_paths(cfg)` helper so other stages can reference its outputs without recomputing the layout.

## Conventions worth knowing

- **One topic = one folder.** `cfg.topic` drives `cfg.output_dir`: if `output_dir` is unset, it resolves to `./runs/<topic-slug>` via `slugify()` in `src/generator/training/config.py`. Slugging strips punctuation, lowercases, and collapses whitespace/underscores to dashes (`"Jarvis-CD HPC workflows"` → `./runs/jarvis-cd-hpc-workflows`). Set `output_dir` explicitly in the YAML to override.
- **LanceDB tables**: `text_chunks` (prose) and `code_chunks` (code) are the canonical table names. `generate` accepts `--table` multiple times to produce a unified dataset; `code_chunks` switches to a code-specific prompt automatically.
- **Output layout**: all training-pipeline artifacts go under `cfg.output_dir`. Reports → `<output_dir>/reports/`, logs/state → `<output_dir>/logs/`, model checkpoints → adapter+merged dirs returned by `chat_artifact_paths` / `tool_artifact_paths`.
- **Tools registry**: `configs/tools/jarvis_tools.{json,yaml}` is the source of truth for the 29 Jarvis MCP tools used in tool-use generation. JSON is what the generator loads; the YAML is the human-edited copy.
- **Ollama tuning**: workers should match `OLLAMA_NUM_PARALLEL` (see `configs/config.yaml` comments). On GH200 with `gpt-oss:20b`, 4–8 workers is typical; higher only helps if VRAM allows.
- **`uv run` vs bare**: all examples use `uv run generator …`; the bare `generator` entrypoint works too once the venv is active (`.venv-delta/` on Delta).
- **Comments / docs**: this repo has heavy existing docstrings and README sections — don't add more unless you're documenting genuinely non-obvious behavior. The `docs/` directory holds long-form design docs; keep new design rationale there, not inline.
