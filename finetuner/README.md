# finetuner — multi-backend Phase 6 fine-tuning

A thin wrapper that dispatches a fine-tune job to one of three back-ends,
sharing a single **model-agnostic** parameter set. Implements the design in
[grc-iit/Phagocyte#4](https://github.com/grc-iit/Phagocyte/issues/4).

> **Status:** model-universal — tested end-to-end on the Delta cluster across
> the `unsloth`, `hf` and `ollama` backends and multiple model families
> (Llama, Qwen, Gemma, Granite, Nemotron).

## The core idea — change the model, everything adapts

The caller (CLI or the Phagocyte web UI) supplies **only the flat,
model-independent config** it already collects: backend, model, epochs, lr,
LoRA rank/alpha/dropout, batch size, etc. Everything **model-specific** is
auto-resolved from the chosen model by `model_profiles.resolve()`:

| Auto-resolved per model | from |
|---|---|
| Unsloth loader class — `FastLanguageModel` / `FastModel` | `model_type` (vision/multimodal → `FastModel`) |
| LoRA `target_modules` | architecture — Mamba-hybrids (Nemotron) auto-add `in_proj`/`out_proj` |
| `train_on_responses_only` turn markers | chat-template family (ChatML / Gemma / Granite / Llama / Mistral) |
| `trust_remote_code` | always on for custom architectures |
| tool-catalog rendering (`tools=`) | enabled when a dataset row carries `tools` |

So **selecting a different model in the UI needs no other input** — the
finetuner reads that model's `AutoConfig` + chat template and adapts. Unknown
models fall back to safe defaults (language loader, standard LoRA targets,
markers sniffed from the chat template) — it degrades gracefully, never crashes.

## Back-ends

| `--backend` | What it does | Needs |
|---|---|---|
| `unsloth` | LoRA training via Unsloth. Loader/targets/masking auto-resolved. Fast, low-VRAM. Response-only masking. | CUDA GPU, `unsloth` |
| `hf` | LoRA training via `transformers` + `peft` + TRL. Portable (no custom kernels). Targets auto-resolved; trains full-sequence (no masking). | CUDA GPU, `transformers`, `peft` |
| `ollama` | **Packages** an already-trained checkpoint into an Ollama model via a generated `Modelfile`. Does **not** train. | `ollama` on PATH |

The three form a pipeline: **`unsloth`/`hf` train → `ollama` deploys**.

## Layout

```
finetuner/
├── config.py            FinetuneConfig — the shared flat parameter set
├── model_profiles.py    resolve() — per-model deltas from the model itself
├── cli.py               `finetuner run` / `finetuner list`
└── backends/
    ├── base.py          Backend ABC + dataset loading
    ├── unsloth_backend.py
    ├── hf_backend.py
    └── ollama_backend.py
```

## Usage

```bash
# Train with Unsloth — loader/targets/masking auto-picked for the model
python -m finetuner.cli run \
    --backend unsloth \
    --base-model unsloth/gemma-4-e4b-it \
    --dataset runs/<topic>/training_data.jsonl \
    --output runs/<topic>/phase6 \
    --lora-rank 16 --lora-alpha 32 --epochs 1 --lr 1e-5 \
    --batch-size 4 --save-merged

# Portable HuggingFace path (same flags — model still auto-resolved)
python -m finetuner.cli run --backend hf --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset training_data.jsonl --output runs/ft --save-merged

# Package the merged checkpoint into Ollama (base-model = a PATH here)
python -m finetuner.cli run --backend ollama \
    --base-model runs/ft/merged_16bit --model-name my-model --output runs/ft

python -m finetuner.cli list           # list back-ends
```

`--backend ollama` reuses `--base-model` as the **path** to a merged
checkpoint or GGUF produced by an `unsloth`/`hf` run with `--save-merged` /
`--export-gguf`.

## Dataset format

JSON (a list) or JSONL (one object per line). Each row should carry a
`conversations` (or `messages`) list of `{role, content}` turns; an optional
`tools` list is rendered into the chat template for tool-calling training.
Plain `{question, answer}` rows — optionally with `reasoning` — are also
accepted and converted to a user/assistant conversation.

## Parameters

**`FinetuneConfig` — the flat set the UI collects** (model-independent):
`backend`, `base_model`, `dataset`, `output_dir`, `lora_rank`, `lora_alpha`,
`lora_dropout`, `epochs`, `learning_rate`, `batch_size`, `grad_accum`,
`warmup_steps`, `max_seq_length`, `max_steps`, `bf16`, `save_merged`,
`export_gguf`, `model_name`, `render_tools`.

**Developer overrides** (optional — the UI never sets these; normally
everything auto-resolves): `loader`, `target_modules`, `instruction_part`,
`response_part`, `trust_remote_code`. Exposed on the CLI as `--loader`,
`--target-modules`, `--instruction-part`, `--response-part`.
