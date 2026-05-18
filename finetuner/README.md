# finetuner — multi-backend Phase 6 fine-tuning

A thin wrapper that dispatches a fine-tune job to one of three back-ends,
sharing a single parameter set. Implements the design in
[grc-iit/Phagocyte#4](https://github.com/grc-iit/Phagocyte/issues/4).

> **Status:** new module — not yet tested. Intended to run on the Delta
> cluster (GPU). The model-agnostic `finetuned_unsloth/` pipeline is the
> other Phase-6 path; this `finetuner/` is the multi-backend design.

## Back-ends

| `--backend` | What it does | Needs |
|---|---|---|
| `unsloth` | LoRA training via Unsloth's `FastLanguageModel` + TRL `SFTTrainer`. Fast, low-VRAM. | CUDA GPU, `unsloth` |
| `hf` | LoRA training via `transformers` + `peft` + TRL. Portable (no custom kernels). | CUDA GPU, `transformers`, `peft` |
| `ollama` | Packages an already-trained checkpoint into an Ollama model via a generated `Modelfile`. Does **not** train. | `ollama` on PATH |

## Layout

```
finetuner/
├── config.py            FinetuneConfig — the shared parameter set
├── cli.py               `finetuner run` / `finetuner list`
└── backends/
    ├── base.py          Backend ABC + dataset loading
    ├── unsloth_backend.py
    ├── hf_backend.py
    └── ollama_backend.py
```

## Usage

```bash
# Train with Unsloth (recommended)
python -m finetuner.cli run \
    --backend unsloth \
    --base-model unsloth/llama-3.1-8b \
    --dataset runs/<topic>/training_data.jsonl \
    --output runs/<topic>/phase6 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --epochs 1 --lr 1e-5 --batch-size 4 --save-merged

# Train with the portable HuggingFace path
python -m finetuner.cli run --backend hf --base-model meta-llama/Llama-3.1-8B \
    --dataset training_data.jsonl --output runs/ft --save-merged

# Package the merged checkpoint into Ollama
python -m finetuner.cli run --backend ollama \
    --base-model runs/ft/merged_16bit --model-name my-model --output runs/ft

# List back-ends
python -m finetuner.cli list
```

`--backend ollama` reuses `--base-model` as the **path** to a merged
checkpoint or GGUF (typically produced by an `unsloth`/`hf` run with
`--save-merged` / `--export-gguf`).

## Dataset format

JSON (a list) or JSONL (one object per line). Each row should carry a
`conversations` (or `messages`) list of `{role, content}` turns. Plain
`{question, answer}` rows — optionally with `reasoning` — are accepted too
and converted to a user/assistant conversation, so Phase-5 QA/CoT dumps
train directly.

## Parameters (`FinetuneConfig`)

`base_model`, `dataset`, `output_dir`, `lora_rank`, `lora_alpha`,
`lora_dropout`, `epochs`, `learning_rate`, `batch_size`, `grad_accum`,
`max_seq_length`, `max_steps`, `bf16`, `save_merged`, `export_gguf`,
`model_name` — mirrors the inputs the Phagocyte web UI collects for Phase 6.
