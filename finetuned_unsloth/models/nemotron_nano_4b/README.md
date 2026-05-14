# Nemotron-3 Nano 4B — fine-tune recipe

Self-contained model folder. Everything model-specific (config, data
prep, training script, install dependencies, SLURM wrapper) lives here.

Source notebook: [Nemotron-3-Nano-30B-A3B_A100.ipynb](../../nemotron/Nemotron-3-Nano-30B-A3B_A100.ipynb) (we
adapt from the 30B-A3B notebook; the 4B uses the same training pattern
with a different `hf_model_id`).

## Layout

```
nemotron_nano_4b/
├── config.yaml          ← edit me (LoRA rank, batch, lr, save targets, ...)
├── prepare_data.py      ← generator JSON → train.jsonl with Nemotron chat template
├── validate_data.py     ← sanity-check the prepared JSONL
├── train.py             ← Unsloth + TRL training (reads config.yaml)
├── install.sh           ← one-time venv setup (mamba_ssm + torch==2.7.1)
├── submit.sbatch        ← SLURM wrapper for Delta
├── data/                ← prepared JSONL lands here
├── artifacts/           ← lora / merged_16bit / gguf land here
└── .venv-nemotron/      ← dedicated venv (built by install.sh on first sbatch run)
```

## Usage (from repo root)

The top-level dispatcher does prep + validate + submit:

```bash
# 1. List available models
python finetuned_unsloth/train.py --list

# 2. Prep + validate only (no training yet)
python finetuned_unsloth/train.py \
    --model nemotron_nano_4b \
    --types tool \
    --in-tool runs/ndp/data/ndp_tool_examples_curated.json \
    --tool-catalog configs/tools/ndp_tools.json

# 3. Prep + validate + sbatch (full pipeline)
python finetuned_unsloth/train.py \
    --model nemotron_nano_4b \
    --types tool \
    --in-tool runs/ndp/data/ndp_tool_examples_curated.json \
    --tool-catalog configs/tools/ndp_tools.json \
    --submit
```

`--types` accepts any combination: `qa` | `qa,cot` | `tool` | `qa,cot,tool`.

For each chosen type, point at the matching `--in-*` file:

```bash
# QA only
python finetuned_unsloth/train.py --model nemotron_nano_4b \
    --types qa --in-qa runs/<topic>/data/qa_curated.json --submit

# QA + CoT
python finetuned_unsloth/train.py --model nemotron_nano_4b \
    --types qa,cot \
    --in-qa  runs/<topic>/data/qa_curated.json \
    --in-cot runs/<topic>/data/cot_curated.json --submit

# All three (Unsloth recommends ~75% reasoning : 25% non-reasoning)
python finetuned_unsloth/train.py --model nemotron_nano_4b \
    --types qa,cot,tool \
    --in-qa   runs/<topic>/data/qa_curated.json \
    --in-cot  runs/<topic>/data/cot_curated.json \
    --in-tool runs/<topic>/data/tool_examples_curated.json \
    --tool-catalog configs/tools/<topic>_tools.json --submit
```

## First-run install

The first `sbatch` run will detect that `.venv-nemotron/` doesn't
exist and call `install.sh` automatically. The venv install takes
~10-15 min on a Delta compute node (`mamba_ssm` builds from source
against the pinned `torch==2.7.1`). Subsequent runs skip install
entirely.

If you want to pre-install (e.g. interactive node):
```bash
ssh gh<NNN>          # any compute node with nvcc
cd /u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b
bash install.sh
```

## Outputs

After training, `artifacts/` contains the formats enabled in
`config.yaml.save`:

| format | use case |
|---|---|
| `lora/` | Tiny LoRA adapter — load alongside the base model in Unsloth/HF/vLLM. |
| `merged_16bit/` | Standalone bf16 checkpoint — drop into vLLM / Ollama / LMStudio. |
| `merged_4bit/` | Quantized standalone — smaller but lossy. |
| `gguf_q8_0/`, `gguf_q4_k_m/` | llama.cpp / Ollama-native formats. |

A `train_summary.json` is written next to the artifacts with loss,
runtime, peak VRAM, and dataset row count for the run.

## Adding another model

Copy this folder and edit the obvious bits:

```bash
cp -r finetuned_unsloth/models/nemotron_nano_4b finetuned_unsloth/models/<new_model>
# then edit <new_model>/config.yaml:
#   - hf_model_id        → the new HF repo
#   - family/architecture → informational
#   - lora.target_modules → match the model's architecture
#   - masking.{instruction_part, response_part} → from the model's chat template
#   - venv.install_deps  → adjust if it doesn't need mamba_ssm, or needs different torch
```

The dispatcher auto-discovers the new folder via `_common/registry.py`
— no other code changes.
