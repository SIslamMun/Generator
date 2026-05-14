# Universal Fine-Tuning Pipeline (Unsloth)

Model-agnostic training infrastructure. Each model has its own folder
under `models/` containing config + data prep + training code. The
top-level `train.py` dispatches based on which model you pick.

## Architecture

```
finetuned_unsloth/
├── train.py                 ← dispatcher: pick model + data, runs prep+validate(+submit)
├── _common/
│   └── registry.py          ← discovers all models/<name>/ subfolders
└── models/
    └── <model_name>/        ← one folder per model (self-contained)
        ├── config.yaml          (LoRA r, batch, lr, save targets, install deps)
        ├── prepare_data.py      (generator JSON → JSONL with this model's chat template)
        ├── validate_data.py     (sanity-check the prepared JSONL)
        ├── train.py             (Unsloth + TRL training)
        ├── install.sh           (one-time venv setup, model-specific deps)
        ├── submit.sbatch        (SLURM wrapper)
        ├── data/                (prepared JSONL lands here)
        ├── artifacts/           (lora / merged_16bit / gguf land here)
        └── .venv-<model>/       (dedicated venv, isolated from generator's .venv-delta)
```

Each model is its own folder because **chat templates, tool-call formats,
target_modules, and dependency versions differ per model family**. We
keep these differences localized so adding a model never modifies global
code — just drop a new folder.

## Flow

```
[generator output: tool/QA/CoT JSON]
            │
            ▼
   finetuned_unsloth/train.py --model X --types ...
            │
            ├──► models/X/prepare_data.py  ──► models/X/data/train.jsonl
            │
            ├──► models/X/validate_data.py
            │
            └──► sbatch models/X/submit.sbatch
                         │
                         └──► models/X/train.py
                                      │
                                      └──► models/X/artifacts/{lora,merged_16bit,gguf}
```

## Currently supported

```bash
python finetuned_unsloth/train.py --list
```

| model | family | use case | notes |
|---|---|---|---|
| `nemotron_nano_4b` | nemotron-3-nano | tool-use / QA / CoT | Hybrid Mamba+Attention, needs `mamba_ssm` |

## How to use

```bash
# 1. Pick a model and prepare data (no training yet)
python finetuned_unsloth/train.py \
    --model nemotron_nano_4b \
    --types tool \
    --in-tool runs/<topic>/data/<file>.json \
    --tool-catalog configs/tools/<topic>_tools.json

# 2. Same command + --submit to actually queue the training job
python finetuned_unsloth/train.py ... --submit
```

`--types` accepts any subset of `{qa, cot, tool}`:
- `qa` — plain question/answer pairs
- `cot` — question / `<think>reasoning</think>` / answer
- `tool` — full tool-use traces (user / assistant-with-tool_calls / tool-result / final)
- combinations: `qa,cot` | `qa,cot,tool` | `tool,cot` | ...

Unsloth recommends ~75% reasoning / 25% non-reasoning when mixing — the
prepare step prints the actual ratio it produced for inspection.

## Adding a new model

```bash
cp -r finetuned_unsloth/models/nemotron_nano_4b finetuned_unsloth/models/<new>
# edit <new>/config.yaml
#   - hf_model_id
#   - lora.target_modules            (architecture-dependent)
#   - masking.{instruction,response} (from the model's chat template)
#   - venv.install_deps              (drop mamba_ssm if not hybrid)
```

The dispatcher auto-discovers the new folder. No changes to global code.

---

# Legacy Jarvis pipeline (pre-dispatcher)

All pre-`models/` code lives under `legacy/`. It still runs (the
universal `generator train-chat` / `train-tool` commands route through
`legacy/{data,train,test}/`), but new model work should use the
`models/` architecture above.

```
finetuned_unsloth/legacy/
├── data/        FunctionGemma dataset prep (convert_to_functiongemma.py, schema_filter.py, …)
├── train/       Legacy training scripts (train.py for FunctionGemma, train_qa.py for Gemma3)
├── test/        Legacy eval harnesses
├── references/  Upstream Unsloth reference notebooks
└── QA_Train/    Gemma3 QA reference notebook
```

The big Jarvis-era trained artifacts (`qa_v1/`, `v10/`) and run logs
moved to `misc/finetuned_unsloth_archive/`. Reproduce v10:

- HF: `AutoModelForCausalLM.from_pretrained("misc/finetuned_unsloth_archive/artifacts/v10/model_merged_16bit")`
- Ollama: `ollama create jarvis-v10 -f misc/finetuned_unsloth_archive/artifacts/v10/Modelfile`

Git history: `856e8fb` (jarvis-qa-v1), `5340802` (v10 tool-use model).

### Delta-AI environment notes

- **Do NOT `module purge`** — it removes compilers that `python/anaconda3/2.10.0` depends on.
- **Keep Ollama's model store on shared `/work/hdd`** so the whole `delta_bekn` group reuses it:
  `OLLAMA_MODELS=/work/hdd/bekn/ollama/models`.
- **Use `uv`** (`~/.local/bin/uv`) for all per-model venvs — that's what each model's `install.sh` does.
