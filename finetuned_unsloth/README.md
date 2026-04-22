# finetuned_unsloth

Fine-tunes `unsloth/functiongemma-270m-it` on the Jarvis-CD tool-use dataset and
serves the trained model via HF transformers **or** Ollama. The model emits
FunctionGemma's `<start_function_call>call:<tool>{...}<end_function_call>`
format and is wired into the real Jarvis MCP server through `inference/` at
the repo root.

## Layout

```
finetuned_unsloth/
├── data/                # dataset prep + training corpus
│   ├── convert_to_functiongemma.py   # v7 raw → FunctionGemma chat-template JSONL
│   ├── validate_dataset.py           # sanity-check conversions
│   └── v7_10k_clean/                 # 6977 training examples
├── train/               # LoRA fine-tune on a GH200
│   ├── train.py
│   ├── submit_delta.sbatch
│   └── train_jarvis_functiongemma.ipynb
├── test/                # held-out evaluation harnesses
│   ├── test_model.py / test_model.sbatch           # HF transformers backend
│   ├── ollama_test_model.py / ollama_test.sbatch   # Ollama backend
│   └── debug_inference.py / debug_inference.sbatch # single-prompt diagnostics
├── artifacts/           # trained weights + ollama assets
│   ├── model_merged_16bit/           # HF-format fp16 (load via transformers)
│   ├── jarvis_v7_fp16.gguf           # standalone GGUF (llama.cpp / LM Studio / koboldcpp)
│   └── Modelfile                     # ollama Modelfile (FROM the .gguf above)
├── references/          # upstream Unsloth reference notebooks
└── logs/                # slurm output
```

## Quick start

**Build the dataset** (from an existing v7 raw JSON with `reasoning_path`):

```bash
python data/convert_to_functiongemma.py \
    --input  path/to/v7_raw.json \
    --output data/v7_10k_clean/jarvis_v7_functiongemma.jsonl
python data/validate_dataset.py data/v7_10k_clean/jarvis_v7_functiongemma.jsonl
```

**Train** (GH200, Delta-AI):

```bash
sbatch train/submit_delta.sbatch      # ~14 min, 3000 steps, merged fp16 saved to $SCRATCH_OUT
```

**Evaluate via HF transformers**:

```bash
sbatch test/test_model.sbatch         # runs test/test_model.py on the 10 held-out prompts
```

**Evaluate via Ollama** (same 10 prompts, routed through Ollama's `/api/generate`):

```bash
ollama create jarvis-v7 -f artifacts/Modelfile     # one-time import (GGUF → ollama blob)
sbatch test/ollama_test.sbatch
```

## Reproducing the model locally

The merged fp16 weights and a standalone GGUF are both in `artifacts/`:

- **HF transformers** — `AutoModelForCausalLM.from_pretrained("finetuned_unsloth/artifacts/model_merged_16bit")`
- **Ollama** — `ollama create jarvis-v7 -f finetuned_unsloth/artifacts/Modelfile`
- **llama.cpp** — `./main -m finetuned_unsloth/artifacts/jarvis_v7_fp16.gguf …`

## Benchmark results

| Backend | Tool selection | Args correct | Avg latency |
|---------|----------------|--------------|-------------|
| HF transformers (GH200) | 10/10 (100%) | 9/10 (90%) | — |
| Ollama (GH200)          | 10/10 (100%) | 9/10 (90%) | 0.6 s/query |

The remaining arg miss is `destroy_pipeline("old_deprecated_test")` instead of
`destroy_pipeline("deprecated_test")` — the model picked up the adjective
"old" from the user's phrasing. That's a training-data coverage gap, not a
runtime issue.

## Delta-AI environment notes

- **Do NOT `module purge`** — it removes compilers that `python/anaconda3/2.10.0`
  depends on.
- **Use `--system-site-packages` on the venv** so it inherits the module's
  CUDA-enabled `torch 2.10.0+cu129` instead of resolving a fresh CPU wheel.
- **Keep Ollama's model store on scratch** — NFS-backed `$HOME` is far too
  slow for the safetensors→GGUF conversion (~6 hours vs. a few seconds).
  Set `OLLAMA_MODELS=/work/hdd/bekn/$USER/ollama_models`.

## Configuration referenced

- Tool catalog: `../configs/tools/jarvis_tools.yaml` (29 Jarvis tools)
- Real MCP harness: `../inference/run.py` (wires the trained model + real
  `jarvis-mcp` server — see `../inference/README.md`)
