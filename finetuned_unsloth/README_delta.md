# Running the FunctionGemma fine-tune on Delta-AI

Three files in this folder work together:

| File | Purpose |
|---|---|
| `train.py` | Headless training script (same recipe as the Colab notebook). Driven by CLI flags. |
| `submit_delta.sbatch` | SLURM batch script that loads modules, activates the venv, and runs `train.py`. |
| `v7_2k/jarvis_v7_functiongemma.jsonl` | The 1,979-example dataset already converted to FunctionGemma chat format. |

## One-time setup on a Delta login node

Delta compute nodes often have no internet, so download all wheels and the base model on a **login node**, then reuse them in the job.

```bash
cd $PROJECT_DIR/context/Generator         # adjust to your checkout path

# 1. Create a Python venv (Delta ships a modern Python; module-load whichever you prefer)
module load python/3.11 cuda/12.4.1
python -m venv .venv-delta
source .venv-delta/bin/activate
pip install --upgrade pip

# 2. Install training deps. Mirrors the Colab notebook's install cell.
pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth"                                     # pulls unsloth_zoo, peft, bitsandbytes, trl, triton, accelerate
pip install "transformers==4.56.2" "trl==0.22.2" --no-deps
pip install "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer sentencepiece protobuf

# 3. Pre-cache the base model on the login node so compute nodes don't need internet
export HF_HOME=$SCRATCH/hf_cache
python -c "
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
snapshot_download('unsloth/functiongemma-270m-it')
AutoTokenizer.from_pretrained('unsloth/functiongemma-270m-it')
print('cached')
"
```

## Edit the SBATCH header

In `submit_delta.sbatch`:

```bash
#SBATCH --account=REPLACE_ME_ACCOUNT       # your Delta allocation name
#SBATCH --partition=gpuA100x4              # or ghx4 on Delta-AI
```

Partition examples:
- Delta (NCSA): `gpuA100x4`, `gpuA100x8`, `gpuA40x4`
- Delta-AI: `ghx4` (Grace-Hopper) — if your allocation is on Delta-AI, edit the partition and adjust modules (`nvhpc` instead of `cuda`).

## Submit

```bash
cd context/Generator/finetuned_unsloth
sbatch submit_delta.sbatch
```

Track progress:

```bash
squeue -u $USER
tail -f logs/jarvis-ft-<jobid>.out
```

## Expected runtime

| Cluster / GPU | Time @ max_steps=500, batch=16 |
|---|---|
| Delta A100-40GB | ~7-10 min |
| Delta-AI GH200 | ~4-6 min |
| Colab T4 | ~25-35 min |

## Outputs

The job writes to `$SCRATCH/jarvis_v7_lora/` by default (set `JARVIS_FT_OUT=...` in the sbatch environment to override):

```
$SCRATCH/jarvis_v7_lora/
├── checkpoints/       # trainer checkpoints (if save_strategy != "no")
├── lora/              # LoRA adapter (always saved)
├── merged_16bit/      # merged HF model (if --save-merged)
└── gguf/              # Q8_0 GGUF for Ollama (if --export-gguf)
```

After the job finishes, `rsync` the `gguf/` directory back to your workstation and register with Ollama:

```bash
ollama create jarvis-v7 -f Modelfile.v7
```

## Smoke-testing `train.py` locally

You can do a fast sanity check on a laptop GPU (or even CPU, very slowly) before submitting:

```bash
python train.py \
  --dataset v7_2k/jarvis_v7_functiongemma.jsonl \
  --output /tmp/jarvis_ft_test \
  --max-steps 5 --batch-size 1 --grad-accum 1
```

A 5-step run should finish in <2 minutes and write a `lora/` directory. If it fails, the SLURM job will fail too — fix it here first.

## Tuning knobs

| Flag | Default | Effect |
|---|---|---|
| `--max-steps` | 500 | More steps → more memorization, but `progress.md` shows diminishing returns past ~500 for 2k examples. |
| `--epochs` | (unset) | If set, overrides `--max-steps`. 3 epochs over 1,979 examples = ~1,485 steps at batch 4. |
| `--batch-size` | 4 | Raise to 16-32 on A100; monitor `nvidia-smi` in the first 10 steps. |
| `--grad-accum` | 2 | Drop to 1 when you raise `--batch-size`. |
| `--lr` | 2e-4 | Unsloth's published value. Lower to 2e-5 for long runs (10k+ steps). |
| `--lora-r` / `--lora-alpha` | 128 / 256 | Unsloth's recipe. r=64 cuts VRAM ~30% at mild quality cost. |
| `--bf16` | off | Turn on for A100/H100. |
| `--save-merged` | off | Produces a merged HF checkpoint in addition to the LoRA adapter. |
| `--export-gguf` | off | Emits a Q8_0 GGUF suitable for `ollama create`. |

## If the job fails

- **`ModuleNotFoundError: unsloth`** — venv not activated; double-check the `source "$VENV/bin/activate"` line.
- **`CUDA error: no kernel image`** — the PyTorch build doesn't match Delta's CUDA. Re-install torch with the matching `--index-url cu12X` wheels.
- **`OSError: ... functiongemma-270m-it is not cached`** — compute node has no internet. Re-run the pre-cache step on a login node with `HF_HOME=$SCRATCH/hf_cache`, and make sure the sbatch script exports the same `HF_HOME`.
- **OOM on the first step** — drop `--batch-size` by half, raise `--grad-accum` to compensate.
