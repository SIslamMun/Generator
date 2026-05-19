#!/bin/bash
#SBATCH --job-name=ab-qwen3
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/ab-qwen3-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/ab-qwen3-%j.err

# Qwen3 4B Instruct A/B: official Unsloth recipe vs our finetuner.
# Same model, same dataset (FineTome-100k 3k), same hyperparams.

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
OUT=$REPO/runs/ab-qwen3
DATA=$REPO/runs/finetome-3k.jsonl
MODEL="unsloth/Qwen3-4B-Instruct-2507"
mkdir -p "$OUT" "$REPO/finetuner/logs"

module load python/miniforge3_pytorch/2.7.0 || true
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME"
source $VENV/bin/activate
cd $REPO

# Ensure FineTome JSONL exists (was built by 2301262); rebuild if missing
if [ ! -f "$DATA" ]; then
echo "=== building FineTome JSONL"
python - <<EOF
import json
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
ds = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
ds = standardize_data_formats(ds)
with open("$DATA","w") as f:
    for ex in ds: f.write(json.dumps({"conversations": ex["conversations"]})+"\n")
EOF
fi

echo ""; echo "############ A: OFFICIAL UNSLOTH RECIPE ############"
python - <<EOF
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained(
    model_name="$MODEL", max_seq_length=1024, load_in_4bit=False, full_finetuning=False)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)
tok = get_chat_template(tok, chat_template="qwen3-instruct")

ds = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
from unsloth.chat_templates import standardize_data_formats
ds = standardize_data_formats(ds)
def fmt(ex):
    return {"text":[tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in ex["conversations"]]}
ds = ds.map(fmt, batched=True)
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=SFTConfig(
    dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
    warmup_steps=5, max_steps=30, learning_rate=2e-4, logging_steps=5,
    optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=3407,
    report_to="none", output_dir="$OUT/official-ckpts"))
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
stats = trainer.train()
print(f"[A:OFFICIAL] train_loss = {stats.metrics.get('train_loss'):.4f}")
print(f"[A:OFFICIAL] runtime_s  = {stats.metrics.get('train_runtime'):.1f}")
model.save_pretrained("$OUT/official-lora"); tok.save_pretrained("$OUT/official-lora")

# inference probe
FastLanguageModel.for_inference(model)
msgs=[{"role":"user","content":"Why is the sky blue?"}]
inp=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True).to("cuda")
out=model.generate(**inp, max_new_tokens=120, temperature=1.0, top_p=0.95, top_k=64)
print("[A:OFFICIAL] INFERENCE:", tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:300])
EOF

echo ""; echo "############ B: OUR FINETUNER (same hyperparams) ############"
time python -m finetuner.cli run \
    --backend unsloth --base-model "$MODEL" \
    --dataset "$DATA" --output "$OUT/finetuner" \
    --lora-rank 32 --lora-alpha 32 --lora-dropout 0 \
    --epochs 1 --max-steps 30 --lr 2e-4 \
    --batch-size 2 --grad-accum 4 --max-seq-length 1024

echo ""; echo "############ B: INFERENCE on finetuner LoRA ############"
python - <<EOF
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
m, t = FastLanguageModel.from_pretrained(model_name="$OUT/finetuner/lora",
    max_seq_length=1024, load_in_4bit=False, full_finetuning=False)
t = get_chat_template(t, chat_template="qwen3-instruct")
FastLanguageModel.for_inference(m)
msgs=[{"role":"user","content":"Why is the sky blue?"}]
inp=t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True).to("cuda")
out=m.generate(**inp, max_new_tokens=120, temperature=1.0, top_p=0.95, top_k=64)
print("[B:FINETUNER] INFERENCE:", t.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:300])
EOF
echo "=== DONE  $(date)"
