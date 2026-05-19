#!/bin/bash
#SBATCH --job-name=ab-granite
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/ab-granite-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/ab-granite-%j.err

# Granite 4.0 micro A/B: notebook recipe vs our finetuner.
# Dataset: unsloth/Support-Bot-Recommendation CSV (transformed to conversations).

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
OUT=$REPO/runs/ab-granite
DATA=$REPO/runs/support-bot.jsonl
MODEL="unsloth/granite-4.0-micro"
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

if [ ! -f "$DATA" ]; then
echo "=== build Support-Bot conversations JSONL"
python - <<EOF
import json
from datasets import load_dataset
url = "https://huggingface.co/datasets/unsloth/Support-Bot-Recommendation/raw/main/support_recs.csv"
ds = load_dataset("csv", data_files={"train": url},
    column_names=["snippet","recommendation"], skiprows=1)["train"]
with open("$DATA","w") as f:
    for ex in ds:
        if ex.get("snippet") and ex.get("recommendation"):
            f.write(json.dumps({"conversations":[
                {"role":"user","content":ex["snippet"]},
                {"role":"assistant","content":ex["recommendation"]}]})+"\n")
print("Support-Bot ->", "$DATA")
EOF
fi

echo ""; echo "############ A: OFFICIAL UNSLOTH RECIPE (Granite 4.0 micro) ############"
python - <<EOF
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import torch, json
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
model, tok = FastLanguageModel.from_pretrained(
    model_name="$MODEL", max_seq_length=2048, load_in_4bit=False, full_finetuning=False)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                    "shared_mlp.input_linear","shared_mlp.output_linear"],
    use_gradient_checkpointing="unsloth", random_state=3407)
rows = [json.loads(l) for l in open("$DATA")]
def fmt(ex):
    return {"text":[tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in ex["conversations"]]}
from datasets import Dataset
ds = Dataset.from_list(rows).map(fmt, batched=True)
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=SFTConfig(
    dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
    warmup_steps=5, max_steps=30, learning_rate=2e-4, logging_steps=5,
    optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear", seed=3407,
    report_to="none", output_dir="$OUT/official-ckpts"))
trainer = train_on_responses_only(trainer,
    instruction_part="<|start_of_role|>user<|end_of_role|>",
    response_part="<|start_of_role|>assistant<|end_of_role|>")
stats = trainer.train()
print(f"[A:OFFICIAL] train_loss = {stats.metrics.get('train_loss'):.4f}")
print(f"[A:OFFICIAL] runtime_s  = {stats.metrics.get('train_runtime'):.1f}")
model.save_pretrained("$OUT/official-lora"); tok.save_pretrained("$OUT/official-lora")
FastLanguageModel.for_inference(model)
msgs=[{"role":"user","content":"I can't connect to my account. What should I do?"}]
inp=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True, padding=True).to("cuda")
out=model.generate(**inp, max_new_tokens=120)
print("[A:OFFICIAL] INFERENCE:", tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:300])
EOF

echo ""; echo "############ B: OUR FINETUNER ############"
time python -m finetuner.cli run \
    --backend unsloth --base-model "$MODEL" \
    --dataset "$DATA" --output "$OUT/finetuner" \
    --lora-rank 32 --lora-alpha 32 --lora-dropout 0 \
    --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,shared_mlp.input_linear,shared_mlp.output_linear" \
    --epochs 1 --max-steps 30 --lr 2e-4 \
    --batch-size 2 --grad-accum 4 --max-seq-length 2048

echo ""; echo "############ B: INFERENCE on finetuner LoRA ############"
python - <<EOF
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
from unsloth import FastLanguageModel
m, t = FastLanguageModel.from_pretrained(model_name="$OUT/finetuner/lora",
    max_seq_length=2048, load_in_4bit=False, full_finetuning=False)
FastLanguageModel.for_inference(m)
msgs=[{"role":"user","content":"I can't connect to my account. What should I do?"}]
inp=t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True, padding=True).to("cuda")
out=m.generate(**inp, max_new_tokens=120)
print("[B:FINETUNER] INFERENCE:", t.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:300])
EOF
echo "=== DONE  $(date)"
