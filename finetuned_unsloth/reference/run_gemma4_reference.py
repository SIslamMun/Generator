"""Unsloth Gemma 4 E4B — REFERENCE run (faithful port of the official notebook).

Verbatim training + inference path from
  reference/notebooks/nb/Gemma4_(E4B)-Text.ipynb
run with Unsloth's own dataset (mlabonne/FineTome-100k). This is the
known-good recipe we compare our finetuned_unsloth/models/gemma4_e4b/
pipeline against. Only the vision/audio demo cells and the `if False`
save cells are omitted; everything else is unchanged from the notebook.
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from unsloth import FastModel

# ── notebook cell 7: load model ─────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E4B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

# ── cell 22: LoRA via the finetune_*_layers flags (the Gemma-4 way) ──
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# ── cell 24: chat template ──────────────────────────────────────────
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")

# ── cells 26/28/32: dataset → standardize → render text ─────────────
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)
print("[ref] dataset rows:", len(dataset))
print("[ref] sample text (300 chars):")
print(dataset[100]["text"][:300])

# ── cell 36: SFTTrainer ─────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        output_dir = "outputs",
    ),
)

# ── cell 38: train on responses only ────────────────────────────────
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part    = "<|turn>model\n",
)

gpu = torch.cuda.get_device_properties(0)
print(f"[ref] GPU = {gpu.name}  max_mem = {round(gpu.total_memory/1e9,1)} GB")

# ── cell 45: train ──────────────────────────────────────────────────
stats = trainer.train()
print(f"[ref] train_loss = {stats.metrics.get('train_loss')}")
print(f"[ref] train_runtime_s = {stats.metrics.get('train_runtime')}")

# ── cell 50: inference (the notebook's exact recipe) ────────────────
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Why is the sky blue?"}],
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
out = model.generate(**inputs, max_new_tokens = 64,
                     temperature = 1.0, top_p = 0.95, top_k = 64)
print("[ref] INFERENCE OUTPUT:")
print(tokenizer.batch_decode(out)[0])

# ── cell 52: save the LoRA adapter ──────────────────────────────────
model.save_pretrained("gemma_4_lora")
tokenizer.save_pretrained("gemma_4_lora")
print("[ref] DONE — LoRA saved to gemma_4_lora/")
