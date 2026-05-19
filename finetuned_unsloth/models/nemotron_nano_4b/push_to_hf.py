"""Convert merged_16bit → GGUF and push to a HuggingFace Hub repo.

Reads HF_TOKEN from the environment (never written to disk). Repo name
defaults to <user>/NDP-Nemotron-3-Nano-4B-tool-calling. Override with
HF_REPO env var.

Outputs:
  artifacts/gguf_q8_0/ndp-nemotron-3-nano-4b.gguf  (~4–5 GB)
  artifacts/gguf_q4_k_m/...                        (~2 GB)

Files uploaded to HF:
  README.md                  (auto-generated model card)
  config.json
  tokenizer.json + tokenizer_config.json + chat_template.jinja
  modeling_nemotron_h.py + configuration_nemotron_h.py
  model.safetensors          (fp16 weights, 7.5 GB)
  *.gguf                     (quantized variants)
"""
from __future__ import annotations

# Same compat shim as test_inference.py — mamba_ssm 2.2.5 vs transformers 5.x
import transformers.generation as _gen
import transformers.generation.utils as _gen_utils
if not hasattr(_gen, "GreedySearchDecoderOnlyOutput"):
    _gen.GreedySearchDecoderOnlyOutput = getattr(_gen_utils, "GenerateDecoderOnlyOutput", _gen_utils.ModelOutput)
if not hasattr(_gen, "SampleDecoderOnlyOutput"):
    _gen.SampleDecoderOnlyOutput = getattr(_gen_utils, "GenerateDecoderOnlyOutput", _gen_utils.ModelOutput)

import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED_DIR = HERE / "artifacts" / "merged_16bit"
LORA_DIR   = HERE / "artifacts" / "lora"
GGUF_BASE  = HERE / "artifacts" / "gguf"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_OVERRIDE = os.environ.get("HF_REPO")
if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN env var is not set")


def render_model_card(repo_id: str, summary_data: dict) -> str:
    train_loss = summary_data.get("train_loss")
    train_runtime = summary_data.get("train_runtime_s")
    n_rows = summary_data.get("n_rows")
    max_steps = summary_data.get("max_steps")
    peak_vram = summary_data.get("peak_vram_gb")
    return f'''---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: unsloth/NVIDIA-Nemotron-3-Nano-4B
library_name: transformers
pipeline_tag: text-generation
tags:
  - nemotron
  - nemotron-h
  - hybrid-mamba-attention
  - tool-calling
  - function-calling
  - ndp
  - national-data-platform
  - unsloth
language:
  - en
---

# NDP Nemotron-3 Nano 4B — Tool Calling

Fine-tune of **[unsloth/NVIDIA-Nemotron-3-Nano-4B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-4B)**
on synthetic tool-use traces for the [National Data Platform (NDP)](http://155.101.6.191:8003)
MCP server. The model emits `<tool_call>` blocks invoking three NDP catalog tools:
`list_organizations`, `search_datasets`, and `get_dataset_details`.

## Quick stats

| | |
|---|---|
| **Base model** | unsloth/NVIDIA-Nemotron-3-Nano-4B (hybrid Mamba+Attention, 4B params) |
| **Training method** | LoRA (r=8, alpha=16) via [Unsloth](https://github.com/unslothai/unsloth) + TRL SFTTrainer |
| **Training data** | {n_rows} synthetic NDP tool-use examples, generated with gpt-oss:120b as teacher and curated with LLM-as-judge (threshold 7.0) |
| **Steps / epochs** | {max_steps} steps (~3 epochs) |
| **Final train loss** | {train_loss:.4f} |
| **Training time** | {train_runtime:.0f} s on 1× NVIDIA GH200 |
| **Peak VRAM** | {peak_vram:.1f} GB |
| **Max seq length** | 4096 |

## Intended use

This model is for **tool-call generation against the NDP MCP server**. Given a natural-language
NDP query plus the three-tool catalog as `tools=...` to `apply_chat_template`, the model
produces a tool call in Nemotron's XML format:

```
<tool_call>
<function=search_datasets>
<parameter=search_terms>
["climate"]
</parameter>
<parameter=server>
global
</parameter>
</function>
</tool_call>
```

## NDP tool surface trained on

| tool | purpose |
|---|---|
| `list_organizations(name_filter?, server?)` | List data publishers, optionally filtered |
| `search_datasets(...)` | Simple (search_terms[]) or advanced (owner_org, resource_format, filter_list, …) dataset search |
| `get_dataset_details(dataset_identifier, identifier_type?, server?)` | Full metadata by UUID or name slug |

Tool catalog source-of-truth: [configs/tools/ndp_tools.json](https://github.com/SIslamMun/Generator/blob/main/configs/tools/ndp_tools.json).

## Inference

The Nemotron-3 Nano family needs:
- `transformers>=5.3,<=5.5.0` (uses `TokenizersBackend` introduced in v5; capped by unsloth-zoo)
- `mamba_ssm==2.2.5` + `causal_conv1d==1.5.2` (CUDA kernels compiled for your arch)
- `use_cache=False` in `generate()` (current Nemotron-H modeling has a bug with cache_position)

```python
import torch
# COMPAT: mamba_ssm 2.2.5 imports a class removed in transformers v5
import transformers.generation as _g, transformers.generation.utils as _gu
for cls in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
    if not hasattr(_g, cls):
        setattr(_g, cls, getattr(_gu, "GenerateDecoderOnlyOutput", _gu.ModelOutput))

from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("{repo_id}", max_seq_length=4096, trust_remote_code=True)
FastLanguageModel.for_inference(model)

import json
TOOLS = json.load(open("ndp_tools_for_chat_template.json"))   # the catalog
messages = [{{"role":"user","content":"Find datasets about climate."}}]
text = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to("cuda")
out = model.generate(
    **inputs, max_new_tokens=512, do_sample=False, use_cache=False,
    eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    stop_strings=["</function>"], tokenizer=tok,
)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
```

## Known limitations

1. **Phantom None parameters**: the model often emits ALL schema parameters with `None`
   values for the unused ones (e.g. a `list_organizations` call may include `dataset_name=None`,
   `search_terms=None`, etc.). This is a learned artifact of seeing the full schema in
   the system message during training. **Use a post-parser** to strip parameters whose
   value is `None`/`null`/empty before invoking the real MCP server. Reference parser:
   [test_inference.py:parse_tool_call](https://github.com/SIslamMun/Generator/blob/main/finetuned_unsloth/models/nemotron_nano_4b/test_inference.py).
2. **Looping on some queries**: rarely (≈ 1/8 in our smoke test) the model loops on
   `<parameter>` blocks without emitting `</function>`. The tolerant parser variant
   recovers args even from truncated output. Setting `stop_strings=["</function>"]` at
   generation time helps when the model does emit it.
3. **NDP-specific only**: the model has seen exactly three tools. It is not a general
   tool-use model — it will not generalize to other MCP catalogs.

## Training data generation pipeline

```
NDP MCP server (3 tools)
   │
   ├─ tool-generate-full           (gpt-oss:120b)   → 1879 raw examples
   ├─ schema-filter                                 → 1879 (no drops)
   ├─ tool-curate                  (gpt-oss:120b)   → 1712 kept @ threshold 7.0
   └─ prepare_data.py → fine-tune  (Unsloth + TRL)
```

Generator + curator: [SIslamMun/Generator](https://github.com/SIslamMun/Generator).

## Files in this repo

| file | purpose |
|---|---|
| `model.safetensors` | fp16 merged weights (~7.5 GB) — load with `transformers` |
| `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja` | tokenizer + Nemotron tool-aware chat template |
| `modeling_nemotron_h.py` + `configuration_nemotron_h.py` | dynamic remote code (required by `trust_remote_code=True`) |
| `*.gguf` (if uploaded) | GGUF quantizations for llama.cpp / Ollama / LMStudio |

## License

Inherits the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) of the base model.
'''


def convert_to_gguf_via_llamacpp(quant: str) -> Path | None:
    """Convert MERGED_DIR → GGUF using a locally-built llama.cpp.

    Returns the path to the .gguf file on success, None on failure.
    """
    import subprocess
    llamacpp = Path(os.environ.get("LLAMACPP_DIR", "/work/nvme/bekn/sislam3/llama.cpp"))
    convert  = llamacpp / "convert_hf_to_gguf.py"
    quantize = llamacpp / "build" / "bin" / "llama-quantize"
    if not convert.exists() or not quantize.exists():
        print(f"  ERROR: llama.cpp not built at {llamacpp}. Run build_llamacpp.sh first.")
        return None

    out_dir = GGUF_BASE / quant
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_path = out_dir / "model-bf16.gguf"
    final_path = out_dir / f"ndp-nemotron-3-nano-4b-{quant}.gguf"
    if final_path.exists():
        print(f"  [gguf] {final_path.name} already exists, skipping")
        return final_path

    # Step 1: HF safetensors → GGUF bf16 (once, shared across quants)
    if not bf16_path.exists():
        print(f"  [gguf] converting HF → bf16 GGUF")
        rc = subprocess.call([
            sys.executable, str(convert),
            str(MERGED_DIR),
            "--outfile", str(bf16_path),
            "--outtype", "bf16",
        ])
        if rc != 0 or not bf16_path.exists():
            print(f"  ERROR: convert_hf_to_gguf.py exited {rc}")
            return None

    # Step 2: bf16 → target quantization
    print(f"  [gguf] quantizing bf16 → {quant}")
    rc = subprocess.call([str(quantize), str(bf16_path), str(final_path), quant.upper()])
    if rc != 0 or not final_path.exists():
        print(f"  ERROR: llama-quantize {quant} exited {rc}")
        return None
    print(f"  [gguf] wrote {final_path} ({final_path.stat().st_size / 1024 / 1024:.0f} MB)")
    return final_path


def patch_chat_template(merged_dir: Path) -> int:
    """Null-guard the Nemotron chat template so it survives minja (the strict
    Jinja engine in llama.cpp / LM Studio).

    Stock template applies `| string` to fields that can be JSON null
    (content, message.content, extra-key values, param type). Python Jinja
    renders `none | string` as 'None'; minja raises. We wrap each with an
    `(X if X is not none else '')` guard — correct for ALL value types
    (unlike `default('', true)`, this never corrupts a legit 0 / false).

    Patches both chat_template.jinja and the chat_template field embedded
    in tokenizer_config.json. Returns the number of substitutions made.
    """
    # (stock fragment) -> (null-guarded fragment)
    REPLACEMENTS = [
        ("(json_dict[json_key] | string)",
         "((json_dict[json_key] if json_dict[json_key] is not none else '') | string)"),
        ("(param_fields.type | string)",
         "((param_fields.type if param_fields.type is not none else '') | string)"),
        ("(content | string)",
         "((content if content is not none else '') | string)"),
        ("message.content | string",
         "(message.content if message.content is not none else '') | string"),
    ]
    total = 0

    jinja = merged_dir / "chat_template.jinja"
    if jinja.exists():
        txt = jinja.read_text()
        for old, new in REPLACEMENTS:
            if old in txt and new not in txt:
                txt = txt.replace(old, new)
                total += 1
        jinja.write_text(txt)

    # tokenizer_config.json may embed the same template under "chat_template"
    tcfg = merged_dir / "tokenizer_config.json"
    if tcfg.exists():
        cfg = json.loads(tcfg.read_text())
        ct = cfg.get("chat_template")
        if isinstance(ct, str):
            for old, new in REPLACEMENTS:
                if old in ct and new not in ct:
                    ct = ct.replace(old, new)
                    total += 1
            cfg["chat_template"] = ct
            tcfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    return total


def main():
    if not MERGED_DIR.exists():
        sys.exit(f"ERROR: {MERGED_DIR} missing")

    # ── 0. Auth + repo check FIRST (fail fast on bad token) ──
    print(f"=== HF auth check (token length: {len(HF_TOKEN)})")
    from huggingface_hub import HfApi, login, create_repo, upload_folder, upload_file
    login(token=HF_TOKEN, add_to_git_credential=False)
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    user = user_info["name"]
    perms = user_info.get("auth", {}).get("accessToken", {}).get("role", "?")
    print(f"  user: {user}  token role: {perms}")
    repo_id = HF_REPO_OVERRIDE or f"{user}/NDP-Nemotron-3-Nano-4B-tool-calling"
    print(f"  repo: {repo_id}")

    print(f"=== creating repo (or confirming write access)")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=HF_TOKEN)
    print(f"  ok — write access confirmed")

    # ── 1. Render model card now (cheap) ──
    summary = json.loads((HERE / "artifacts" / "train_summary.json").read_text())
    card = render_model_card(repo_id, summary)
    card_path = HERE / "artifacts" / "README.md"
    card_path.write_text(card)
    print(f"=== model card → {card_path}")

    # ── 1b. Copy dynamic remote-code files (Unsloth's save_pretrained_merged
    #        omits them; transformers' trust_remote_code load needs them) ──
    import glob, shutil
    snap_glob = glob.glob(str(Path(os.environ.get(
        "HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        / "models--unsloth--NVIDIA-Nemotron-3-Nano-4B" / "snapshots" / "*"))
    if snap_glob:
        snap = Path(snap_glob[0])
        for f in ("modeling_nemotron_h.py", "configuration_nemotron_h.py",
                  "generation_config.json", "special_tokens_map.json"):
            src, dst = snap / f, MERGED_DIR / f
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"=== copied dynamic-code file: {f}")

    # ── 1c. Null-guard the chat template (LM Studio / minja compatibility) ──
    n_patched = patch_chat_template(MERGED_DIR)
    print(f"=== chat-template null-guards applied: {n_patched}")

    # ── 2. Convert GGUF (q8_0, q4_k_m) using locally-built llama.cpp ──
    gguf_files: list[Path] = []
    for quant in ("q8_0", "q4_k_m"):
        print(f"\n=== GGUF {quant}")
        gguf = convert_to_gguf_via_llamacpp(quant)
        if gguf:
            gguf_files.append(gguf)

    # ── 3. Upload merged_16bit (everything except the .cache/) ──
    print(f"\n=== uploading merged_16bit → {repo_id}")
    upload_folder(
        repo_id=repo_id, repo_type="model", token=HF_TOKEN,
        folder_path=str(MERGED_DIR),
        path_in_repo=".",
        ignore_patterns=[".cache", "*.tmp", "*.lock"],
        commit_message="upload merged_16bit + tokenizer + dynamic code",
    )

    # ── 4. Upload README ──
    upload_file(
        repo_id=repo_id, repo_type="model", token=HF_TOKEN,
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        commit_message="add model card",
    )

    # ── 5. Render model card ──
    summary = json.loads((HERE / "artifacts" / "train_summary.json").read_text())
    card = render_model_card(repo_id, summary)
    card_path = HERE / "artifacts" / "README.md"
    card_path.write_text(card)
    print(f"  wrote model card → {card_path}")

    # ── 6. Upload merged_16bit (everything except the .cache/) ──
    print(f"=== uploading merged_16bit → {repo_id}")
    upload_folder(
        repo_id=repo_id, repo_type="model", token=HF_TOKEN,
        folder_path=str(MERGED_DIR),
        path_in_repo=".",
        ignore_patterns=[".cache", "*.tmp", "*.lock"],
        commit_message="upload merged_16bit + tokenizer + dynamic code",
    )

    # ── 7. Upload model card ──
    upload_file(
        repo_id=repo_id, repo_type="model", token=HF_TOKEN,
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        commit_message="add model card",
    )

    # ── 5. Upload GGUFs ──
    for ggfile in gguf_files:
        print(f"=== uploading {ggfile.name}")
        try:
            upload_file(
                repo_id=repo_id, repo_type="model", token=HF_TOKEN,
                path_or_fileobj=str(ggfile), path_in_repo=ggfile.name,
                commit_message=f"add {ggfile.name}",
            )
        except Exception as e:
            print(f"  WARN: upload of {ggfile} failed: {e}")

    # ── 6. Upload LoRA adapter (small, useful for composable inference) ──
    if LORA_DIR.exists():
        print(f"=== uploading LoRA → {repo_id}/lora")
        upload_folder(
            repo_id=repo_id, repo_type="model", token=HF_TOKEN,
            folder_path=str(LORA_DIR),
            path_in_repo="lora",
            commit_message="add LoRA adapter",
        )

    print(f"\n[done] https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
