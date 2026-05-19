"""Model-profile resolver — turns a base-model id into the per-model deltas.

The web UI sends only a flat, model-agnostic config (backend, model, epochs,
lr, LoRA rank/alpha/dropout, batch, iterations …). Everything model-SPECIFIC
— the Unsloth loader class, LoRA target modules, response-masking turn
markers, trust_remote_code — is derived HERE from the model itself. So
changing the model in the UI changes the whole fine-tune correctly with no
other input.

Resolution: read the model's `AutoConfig` (`model_type`) + tokenizer chat
template, match a built-in family table, fall back to safe defaults for
unknown models. Anything explicitly set on the config overrides the
auto-resolved value (a developer escape hatch — the UI never sets these).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

# Standard attention + MLP LoRA targets — correct for the vast majority of
# decoder transformers (Llama, Qwen, Gemma, Granite, Mistral, Phi …).
_STD_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
# Mamba / SSM blocks add these projections.
_MAMBA_EXTRA = ["in_proj", "out_proj"]

# train_on_responses_only turn markers, keyed by chat-template family.
_MARKERS = {
    "chatml":  ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gemma":   ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma4":  ("<|turn>user\n", "<|turn>model\n"),
    "granite": ("<|start_of_role|>user<|end_of_role|>",
                "<|start_of_role|>assistant<|end_of_role|>"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "mistral": ("[INST]", "[/INST]"),
}

# model_type → per-family deltas.
#   loader       : "language" (FastLanguageModel) | "general" (FastModel)
#   targets      : "std" | "mamba" | "granite_h" (adds shared_mlp.* for H-hybrids)
#   markers      : key into _MARKERS (response-only masking)
#   chat_template: unsloth.chat_templates name to install via get_chat_template,
#                  or None to use the tokenizer's built-in template
#   multimodal   : if True, LoRA via finetune_*_layers flags (Gemma 3/4 style)
#                  instead of target_modules — the official notebook recipe.
_FAMILY = {
    "llama":       {"loader":"language", "targets":"std",       "markers":"llama3",  "chat_template": None,      "multimodal": False},
    "mistral":     {"loader":"language", "targets":"std",       "markers":"mistral", "chat_template": None,      "multimodal": False},
    "qwen2":       {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "qwen3":       {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": "qwen3-instruct", "multimodal": False},
    "qwen2_moe":   {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "qwen3_moe":   {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "phi3":        {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "phi4":        {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "mixtral":     {"loader":"language", "targets":"std",       "markers":"mistral", "chat_template": None,      "multimodal": False},
    "deepseek_v2": {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "deepseek_v3": {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "yi":          {"loader":"language", "targets":"std",       "markers":"chatml",  "chat_template": None,      "multimodal": False},
    "gemma":       {"loader":"language", "targets":"std",       "markers":"gemma",   "chat_template": None,      "multimodal": False},
    "gemma2":      {"loader":"language", "targets":"std",       "markers":"gemma",   "chat_template": None,      "multimodal": False},
    "gemma3":      {"loader":"general",  "targets":"std",       "markers":"gemma",   "chat_template": "gemma-3", "multimodal": True},
    "gemma3_text": {"loader":"language", "targets":"std",       "markers":"gemma",   "chat_template": None,      "multimodal": False},
    "gemma3n":     {"loader":"general",  "targets":"std",       "markers":"gemma",   "chat_template": "gemma-3", "multimodal": True},
    "gemma4":      {"loader":"general",  "targets":"std",       "markers":"gemma4",  "chat_template": "gemma-4", "multimodal": True},
    "granite":     {"loader":"language", "targets":"std",       "markers":"granite", "chat_template": None,      "multimodal": False},
    "granitemoe":  {"loader":"language", "targets":"granite_h", "markers":"granite", "chat_template": None,      "multimodal": False},
    "granitehybrid":{"loader":"language","targets":"granite_h", "markers":"granite", "chat_template": None,      "multimodal": False},
    "granitemoehybrid":{"loader":"language","targets":"granite_h","markers":"granite","chat_template": None,    "multimodal": False},
    "nemotron_h":  {"loader":"language", "targets":"mamba",     "markers":"chatml",  "chat_template": None,      "multimodal": False},
}
_GRANITE_H_EXTRA = ["shared_mlp.input_linear", "shared_mlp.output_linear"]
# Vision-language model_types → the vision-capable loader.
_VISION = {"qwen2_vl", "qwen2_5_vl", "mllama", "llava", "idefics3"}


@dataclass
class ModelProfile:
    """The per-model deltas an Unsloth fine-tune needs beyond the flat config."""

    model_type: str
    loader: str                       # "language" | "general" | "vision"
    target_modules: list[str]
    instruction_part: str | None      # None → skip response-only masking
    response_part: str | None
    trust_remote_code: bool
    chat_template_name: str | None    # get_chat_template(tok, this); None → built-in
    multimodal: bool                  # True → LoRA via finetune_*_layers flags
    source: str                       # how each field was decided (for logs)

    def summary(self) -> str:
        masking = (f"{self.instruction_part!r} / {self.response_part!r}"
                   if self.instruction_part else "(none — full-sequence)")
        lora_path = ("finetune_*_layers flags (multimodal)" if self.multimodal
                     else f"target_modules={self.target_modules}")
        return (f"  model_type     : {self.model_type}\n"
                f"  loader         : {self.loader}\n"
                f"  lora           : {lora_path}\n"
                f"  chat_template  : {self.chat_template_name or '(built-in)'}\n"
                f"  masking        : {masking}\n"
                f"  resolved by    : {self.source}")


def _marker_family_from_template(chat_template: str | None) -> str | None:
    """Heuristic fallback: detect the marker family from the chat template."""
    if not chat_template:
        return None
    if "<|turn|>" in chat_template or "<|turn>" in chat_template:
        return "gemma4"
    if "<start_of_turn>" in chat_template:
        return "gemma"
    if "<|start_of_role|>" in chat_template:
        return "granite"
    if "<|start_header_id|>" in chat_template:
        return "llama3"
    if "<|im_start|>" in chat_template:
        return "chatml"
    return None


def resolve(base_model: str, cfg=None) -> ModelProfile:
    """Resolve the per-model fine-tuning deltas for ``base_model``.

    ``base_model`` is an HF id or a local path. ``cfg`` (a FinetuneConfig) is
    consulted only for explicit overrides — normally all fields auto-resolve.
    """
    from transformers import AutoConfig, AutoTokenizer

    model_type = "?"
    chat_template = None
    try:
        ac = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        model_type = getattr(ac, "model_type", "?")
    except Exception as e:                       # noqa: BLE001
        warnings.warn(f"[profiles] could not read AutoConfig for {base_model}: {e}")
    try:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        chat_template = getattr(tok, "chat_template", None)
    except Exception:                            # noqa: BLE001
        pass

    if model_type in _VISION:
        entry = {"loader": "vision", "targets": "std", "markers": None,
                 "chat_template": None, "multimodal": True}
        source = f"family table ({model_type}: vision)"
    elif model_type in _FAMILY:
        entry = _FAMILY[model_type]
        source = f"family table ({model_type})"
    else:
        entry = {"loader": "language", "targets": "std", "markers": None,
                 "chat_template": None, "multimodal": False}
        source = f"DEFAULTS — unknown model_type '{model_type}'"
        warnings.warn(f"[profiles] unknown model_type '{model_type}' for "
                      f"{base_model}; using language loader + standard targets")

    fam = entry["markers"]
    # Markers: family table first; else sniff the chat template.
    if fam is None:
        fam = _marker_family_from_template(chat_template)
        if fam:
            source += f" + template-detected markers ({fam})"
    instruction_part, response_part = _MARKERS.get(fam, (None, None))

    tkind = entry["targets"]
    target_modules = list(_STD_TARGETS)
    if tkind == "mamba":
        target_modules += _MAMBA_EXTRA
    elif tkind == "granite_h":
        target_modules += _GRANITE_H_EXTRA

    profile = ModelProfile(
        model_type=model_type,
        loader=entry["loader"],
        target_modules=target_modules,
        instruction_part=instruction_part,
        response_part=response_part,
        trust_remote_code=True,
        chat_template_name=entry["chat_template"],
        multimodal=entry["multimodal"],
        source=source,
    )

    # Developer overrides (the UI never sets these).
    if cfg is not None:
        if getattr(cfg, "loader", ""):
            profile.loader = cfg.loader
        if getattr(cfg, "target_modules", None):
            profile.target_modules = list(cfg.target_modules)
        if getattr(cfg, "instruction_part", ""):
            profile.instruction_part = cfg.instruction_part
        if getattr(cfg, "response_part", ""):
            profile.response_part = cfg.response_part
        if getattr(cfg, "trust_remote_code", None) is not None:
            profile.trust_remote_code = cfg.trust_remote_code
        if any(getattr(cfg, f, None) for f in
               ("loader", "target_modules", "instruction_part", "response_part")):
            profile.source += " + cfg overrides"
    return profile
