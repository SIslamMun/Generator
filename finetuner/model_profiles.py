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

# model_type → (loader, target-kind, marker-family).
#   loader      : "language" (FastLanguageModel) | "general" (FastModel)
#   target-kind : "std" | "mamba"
_FAMILY = {
    "llama":       ("language", "std",   "llama3"),
    "mistral":     ("language", "std",   "mistral"),
    "qwen2":       ("language", "std",   "chatml"),
    "qwen3":       ("language", "std",   "chatml"),
    "qwen2_moe":   ("language", "std",   "chatml"),
    "qwen3_moe":   ("language", "std",   "chatml"),
    "phi3":        ("language", "std",   "chatml"),
    "gemma":       ("language", "std",   "gemma"),
    "gemma2":      ("language", "std",   "gemma"),
    "gemma3":      ("general",  "std",   "gemma"),
    "gemma3_text": ("language", "std",   "gemma"),
    "gemma3n":     ("general",  "std",   "gemma"),
    "gemma4":      ("general",  "std",   "gemma4"),
    "granite":     ("language", "std",   "granite"),
    "granitemoe":  ("language", "std",   "granite"),
    "nemotron_h":  ("language", "mamba", "chatml"),
}
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
    source: str                       # how each field was decided (for logs)

    def summary(self) -> str:
        masking = (f"{self.instruction_part!r} / {self.response_part!r}"
                   if self.instruction_part else "(none — full-sequence)")
        return (f"  model_type     : {self.model_type}\n"
                f"  loader         : {self.loader}\n"
                f"  target_modules : {self.target_modules}\n"
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
        loader, tkind, fam = "vision", "std", None
        source = f"family table ({model_type}: vision)"
    elif model_type in _FAMILY:
        loader, tkind, fam = _FAMILY[model_type]
        source = f"family table ({model_type})"
    else:
        loader, tkind, fam = "language", "std", None
        source = f"DEFAULTS — unknown model_type '{model_type}'"
        warnings.warn(f"[profiles] unknown model_type '{model_type}' for "
                      f"{base_model}; using language loader + standard targets")

    # Markers: family table first; else sniff the chat template.
    if fam is None:
        fam = _marker_family_from_template(chat_template)
        if fam:
            source += f" + template-detected markers ({fam})"
    instruction_part, response_part = _MARKERS.get(fam, (None, None))

    target_modules = list(_STD_TARGETS) + (_MAMBA_EXTRA if tkind == "mamba" else [])

    profile = ModelProfile(
        model_type=model_type, loader=loader, target_modules=target_modules,
        instruction_part=instruction_part, response_part=response_part,
        trust_remote_code=True, source=source,
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
