"""YAML config schema, defaults, validation, and interactive wizard."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


def slugify(text: str) -> str:
    """Topic string → directory-safe slug. `Jarvis-CD HPC` → `jarvis-cd-hpc`."""
    s = re.sub(r"[^\w\s-]", "", (text or "").strip().lower())
    s = re.sub(r"[-\s_]+", "-", s)
    return s.strip("-") or "untitled"


def derive_output_dir(topic: str, base: str = "./runs") -> str:
    """Default output_dir for a topic: <base>/<topic-slug>."""
    return f"{base}/{slugify(topic)}"


# ────────────────────────── defaults ──────────────────────────

CHAT_DEFAULTS = {
    "enabled": True,
    "data": {
        "lancedb_path": "./lancedb",
        "table": "code_chunks",
        "target_pairs": 3000,
        "max_chunks": None,
    },
    "curate":  {"threshold": 7.0},
    "train": {
        "base_model": "unsloth/gemma-3-270m-it",
        "lora_r": 128,
        "lora_alpha": 256,
        "epochs": 3,
        "batch_size": 16,
        "lr": 2e-4,
        "bf16": True,
        "save_merged": True,
    },
    "eval": {
        "enabled": True,
        "val_split": 0.05,
        "baseline": "unsloth/gemma-3-270m-it",
        "max_examples": 0,
    },
}

TOOL_DEFAULTS = {
    "enabled": True,
    "data": {
        "tools_path": "configs/tools/jarvis_tools.json",
        "target_pairs": 3000,
        "ratios": {"single": 0.10, "multi": 0.15, "chain": 0.45, "error": 0.30},
        "tools_per_example": 10,
        "distractor_strategy": "mixed",
    },
    "train": {
        "base_model": "unsloth/functiongemma-270m-it",
        "lora_r": 128,
        "lora_alpha": 256,
        "epochs": 3,
        "max_steps": 0,            # 0 = use epochs
        "batch_size": 16,
        "lr": 2e-4,
        "bf16": True,
        "save_merged": True,
    },
    "eval": {
        "enabled": True,
        "max_examples": 0,
    },
}

GLOBAL_DEFAULTS = {
    "output_dir": None,            # if absent/null, derived from topic → ./runs/<topic-slug>
    "topic": "my-domain",
    "llm": {
        "provider": "ollama",
        "model": "gpt-oss:20b",
        "workers": 4,
    },
    "chat": CHAT_DEFAULTS,
    "tool": TOOL_DEFAULTS,
}


# ────────────────────────── load / save ──────────────────────────

def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    merged = _deep_merge(GLOBAL_DEFAULTS, cfg)
    if not merged.get("output_dir"):
        merged["output_dir"] = derive_output_dir(merged.get("topic", "untitled"))
    return merged


def save_config(cfg: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return p


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ────────────────────────── interactive wizard ──────────────────────────

def wizard(kind: str = "both") -> dict:
    """Build a config interactively. `kind` is 'chat', 'tool', or 'both'."""
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.console import Console

    c = Console()
    cfg: dict = _deep_merge(GLOBAL_DEFAULTS, {})
    cfg["chat"]["enabled"] = kind in ("chat", "both")
    cfg["tool"]["enabled"] = kind in ("tool", "both")

    c.rule("[bold]Generator training pipeline · interactive setup[/bold]")
    cfg["topic"]      = Prompt.ask("🎯 Topic / domain (drives output folder name)", default=cfg["topic"])
    cfg["output_dir"] = Prompt.ask("📁 Output directory", default=derive_output_dir(cfg["topic"]))

    c.rule("[bold]LLM[/bold]")
    cfg["llm"]["provider"] = Prompt.ask(
        "Provider", default=cfg["llm"]["provider"],
        choices=["ollama", "claude-sdk", "gemini", "openai", "vllm"],
    )
    cfg["llm"]["model"]   = Prompt.ask("Model",  default=cfg["llm"]["model"])
    cfg["llm"]["workers"] = IntPrompt.ask("Workers", default=cfg["llm"]["workers"])

    if cfg["chat"]["enabled"]:
        c.rule("[bold]Chat (QA + CoT)[/bold]")
        cfg["chat"]["data"]["lancedb_path"] = Prompt.ask(
            "LanceDB path", default=cfg["chat"]["data"]["lancedb_path"])
        cfg["chat"]["data"]["table"] = Prompt.ask(
            "Table", default=cfg["chat"]["data"]["table"])
        cfg["chat"]["data"]["target_pairs"] = IntPrompt.ask(
            "Target QA+CoT pairs", default=cfg["chat"]["data"]["target_pairs"])
        cfg["chat"]["curate"]["threshold"] = FloatPrompt.ask(
            "Curate threshold (1-10)", default=cfg["chat"]["curate"]["threshold"])
        cfg["chat"]["train"]["base_model"] = Prompt.ask(
            "Chat base model", default=cfg["chat"]["train"]["base_model"])
        cfg["chat"]["train"]["epochs"]    = IntPrompt.ask("Epochs", default=cfg["chat"]["train"]["epochs"])
        cfg["chat"]["train"]["batch_size"] = IntPrompt.ask("Batch size", default=cfg["chat"]["train"]["batch_size"])
        cfg["chat"]["eval"]["enabled"] = Confirm.ask("Run chat eval?", default=True)

    if cfg["tool"]["enabled"]:
        c.rule("[bold]Tool-use[/bold]")
        cfg["tool"]["data"]["tools_path"] = Prompt.ask(
            "Tool catalog (json)", default=cfg["tool"]["data"]["tools_path"])
        cfg["tool"]["data"]["target_pairs"] = IntPrompt.ask(
            "Target tool examples", default=cfg["tool"]["data"]["target_pairs"])
        cfg["tool"]["train"]["base_model"] = Prompt.ask(
            "Tool base model", default=cfg["tool"]["train"]["base_model"])
        cfg["tool"]["train"]["epochs"] = IntPrompt.ask("Epochs", default=cfg["tool"]["train"]["epochs"])
        cfg["tool"]["train"]["batch_size"] = IntPrompt.ask("Batch size", default=cfg["tool"]["train"]["batch_size"])
        cfg["tool"]["eval"]["enabled"] = Confirm.ask("Run tool eval?", default=True)

    c.rule()
    return cfg
