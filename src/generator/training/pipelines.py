"""Top-level pipelines: chat, tool, both. Wires stages → runner."""
from __future__ import annotations

from pathlib import Path
from rich.console import Console

from .config import load_config
from .runner  import Stage, run_stages
from .stages.gen_cot       import run_gen_cot,       chat_data_paths
from .stages.fix_cot       import run_fix_cot
from .stages.curate        import run_curate
from .stages.prep_chat     import run_prep_chat
from .stages.train_chat    import run_train_chat,    chat_artifact_paths
from .stages.eval_chat     import run_eval_chat
from .stages.gen_tool      import run_gen_tool,      tool_data_paths
from .stages.schema_filter import run_schema_filter
from .stages.prep_tool     import run_prep_tool
from .stages.train_tool    import run_train_tool,    tool_artifact_paths
from .stages.eval_tool     import run_eval_tool


console = Console()


def chat_stages(cfg: dict) -> list[Stage]:
    p = chat_data_paths(cfg)
    a = chat_artifact_paths(cfg)
    return [
        Stage(
            name="gen-cot", description="LanceDB chunks → QA + CoT",
            inputs=[p["lancedb"]], outputs=[p["raw_cot"]],
            fn=run_gen_cot, skip_if_disabled_path="chat.enabled",
        ),
        Stage(
            name="fix-cot", description="re-attempt empty-CoT rows",
            inputs=[p["raw_cot"]], outputs=[p["fixed"]],
            fn=run_fix_cot, skip_if_disabled_path="chat.enabled",
        ),
        Stage(
            name="curate", description="LLM-judge filter (threshold)",
            inputs=[p["fixed"]], outputs=[p["curated"]],
            fn=run_curate, skip_if_disabled_path="chat.enabled",
        ),
        Stage(
            name="prep-chat", description="JSON → ChatML JSONL (train + val)",
            inputs=[p["curated"]], outputs=[p["train"]],
            fn=run_prep_chat, skip_if_disabled_path="chat.enabled",
        ),
        Stage(
            name="train-chat", description="Gemma3 + LoRA",
            inputs=[p["train"]], outputs=[a["merged"] / "model.safetensors"],
            fn=run_train_chat, skip_if_disabled_path="chat.enabled",
        ),
        Stage(
            name="eval-chat", description="held-out scoring vs baseline",
            inputs=[a["merged"] / "model.safetensors", p["val"]],
            outputs=[Path(cfg["output_dir"]) / "reports" / "chat_eval" / "jarvis_qa_v1_summary.json"],
            fn=run_eval_chat, skip_if_disabled_path="chat.enabled",
        ),
    ]


def tool_stages(cfg: dict) -> list[Stage]:
    p = tool_data_paths(cfg)
    a = tool_artifact_paths(cfg)
    return [
        Stage(
            name="gen-tool", description="generate tool-use examples",
            inputs=[p["tools_path"]], outputs=[p["raw"]],
            fn=run_gen_tool, skip_if_disabled_path="tool.enabled",
        ),
        Stage(
            name="schema-filter", description="drop bad arg-shape rows",
            inputs=[p["raw"]], outputs=[p["clean"]],
            fn=run_schema_filter, skip_if_disabled_path="tool.enabled",
        ),
        Stage(
            name="prep-tool", description="convert to FunctionGemma JSONL",
            inputs=[p["clean"]], outputs=[p["train"]],
            fn=run_prep_tool, skip_if_disabled_path="tool.enabled",
        ),
        Stage(
            name="train-tool", description="FunctionGemma + LoRA",
            inputs=[p["train"]], outputs=[a["merged"] / "model.safetensors"],
            fn=run_train_tool, skip_if_disabled_path="tool.enabled",
        ),
        Stage(
            name="eval-tool", description="offline tool-call structure grade",
            inputs=[a["merged"] / "model.safetensors", p["train"]],
            outputs=[Path(cfg["output_dir"]) / "reports" / "tool_eval" / "summary.json"],
            fn=run_eval_tool, skip_if_disabled_path="tool.enabled",
        ),
    ]


def run_chat(cfg: dict, **kw):
    state = Path(cfg["output_dir"]) / "logs" / "chat_state.json"
    return run_stages(chat_stages(cfg), cfg, state_path=state, **kw)


def run_tool(cfg: dict, **kw):
    state = Path(cfg["output_dir"]) / "logs" / "tool_state.json"
    return run_stages(tool_stages(cfg), cfg, state_path=state, **kw)


def run_both(cfg: dict, **kw):
    """Run chat then tool. (They could run in parallel but keeping it serial to be safe.)"""
    run_chat(cfg, **kw)
    run_tool(cfg, **kw)
