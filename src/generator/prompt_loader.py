"""
Prompt loading utilities.

Loads prompts from individual YAML files in configs/prompts/ directory.
- Single-prompt files: have a 'prompt' key containing the template
- Multi-prompt files (e.g., tool_prompts.yaml): have multiple keys, each being a prompt
"""

import yaml  # type: ignore[import-untyped]
from pathlib import Path
from typing import Dict


def load_prompts(config_dir: Path) -> Dict[str, str]:
    """
    Load all prompts from configs/prompts/ directory.

    Supports two formats:
    - Single-prompt files: YAML with a 'prompt' key
    - Multi-prompt files: YAML with multiple prompt keys (e.g., tool_prompts.yaml)

    Args:
        config_dir: Path to configs directory

    Returns:
        Dict mapping prompt name to prompt template string
    """
    config_dir = Path(config_dir)
    prompts_dir = config_dir / "prompts"

    # Accept either a config dir (with a `prompts/` subdir) or the prompts
    # directory itself (the *.yaml files directly inside it).
    if not prompts_dir.exists():
        if config_dir.is_dir() and any(config_dir.glob("*.yaml")):
            prompts_dir = config_dir
        else:
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompts = {}

    for prompt_file in prompts_dir.glob("*.yaml"):
        prompt_name = prompt_file.stem  # filename without .yaml

        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            continue
            
        # Check if it's a single-prompt file (has 'prompt' key)
        if "prompt" in data:
            prompts[prompt_name] = data["prompt"]
        else:
            # Multi-prompt file: all keys are prompt names
            # Only include string values (skip comments, metadata, etc.)
            for key, value in data.items():
                if isinstance(value, str):
                    prompts[key] = value

    return prompts
