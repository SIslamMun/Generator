"""
Prompt loading utilities.

Loads prompts from individual YAML files in configs/prompts/ directory.
Each prompt file has a 'prompt' key containing the template.
"""

import yaml  # type: ignore[import-untyped]
from pathlib import Path
from typing import Dict


def load_prompts(config_dir: Path) -> Dict[str, str]:
    """
    Load all prompts from configs/prompts/ directory.

    Each YAML file should have a 'prompt' key with the template.

    Args:
        config_dir: Path to configs directory

    Returns:
        Dict mapping prompt name to prompt template string
    """
    prompts_dir = config_dir / "prompts"
    
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompts = {}
    
    for prompt_file in prompts_dir.glob("*.yaml"):
        prompt_name = prompt_file.stem  # filename without .yaml
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        if "prompt" not in data:
            raise ValueError(f"'{prompt_file.name}' missing 'prompt' key")
            
        prompts[prompt_name] = data["prompt"]
    
    return prompts
