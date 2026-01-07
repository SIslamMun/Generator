"""
Export QA pairs and tool examples to various training formats.

Supported formats:
- chatml: ChatML format (for OpenAI-style models)
- alpaca: Alpaca instruction format
- sharegpt: ShareGPT conversation format
- jsonl: Simple JSONL (one pair per line)

Supports both:
- QA pairs: {"question": "...", "answer": "..."}
- Tool examples: {"instruction": "...", "solution": {...}}
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console

console = Console()


def _is_tool_example(pair: Dict) -> bool:
    """Check if this is a tool example vs QA pair."""
    return "instruction" in pair and "solution" in pair


def _format_tool_solution(solution: Dict) -> str:
    """Format tool solution as readable assistant response."""
    parts = []
    
    # Add reasoning steps
    if "reasoning_path" in solution:
        for step in solution["reasoning_path"]:
            step_num = step.get("step", "")
            thought = step.get("thought", "")
            tool = step.get("tool", "")
            args = step.get("args", {})
            
            parts.append(f"Step {step_num}: {thought}")
            parts.append(f"Tool: {tool}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})")
            parts.append("")
    
    # Add final answer
    if "final_answer" in solution:
        parts.append(solution["final_answer"])
    
    return "\n".join(parts).strip()


def export_to_format(
    input_path: str, output_path: str, format_type: str = "chatml", system_prompt: Optional[str] = None
) -> int:
    """
    Export QA pairs to training format.

    Args:
        input_path: Path to curated QA pairs JSON
        output_path: Where to save formatted data
        format_type: Output format ("chatml", "alpaca", "sharegpt", "jsonl")
        system_prompt: Optional system prompt for conversation formats

    Returns:
        Number of examples exported
    """
    console.print(f"\n[bold cyan]📊 Loading QA pairs from: {input_path}[/bold cyan]")

    # Load QA pairs
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    console.print(f"[green]✓ Loaded {len(qa_pairs)} QA pairs[/green]")
    console.print(f"[bold cyan]🔄 Converting to {format_type} format...[/bold cyan]\n")

    # Convert to target format
    if format_type == "chatml":
        formatted = _to_chatml(qa_pairs, system_prompt)
    elif format_type == "alpaca":
        formatted = _to_alpaca(qa_pairs)
    elif format_type == "sharegpt":
        formatted = _to_sharegpt(qa_pairs, system_prompt)
    elif format_type == "jsonl":
        formatted = _to_jsonl(qa_pairs)
    else:
        raise ValueError(f"Unknown format: {format_type}")

    # Save output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"[bold green]✓ Exported {len(formatted)} examples[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]\n")

    return len(formatted)


def _to_chatml(qa_pairs: List[Dict], system_prompt: Optional[str] = None) -> List[Dict]:
    """
    Convert to ChatML format.

    Example:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is HDF5?"},
            {"role": "assistant", "content": "HDF5 is a data model..."}
        ]
    }
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that provides accurate information."

    formatted = []
    for pair in qa_pairs:
        # Handle tool examples
        if _is_tool_example(pair):
            user_content = pair["instruction"]
            assistant_content = _format_tool_solution(pair["solution"])
        else:
            # Standard QA pair
            user_content = pair["question"]
            assistant_content = pair["answer"]
        
        item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        formatted.append(item)

    return formatted


def _to_alpaca(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Convert to Alpaca instruction format.

    Example:
    {
        "instruction": "What is HDF5?",
        "input": "",
        "output": "HDF5 is a data model..."
    }
    """
    formatted = []
    for pair in qa_pairs:
        # Handle tool examples
        if _is_tool_example(pair):
            instruction = pair["instruction"]
            output = _format_tool_solution(pair["solution"])
        else:
            # Standard QA pair
            instruction = pair["question"]
            output = pair["answer"]
        
        item = {
            "instruction": instruction,
            "input": "",  # No additional input in our case
            "output": output,
        }
        formatted.append(item)

    return formatted


def _to_sharegpt(qa_pairs: List[Dict], system_prompt: Optional[str] = None) -> List[Dict]:
    """
    Convert to ShareGPT format.

    Example:
    {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "What is HDF5?"},
            {"from": "gpt", "value": "HDF5 is a data model..."}
        ]
    }
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that provides accurate information."

    formatted = []
    for pair in qa_pairs:
        # Handle tool examples
        if _is_tool_example(pair):
            user_content = pair["instruction"]
            assistant_content = _format_tool_solution(pair["solution"])
        else:
            # Standard QA pair
            user_content = pair["question"]
            assistant_content = pair["answer"]
        
        item = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": user_content},
                {"from": "gpt", "value": assistant_content},
            ]
        }
        formatted.append(item)

    return formatted


def _to_jsonl(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Convert to simple JSONL format.

    Example (QA):
    {
        "question": "What is HDF5?",
        "answer": "HDF5 is a data model...",
        "metadata": {...}
    }
    
    Example (Tool):
    {
        "instruction": "Open my data file",
        "response": "Step 1: ...",
        "metadata": {...}
    }
    """
    formatted = []
    for pair in qa_pairs:
        # Handle tool examples
        if _is_tool_example(pair):
            item = {
                "instruction": pair["instruction"],
                "response": _format_tool_solution(pair["solution"]),
                "metadata": {
                    k: v
                    for k, v in pair.items()
                    if k not in ["instruction", "solution"]
                },
            }
        else:
            # Standard QA pair
            item = {
                "question": pair["question"],
                "answer": pair["answer"],
                "metadata": {
                    k: v
                    for k, v in pair.items()
                    if k not in ["question", "answer"]
                },
            }
        formatted.append(item)

    return formatted
