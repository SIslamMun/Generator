"""
QA pair curation using LLM-as-Judge.

Implements quality filtering based on:
- AlpaGasus (2023): LLM rates pairs 1-10
- LIMA (2023): High threshold (≥7.0) for quality
"""

import json
import json5
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from .clients import get_client, BaseLLMClient

console = Console()


def curate_qa_pairs(
    input_path: str,
    output_path: str,
    prompts: Dict[str, str],
    llm_config: Dict[str, Any],
    threshold: float = 7.0,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """
    Filter QA pairs by quality using LLM-as-Judge.

    Args:
        input_path: Path to input QA pairs JSON
        output_path: Where to save curated pairs
        prompts: Prompt templates dict
        llm_config: LLM configuration dict
        threshold: Minimum rating to keep (1-10 scale)
        batch_size: Number of pairs to rate in one LLM call

    Returns:
        Dict with metrics (total, filtered, retention_rate, avg_score)
    """
    console.print(f"\n[bold cyan]📊 Loading QA pairs from: {input_path}[/bold cyan]")

    # Load QA pairs
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    total_pairs = len(qa_pairs)
    console.print(f"[green]✓ Loaded {total_pairs} QA pairs[/green]\n")

    # Initialize LLM client
    console.print(f"[bold cyan]🤖 Initializing Judge LLM: {llm_config['provider']}[/bold cyan]")
    provider = llm_config.pop("provider")
    llm = get_client(provider, llm_config)
    console.print(f"[green]✓ Judge ready: {llm.model}[/green]\n")

    # Get rating prompt
    rating_prompt_template = prompts.get("qa_rating")
    if not rating_prompt_template:
        raise ValueError("qa_rating prompt not found in prompts config")

    rated_pairs = []
    filtered_pairs = []
    total_score = 0.0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process in batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Rating QA pairs...", total=total_pairs)

        for i in range(0, total_pairs, batch_size):
            batch = qa_pairs[i : i + batch_size]

            try:
                # Rate this batch
                rated_batch = _rate_batch(
                    pairs=batch, llm=llm, prompt_template=rating_prompt_template
                )

                for pair in rated_batch:
                    rating = pair.get("rating", 0)
                    total_score += rating

                    rated_pairs.append(pair)

                    # Filter by threshold
                    if rating >= threshold:
                        filtered_pairs.append(pair)

            except Exception as e:
                console.print(f"[yellow]⚠ Error rating batch {i}-{i+batch_size}: {e}[/yellow]")

            progress.advance(task, advance=len(batch))

    # Calculate metrics
    avg_score = total_score / len(rated_pairs) if rated_pairs else 0
    retention_rate = len(filtered_pairs) / total_pairs if total_pairs > 0 else 0

    metrics = {
        "total_pairs": total_pairs,
        "rated_pairs": len(rated_pairs),
        "filtered_pairs": len(filtered_pairs),
        "retention_rate": retention_rate,
        "avg_score": avg_score,
        "threshold": threshold,
        "curated_at": datetime.now().isoformat(),
    }

    # Save results
    _save_curated_results(
        curated_pairs=filtered_pairs, all_rated=rated_pairs, metrics=metrics, output_path=output_file
    )

    console.print(f"\n[bold green]✓ Curated {len(filtered_pairs)} / {total_pairs} pairs[/bold green]")
    console.print(f"[bold green]✓ Retention rate: {retention_rate:.1%}[/bold green]")
    console.print(f"[bold green]✓ Average score: {avg_score:.2f}/10[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]\n")

    return metrics


def _rate_batch(
    pairs: List[Dict], llm: BaseLLMClient, prompt_template: str
) -> List[Dict]:
    """Rate a batch of QA pairs."""
    # Format pairs for prompt
    pairs_str = json.dumps(
        [{"question": p["question"], "answer": p["answer"]} for p in pairs],
        indent=2,
        ensure_ascii=False,
    )

    # Fill prompt
    prompt = prompt_template.format(pairs=pairs_str)

    # Generate ratings
    response = llm.generate(prompt, temperature=0.1)  # Low temp for consistency

    # Parse JSON response
    try:
        rated = json5.loads(response)
    except Exception:
        # Try to extract JSON from markdown
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            rated = json5.loads(json_str)
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            rated = json5.loads(json_str)
        else:
            # Fallback: assign default rating
            rated = [{"question": p["question"], "answer": p["answer"], "rating": 5} for p in pairs]

    # Merge ratings with original pairs
    result = []
    for i, pair in enumerate(pairs):
        if i < len(rated):
            pair["rating"] = rated[i].get("rating", 5)
        else:
            pair["rating"] = 5  # Default if not enough ratings
        result.append(pair)

    return result


def _save_curated_results(
    curated_pairs: List[Dict],
    all_rated: List[Dict],
    metrics: Dict[str, Any],
    output_path: Path,
):
    """Save curated results with metrics."""
    # Save curated pairs (threshold filtered)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(curated_pairs, f, indent=2, ensure_ascii=False)

    # Save JSONL format
    jsonl_path = output_path.parent / f"{output_path.stem}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in curated_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Save all rated pairs (with scores)
    all_rated_path = output_path.parent / f"{output_path.stem}_all_rated.json"
    with open(all_rated_path, "w", encoding="utf-8") as f:
        json.dump(all_rated, f, indent=2, ensure_ascii=False)

    # Save metrics
    metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save score distribution
    if all_rated:
        scores = [p.get("rating", 0) for p in all_rated]
        distribution = {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "histogram": {
                str(i): sum(1 for s in scores if i <= s < i + 1) for i in range(1, 11)
            },
        }

        dist_path = output_path.parent / f"{output_path.stem}_distribution.json"
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(distribution, f, indent=2, ensure_ascii=False)
