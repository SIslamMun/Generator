"""QA enrichment module for improving answer quality through response rewriting."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import json5
from rich.console import Console
from rich.progress import track

from .clients import get_client
from .prompt_loader import load_prompts

console = Console()


def enrich_qa_pairs(
    qa_pairs: List[Dict],
    llm_config: Dict,
    prompts_dir: Path,
    batch_size: int = 5,
    preserve_original: bool = True,
    temperature: float = 0.3,
) -> List[Dict]:
    """
    Enrich QA pairs by rewriting answers for better clarity and structure.

    Uses LLM to rewrite answers while preserving all information from the original.
    This implements a "back-and-forth translation" style approach where we:
    1. Rewrite the answer for better formatting and clarity
    2. Preserve the original for comparison
    3. Track what changes were made

    Args:
        qa_pairs: List of QA pairs to enrich
        llm_config: LLM configuration (provider, model, etc.)
        prompts_dir: Directory containing prompt templates
        batch_size: Number of pairs to process per LLM call
        preserve_original: Keep original answer in output

    Returns:
        List of enriched QA pairs with improved answers
    """
    console.print(f"[cyan]Enriching {len(qa_pairs)} QA pairs with improved responses...[/cyan]")

    # Load prompt template
    prompts = load_prompts(prompts_dir)
    enrichment_prompt = prompts.get("qa_enrichment", "")

    if not enrichment_prompt:
        console.print("[red]Error: qa_enrichment prompt not found![/red]")
        return qa_pairs

    # Initialize LLM client
    provider = llm_config.pop("provider")
    llm = get_client(provider, llm_config)

    # Process in batches
    enriched_pairs = []
    total_batches = (len(qa_pairs) + batch_size - 1) // batch_size

    for i in track(
        range(0, len(qa_pairs), batch_size),
        description="Enriching QA pairs",
        total=total_batches,
    ):
        batch = qa_pairs[i : i + batch_size]
        enriched_batch = _enrich_batch(
            batch, llm, enrichment_prompt, preserve_original, temperature
        )
        enriched_pairs.extend(enriched_batch)

    console.print(
        f"[green]✓ Enriched {len(enriched_pairs)} pairs successfully[/green]"
    )
    return enriched_pairs


def _enrich_batch(
    pairs: List[Dict],
    llm,
    prompt_template: str,
    preserve_original: bool,
    temperature: float = 0.3,
) -> List[Dict]:
    """
    Enrich a batch of QA pairs.

    Args:
        pairs: Batch of QA pairs
        llm: LLM client
        prompt_template: Enrichment prompt template
        preserve_original: Keep original answer
        temperature: Sampling temperature (default: 0.3)

    Returns:
        List of enriched QA pairs
    """
    enriched = []

    for pair in pairs:
        try:
            # Format prompt with question and answer
            prompt = prompt_template.format(
                question=pair["question"], answer=pair["answer"]
            )

            # Get enriched response from LLM
            response = llm.generate(prompt, temperature=temperature)

            # Parse JSON response
            try:
                result = json5.loads(response)
            except Exception:
                # If JSON parsing fails, try to extract JSON from markdown code blocks
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    result = json5.loads(json_str)
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    result = json5.loads(json_str)
                else:
                    raise

            # Create enriched pair
            enriched_pair = {
                "question": result.get("question", pair["question"]),
                "answer": result.get("enriched_answer", pair["answer"]),
                "enrichment_changes": result.get("changes", "No changes tracked"),
            }

            # Preserve original if requested
            if preserve_original:
                enriched_pair["original_answer"] = pair["answer"]

            # Copy any additional fields (like rating, metadata, etc.)
            for key, value in pair.items():
                if key not in ["question", "answer"]:
                    enriched_pair[key] = value

            enriched.append(enriched_pair)

        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to enrich pair, keeping original: {e}[/yellow]"
            )
            # On failure, keep original pair
            enriched.append(pair)

    return enriched


def load_qa_pairs(input_file: Path) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(input_file) as f:
        return json.load(f)


def save_qa_pairs(qa_pairs: List[Dict], output_file: Path) -> None:
    """Save QA pairs to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    console.print(f"[green]✓ Saved enriched pairs to {output_file}[/green]")
