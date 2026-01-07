"""
QA pair curation using LLM-as-Judge.

Implements quality filtering based on:
- AlpaGasus (2023): LLM rates pairs 1-10
- LIMA (2023): High threshold (≥7.0) for quality
"""

import json
import json5
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from .clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)

def _detect_format(data: Any) -> str:
    """Detect input format: 'qa', 'cot', or 'unknown'."""
    if isinstance(data, list):
        if len(data) == 0:
            return "qa"  # Default to QA for empty list

        first_item = data[0]
        # Check for CoT format (has reasoning/cot/thinking fields)
        if any(key in first_item for key in ["reasoning", "cot", "thinking", "thought"]):
            return "cot"
        # Check if already in conversation format
        elif "conversations" in first_item or "messages" in first_item:
            return "conversation"
        # Check for QA format
        elif "question" in first_item and "answer" in first_item:
            return "qa"

    return "qa"  # Default


def _convert_to_conversation_format(pairs: List[Dict], format_type: str) -> List[Dict]:
    """Convert QA or CoT to internal conversation format for uniform processing."""
    conversations = []

    for pair in pairs:
        # Skip malformed pairs (not dict or missing keys)
        if not isinstance(pair, dict):
            logger.warning(f"Skipping malformed pair (not a dict): {type(pair)}")
            continue
            
        if format_type == "qa":
            # QA format: question + answer
            if "question" not in pair or "answer" not in pair:
                logger.warning(f"Skipping QA pair missing question/answer keys: {list(pair.keys())}")
                continue
                
            conversations.append({
                "conversations": [
                    {"role": "user", "content": pair.get("question", "")},
                    {"role": "assistant", "content": pair.get("answer", "")}
                ],
                "_original_pair": pair,  # Preserve original for restoration
                "_format": "qa"
            })
        elif format_type == "cot":
            # CoT format: may have reasoning steps
            reasoning = pair.get("reasoning", pair.get("cot", pair.get("thinking", "")))
            question = pair.get("question", pair.get("prompt", ""))
            answer = pair.get("answer", pair.get("response", ""))

            # Combine reasoning + answer for CoT
            full_answer = f"{reasoning}\n\n{answer}" if reasoning else answer

            conversations.append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_answer}
                ],
                "_original_pair": pair,
                "_format": "cot"
            })
        else:
            # Already in conversation format or unknown
            conversations.append(pair)

    return conversations


def _extract_qa_from_conversation(conv: Dict) -> Dict:
    """Extract QA pair from conversation format."""
    if "conversations" in conv:
        messages = conv["conversations"]
    elif "messages" in conv:
        messages = conv["messages"]
    else:
        return {"question": "", "answer": ""}

    question = ""
    answer = ""

    for msg in messages:
        # Skip if msg is not a dict (safety check)
        if not isinstance(msg, dict):
            continue
            
        if msg.get("role") in ["user", "human"]:
            question = msg.get("content", msg.get("value", ""))
        elif msg.get("role") in ["assistant", "gpt"]:
            answer = msg.get("content", msg.get("value", ""))

    return {"question": question, "answer": answer}


def _restore_original_format(conversations: List[Dict], original_format: str) -> List[Dict]:
    """Restore to original format (QA or CoT) after curation."""
    restored = []

    for conv in conversations:
        original_pair = conv.get("_original_pair", {})
        format_type = conv.get("_format", original_format)

        if format_type == "qa":
            # Restore QA format
            qa = _extract_qa_from_conversation(conv)
            # Merge with original metadata and rating info
            result = {**original_pair, **qa}
            # Add rating fields if present
            for key in ["rating", "clarity", "accuracy", "usefulness", "difficulty", "reasoning"]:
                if key in conv:
                    result[key] = conv[key]
            restored.append(result)
        elif format_type == "cot":
            # Restore CoT format (preserve reasoning field)
            qa = _extract_qa_from_conversation(conv)
            result = {**original_pair}
            result["question"] = qa["question"]
            # Keep reasoning separate from answer in CoT format
            if "reasoning" in original_pair:
                result["reasoning"] = original_pair["reasoning"]
            result["answer"] = qa["answer"]
            # Add rating fields
            for key in ["rating", "clarity", "accuracy", "usefulness", "difficulty", "reasoning"]:
                if key in conv and key not in result:  # Don't overwrite CoT reasoning
                    result[key] = conv[key]
            restored.append(result)
        else:
            # Unknown format, keep as-is
            restored.append(conv)

    return restored


def curate_qa_pairs(
    input_path: str,
    output_path: str,
    prompts: Dict[str, str],
    llm_config: Dict[str, Any],
    threshold: float = 7.0,
    batch_size: int = 5,
    topic_filter: str = None,
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
        topic_filter: Optional topic (e.g., 'HDF5') to filter irrelevant pairs

    Returns:
        Dict with metrics (total, filtered, retention_rate, avg_score)
    """
    console.print(f"\n[bold cyan]📊 Loading data from: {input_path}[/bold cyan]")

    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect format
    original_format = _detect_format(data)
    console.print(f"[cyan]→ Detected format: {original_format}[/cyan]")

    # Convert to conversation format for uniform processing
    console.print("[cyan]→ Converting to conversation format...[/cyan]")
    conversations = _convert_to_conversation_format(data, original_format)

    total_pairs = len(conversations)
    console.print(f"[green]✓ Loaded {total_pairs} pairs[/green]\n")

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
        task = progress.add_task("[cyan]Rating pairs...", total=total_pairs)

        for i in range(0, total_pairs, batch_size):
            batch = conversations[i : i + batch_size]

            try:
                # Extract QA for rating
                qa_batch = [_extract_qa_from_conversation(conv) for conv in batch]

                # Rate this batch
                temperature = llm_config.get('temperature', 0.1) if isinstance(llm_config, dict) else 0.1
                rated_batch = _rate_batch(
                    pairs=qa_batch, llm=llm, prompt_template=rating_prompt_template, temperature=temperature, topic_filter=topic_filter
                )

                # Merge ratings back to conversation format
                for j, conv in enumerate(batch):
                    if j < len(rated_batch):
                        rating = rated_batch[j].get("rating", 0)
                        total_score += rating

                        # Copy rating fields to conversation
                        for key in ["rating", "clarity", "accuracy", "usefulness", "difficulty", "reasoning", "topic_relevant"]:
                            if key in rated_batch[j]:
                                conv[key] = rated_batch[j][key]

                        rated_pairs.append(conv)

                        # Filter by threshold AND topic relevance
                        topic_relevant = rated_batch[j].get("topic_relevant", True)
                        if rating >= threshold and topic_relevant:
                            filtered_pairs.append(conv)

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
        "original_format": original_format,
        "curated_at": datetime.now().isoformat(),
    }

    # Restore to original format (QA or CoT)
    console.print(f"[cyan]→ Restoring to original format ({original_format})...[/cyan]")
    curated_restored = _restore_original_format(filtered_pairs, original_format)
    all_rated_restored = _restore_original_format(rated_pairs, original_format)

    # Save results
    _save_curated_results(
        curated_pairs=curated_restored, all_rated=all_rated_restored, metrics=metrics, output_path=output_file
    )

    console.print(f"\n[bold green]✓ Curated {len(filtered_pairs)} / {total_pairs} pairs[/bold green]")
    console.print(f"[bold green]✓ Retention rate: {retention_rate:.1%}[/bold green]")
    console.print(f"[bold green]✓ Average score: {avg_score:.2f}/10[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]\n")

    return metrics


def _rate_batch(
    pairs: List[Dict], llm: BaseLLMClient, prompt_template: str, temperature: float = 0.1, topic_filter: str = None
) -> List[Dict]:
    """Rate a batch of QA pairs with detailed criteria breakdown."""
    # Format pairs for prompt
    pairs_str = json.dumps(
        [{"question": p["question"], "answer": p["answer"]} for p in pairs],
        indent=2,
        ensure_ascii=False,
    )

    # Add topic filter instruction if provided
    topic_instruction = ""
    if topic_filter:
        topic_instruction = f"\n\n**TOPIC FILTER:** Only keep pairs directly related to **{topic_filter}**. Mark pairs as topic_relevant: false if they are off-topic or not directly about {topic_filter}."

    # Fill prompt
    prompt = prompt_template.format(pairs=pairs_str) + topic_instruction

    # Generate ratings
    response = llm.generate(prompt, temperature=temperature)

    # Clean response - check for markdown wrapping FIRST
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.split("```json", 1)[1].split("```")[0].strip()
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response.split("```", 1)[1].split("```")[0].strip()

    # Parse JSON response
    try:
        rated = json5.loads(cleaned_response)
    except Exception:
        # Fallback: assign default rating
        rated = [
            {
                "question": p["question"],
                "answer": p["answer"],
                "rating": 5,
                "clarity": 2,
                "accuracy": 2,
                "usefulness": 1,
                "difficulty": 0,
                "reasoning": "Default rating (parsing failed)",
            }
            for p in pairs
            ]

    # Merge ratings with original pairs
    result = []
    for i, pair in enumerate(pairs):
        if i < len(rated):
            # Copy all rating details
            pair["rating"] = rated[i].get("rating", 5)
            pair["clarity"] = rated[i].get("clarity", 2)
            pair["accuracy"] = rated[i].get("accuracy", 2)
            pair["usefulness"] = rated[i].get("usefulness", 1)
            pair["topic_relevant"] = rated[i].get("topic_relevant", True)
            pair["difficulty"] = rated[i].get("difficulty", 0)
            pair["reasoning"] = rated[i].get("reasoning", "No reasoning provided")
        else:
            # Default if not enough ratings
            pair["rating"] = 5
            pair["clarity"] = 2
            pair["accuracy"] = 2
            pair["usefulness"] = 1
            pair["difficulty"] = 0
            pair["reasoning"] = "Default rating (insufficient results)"
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
