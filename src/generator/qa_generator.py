"""
QA pair generation from LanceDB chunks.

Uses Instruction Backtranslation methodology:
- Treat document chunks as "answers"
- Generate "questions" for each chunk
- Paper: Self-Alignment with Instruction Backtranslation (Meta AI, ICLR 2024)
"""

import json
import json5
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import lancedb  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from .clients import get_client, BaseLLMClient

console = Console()


def generate_qa_from_lancedb(
    db_path: str,
    output_path: str,
    prompts: Dict[str, str],
    llm_config: Dict[str, Any],
    table_name: str = "text_chunks",
    n_pairs_per_chunk: Optional[int] = None,
    target_pairs: Optional[int] = None,
    batch_size: int = 50,
    max_chunks: Optional[int] = None,
) -> List[Dict]:
    """
    Generate QA pairs from LanceDB chunks.

    Args:
        db_path: Path to LanceDB directory
        output_path: Where to save generated QA pairs
        prompts: Prompt templates dict
        llm_config: LLM configuration dict
        table_name: LanceDB table name ("text_chunks" or "code_chunks")
        n_pairs_per_chunk: Fixed number per chunk (if specified)
        target_pairs: Total pairs target (calculates per-chunk if specified)
        batch_size: Number of chunks to process in one batch
        max_chunks: Optional limit on chunks to process (for testing)

    Returns:
        List of generated QA pairs with metadata
    """
    console.print(f"\n[bold cyan]📊 Connecting to LanceDB: {db_path}[/bold cyan]")

    # Connect to LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    total_chunks = table.count_rows()
    if max_chunks:
        total_chunks = min(total_chunks, max_chunks)

    console.print(f"[green]✓ Found {total_chunks} chunks in '{table_name}'[/green]")

    # Calculate pairs per chunk from target (if specified)
    if target_pairs and not n_pairs_per_chunk:
        n_pairs_per_chunk = max(1, round(target_pairs / total_chunks))
        console.print(f"[cyan]→ Target: {target_pairs} pairs → {n_pairs_per_chunk} per chunk[/cyan]")
    elif not n_pairs_per_chunk:
        n_pairs_per_chunk = 3  # Default
        console.print(f"[cyan]→ Using default: {n_pairs_per_chunk} pairs per chunk[/cyan]")
    else:
        console.print(f"[cyan]→ Fixed: {n_pairs_per_chunk} pairs per chunk[/cyan]")

    console.print()

    # Initialize LLM client
    console.print(f"[bold cyan]🤖 Initializing LLM: {llm_config['provider']}[/bold cyan]")
    provider = llm_config.pop("provider")
    llm = get_client(provider, llm_config)
    console.print(f"[green]✓ LLM ready: {llm.model}[/green]\n")

    # Get prompt template
    qa_prompt_template = prompts.get("qa_generation")
    if not qa_prompt_template:
        raise ValueError("qa_generation prompt not found in prompts config")

    all_qa_pairs = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process chunks in batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating QA pairs...", total=total_chunks)

        # Read chunks in batches
        offset = 0
        while offset < total_chunks:
            # Get batch of chunks
            limit = min(batch_size, total_chunks - offset)
            df = table.to_pandas()[offset : offset + limit]

            for idx, row in df.iterrows():
                chunk_id = row.get("id", f"chunk_{offset + idx}")
                content = row.get("content", "")
                source_file = row.get("source_file", "unknown")

                # Skip very short chunks (headers, citations, etc.)
                if not content or len(content.strip()) < 200:
                    progress.advance(task)
                    continue

                # Generate QA pairs for this chunk with retry logic
                try:
                    pairs = _generate_pairs_for_chunk(
                        content=content,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        llm=llm,
                        prompt_template=qa_prompt_template,
                        n_pairs=n_pairs_per_chunk,
                    )

                    all_qa_pairs.extend(pairs)

                    # Rate limiting: 6 seconds between requests (max 10/min for Gemini free tier)
                    time.sleep(6)

                    # Save intermediate results every 10 chunks
                    if len(all_qa_pairs) % 50 == 0:
                        _save_intermediate(all_qa_pairs, output_file)

                except Exception as e:
                    console.print(f"[yellow]⚠ Error on chunk {chunk_id}: {e}[/yellow]")

                progress.advance(task)

            offset += limit

    # Final save
    _save_results(all_qa_pairs, output_file)

    console.print(f"\n[bold green]✓ Generated {len(all_qa_pairs)} QA pairs[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]\n")

    return all_qa_pairs


def _generate_pairs_for_chunk(
    content: str,
    chunk_id: str,
    source_file: str,
    llm: BaseLLMClient,
    prompt_template: str,
    n_pairs: int,
    max_retries: int = 3,
) -> List[Dict]:
    """Generate QA pairs for a single chunk with retry logic."""
    # Fill in prompt template
    prompt = prompt_template.format(text=content, n_pairs=n_pairs)

    # Generate response with exponential backoff retry
    last_error: Exception = ValueError("Max retries exhausted")
    for attempt in range(max_retries):
        try:
            response = llm.generate(prompt)

            # Check for empty response before proceeding
            if not response or not response.strip():
                raise ValueError("LLM returned empty response")

            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Don't retry empty responses or JSON parsing errors
            if 'empty' in error_msg or 'json' in error_msg:
                raise ValueError(f"Invalid response (won't retry): {str(e)[:100]}")

            # Check for daily quota exhaustion (stop immediately, don't retry)
            if 'per_day' in error_msg or 'daily' in error_msg or 'generate_requests_per_model_per_day' in str(e):
                raise ValueError("Daily quota exceeded. Try again tomorrow or switch to gemini-1.5-flash or gemini-2.5-flash-image")

            # Check if it's a per-minute rate limit error (429) - these can be retried
            elif '429' in error_msg or 'quota' in error_msg or 'rate limit' in error_msg:
                # Exponential backoff: 60s, 120s, 240s
                wait_time = 60 * (2 ** attempt)
                console.print(f"[yellow]⚠ Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...[/yellow]")
                time.sleep(wait_time)
            else:
                # Non-rate-limit error, don't retry
                raise
    else:
        # All retries exhausted
        raise last_error

    # Parse JSON response (lenient with json5)
    try:
        pairs = json5.loads(response)
    except Exception as e:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            pairs = json5.loads(json_str)
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            pairs = json5.loads(json_str)
        else:
            # Try to extract JSON array from response with extra text
            # Find first [ and last ]
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                try:
                    pairs = json5.loads(json_str)
                except Exception as e2:
                    raise ValueError(f"Failed to parse JSON response: {e}\nExtraction also failed: {e2}\nResponse: {response[:200]}")
            else:
                raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response[:200]}")

    # Add metadata to each pair
    for pair in pairs:
        pair["chunk_id"] = chunk_id
        pair["source_file"] = source_file
        pair["generated_at"] = datetime.now().isoformat()
        pair["model"] = llm.model

    return pairs  # type: ignore[no-any-return]


def _save_intermediate(qa_pairs: List[Dict], output_path: Path):
    """Save intermediate results."""
    intermediate_path = output_path.parent / f"{output_path.stem}_intermediate.json"
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)


def _save_results(qa_pairs: List[Dict], output_path: Path):
    """Save final results with metadata."""
    # Save main file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    # Save JSONL format (one pair per line)
    jsonl_path = output_path.parent / f"{output_path.stem}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Save metadata summary
    summary = {
        "total_pairs": len(qa_pairs),
        "generated_at": datetime.now().isoformat(),
        "sources": list(set(p.get("source_file", "unknown") for p in qa_pairs)),
        "models_used": list(set(p.get("model", "unknown") for p in qa_pairs)),
    }

    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
