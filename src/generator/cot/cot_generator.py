"""
CoT (Chain-of-Thought) Generator

Generates QA pairs with step-by-step reasoning from LanceDB chunks.
Based on "Distilling Step-by-Step" methodology (Google, 2023).

Paper: https://arxiv.org/abs/2305.02301
"""

import json
import json5
import lancedb
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..prompt_loader import load_prompts

logger = logging.getLogger(__name__)


def generate_cot_pairs(
    lancedb_path: str,
    output_path: str,
    llm_config: Dict,
    table_name: str = "text_chunks",
    n_pairs: Optional[int] = None,
    target_pairs: Optional[int] = None,
    batch_size: int = 50,
    max_chunks: Optional[int] = None,
    workers: int = 1,
) -> Dict:
    """
    Generate CoT (Chain-of-Thought) pairs from LanceDB chunks.
    
    Args:
        lancedb_path: Path to LanceDB database
        output_path: Path to save generated CoT pairs
        llm_config: LLM configuration dict
        table_name: Name of the LanceDB table
        n_pairs: Fixed number of CoT pairs per chunk (mutually exclusive with target_pairs)
        target_pairs: Total target number of pairs (auto-calculates per-chunk)
        batch_size: Number of chunks to process per batch
        max_chunks: Maximum number of chunks to process (for testing)
        workers: Number of parallel workers (1=sequential, 4+ recommended for Ollama)
    
    Returns:
        Dict with summary statistics
    """
    from ..clients import get_client

    # Validate n_pairs and target_pairs
    if n_pairs is not None and target_pairs is not None:
        raise ValueError("Cannot specify both --n-pairs and --target-pairs. Choose one.")
    if n_pairs is None and target_pairs is None:
        n_pairs = llm_config.get("generation", {}).get("n_pairs_per_chunk", 3)

    # Load prompt template
    config_dir = Path(__file__).parent.parent.parent / "configs"
    prompts = load_prompts(config_dir)
    cot_prompt = prompts.get("cot_generation")
    if not cot_prompt:
        raise ValueError("CoT generation prompt not found in prompts")

    # Connect to LanceDB
    logger.info(f"Connecting to LanceDB at {lancedb_path}")
    db = lancedb.connect(lancedb_path)
    table = db.open_table(table_name)

    # Get chunks (filter out empty content)
    chunks_df = table.to_pandas()
    # Try 'content' first (standard), fallback to 'text'
    text_col = 'content' if 'content' in chunks_df.columns else 'text'
    chunks_df = chunks_df[chunks_df[text_col].str.strip() != ""]

    if max_chunks:
        chunks_df = chunks_df.head(max_chunks)

    total_chunks = len(chunks_df)
    logger.info(f"Found {total_chunks} non-empty chunks")

    if total_chunks == 0:
        raise ValueError("No valid chunks found in LanceDB")

    # Calculate pairs per chunk if using target_pairs
    if target_pairs:
        pairs_per_chunk = max(1, round(target_pairs / total_chunks))
        logger.info(f"Target pairs: {target_pairs}, Pairs per chunk: {pairs_per_chunk}")
    else:
        pairs_per_chunk = n_pairs
        logger.info(f"Fixed pairs per chunk: {pairs_per_chunk}")

    # Store config values before popping (use llm_config since config is not defined)
    skip_patterns = llm_config.get('filtering', {}).get('skip_source_patterns', ['_log.md', 'login.md', 'retrieval_progress', 'signup'])
    
    # Initialize LLM client
    provider = llm_config.pop("provider")
    client = get_client(provider, llm_config)

    # Generate CoT pairs
    all_cot_pairs = []

    # Prepare chunks for processing
    chunks_to_process = []
    for idx, row in chunks_df.iterrows():
        chunk_text = row[text_col]
        chunk_id = row.get("id", f"chunk_{idx}")
        source = row.get("source", "unknown")
        source_file = row.get("source_file", "unknown")

        # Skip filtered source files (logs, login pages, metadata)
        if any(pattern in source_file for pattern in skip_patterns):
            continue

        chunks_to_process.append({
            'text': chunk_text,
            'chunk_id': chunk_id,
            'source': source,
            'source_file': source_file
        })

    logger.info(f"Processing {len(chunks_to_process)} chunks with {workers} workers")

    if workers == 1:
        # Sequential processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Generating CoT pairs ({pairs_per_chunk}/chunk)...",
                total=len(chunks_to_process),
            )

            for chunk in chunks_to_process:
                try:
                    pairs = _generate_cot_for_chunk(
                        chunk['text'], chunk['chunk_id'], chunk['source'],
                        client, cot_prompt, pairs_per_chunk
                    )
                    all_cot_pairs.extend(pairs)
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Generated {len(all_cot_pairs)} CoT pairs...",
                    )

                    if target_pairs and len(all_cot_pairs) >= target_pairs:
                        logger.info(f"Reached target of {target_pairs} pairs, stopping...")
                        break

                except Exception as e:
                    logger.warning(f"Failed for chunk {chunk['chunk_id']}: {e}")
                    progress.update(task, advance=1)
    else:
        # Parallel processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Generating CoT pairs ({pairs_per_chunk}/chunk)...",
                total=len(chunks_to_process),
            )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for chunk in chunks_to_process:
                    future = executor.submit(
                        _generate_cot_for_chunk,
                        chunk['text'], chunk['chunk_id'], chunk['source'],
                        client, cot_prompt, pairs_per_chunk
                    )
                    futures[future] = chunk['chunk_id']

                completed = 0
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        pairs = future.result()
                        all_cot_pairs.extend(pairs)
                        completed += 1
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Generated {len(all_cot_pairs)} CoT pairs... ({completed}/{len(chunks_to_process)})",
                        )

                        if target_pairs and len(all_cot_pairs) >= target_pairs:
                            logger.info(f"Reached target of {target_pairs} pairs, stopping...")
                            break

                    except Exception as e:
                        logger.warning(f"Failed for chunk {chunk_id}: {e}")
                        completed += 1
                        progress.update(task, advance=1)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_cot_pairs, f, indent=2)

    logger.info(f"Saved {len(all_cot_pairs)} CoT pairs to {output_path}")

    return {
        "total_chunks_processed": min(total_chunks, len(chunks_df)),
        "cot_pairs_generated": len(all_cot_pairs),
        "pairs_per_chunk": pairs_per_chunk,
        "output_file": str(output_path),
    }


def _generate_cot_for_chunk(text: str, chunk_id: str, source: str, client, cot_prompt, n_pairs: int) -> List[Dict]:
    """Generate CoT pairs for a single chunk."""
    # Use wiggle room like QA generator (adaptive pairs per chunk)
    n_pairs_min = n_pairs
    n_pairs_max = min(n_pairs * 2, 10)  # Cap at 10 max
    prompt = cot_prompt.format(text=text, n_pairs_min=n_pairs_min, n_pairs_max=n_pairs_max)
    response = client.generate(prompt)
    pairs = _parse_cot_response(response)
    
    # Add metadata
    for pair in pairs:
        pair["chunk_id"] = chunk_id
        pair["source"] = source
    
    return pairs


def _parse_cot_response(response: str) -> List[Dict]:
    """
    Parse CoT generation response into structured pairs.
    
    Expected format:
    [
      {
        "question": "...",
        "reasoning": "Step 1: ...\nStep 2: ...",
        "answer": "..."
      }
    ]
    """
    # Try json5 first (handles trailing commas, comments)
    try:
        # Extract JSON array from response
        start_idx = response.find("[")
        end_idx = response.rfind("]")

        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON array found in response")
            return []

        json_str = response[start_idx : end_idx + 1]
        pairs = json5.loads(json_str)

        if not isinstance(pairs, list):
            logger.warning("Response is not a list")
            return []

        # Validate structure
        valid_pairs = []
        for pair in pairs:
            if (
                isinstance(pair, dict)
                and "question" in pair
                and "reasoning" in pair
                and "answer" in pair
            ):
                valid_pairs.append(pair)
            else:
                logger.warning(f"Invalid CoT pair structure: {pair}")

        return valid_pairs

    except Exception as e:
        logger.error(f"Failed to parse CoT response: {e}")
        return []


def load_cot_pairs(input_path: str) -> List[Dict]:
    """Load CoT pairs from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def save_cot_pairs(pairs: List[Dict], output_path: str):
    """Save CoT pairs to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
