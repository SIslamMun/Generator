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
from concurrent.futures import ThreadPoolExecutor, as_completed
import lancedb  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from ..clients import get_client, BaseLLMClient

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
    chunk_ids: Optional[List[str]] = None,
    workers: int = 1,
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
        chunk_ids: Optional list of specific chunk IDs to process
        workers: Number of parallel workers (1=sequential, 4=recommended for Ollama)

    Returns:
        List of generated QA pairs with metadata
    """
    console.print(f"\n[bold cyan]📊 Connecting to LanceDB: {db_path}[/bold cyan]")

    # Connect to LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # Filter by chunk IDs if provided
    if chunk_ids:
        total_chunks = len(chunk_ids)
        console.print(f"[cyan]→ Filtering to {total_chunks} specific chunks[/cyan]")
    elif max_chunks:
        total_chunks = min(table.count_rows(), max_chunks)
    else:
        total_chunks = table.count_rows()

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
    # Start timing
    start_time = time.time()
    
    console.print(f"[bold cyan]🤖 Initializing LLM: {llm_config['provider']}[/bold cyan]")
    
    # Store config values before popping
    skip_patterns = llm_config.get('skip_source_patterns', ['_log.md', 'login.md', 'retrieval_progress', 'signup'])
    min_length = llm_config.get('min_chunk_length', 200)
    delay = llm_config.get('rate_limit', {}).get('delay_between_requests', 6)
    save_freq = llm_config.get('rate_limit', {}).get('save_every_n_pairs', 50)
    backoff_base = llm_config.get('rate_limit', {}).get('backoff_base', 60)
    max_retries = llm_config.get('max_retries', 3)
    
    console.print(f"[dim]→ Rate limit delay: {delay}s, Save every: {save_freq} pairs[/dim]")
    
    provider = llm_config.pop("provider")
    llm = get_client(provider, llm_config)
    console.print(f"[green]✓ LLM ready: {llm.model}[/green]\n")

    # Get prompt template - auto-detect based on table type
    if "code" in table_name.lower():
        qa_prompt_template = prompts.get("code_qa_generation") or prompts.get("qa_generation")
        prompt_type = "code" if "code_qa_generation" in prompts else "text (fallback)"
    else:
        qa_prompt_template = prompts.get("qa_generation")
        prompt_type = "text"
    
    if not qa_prompt_template:
        raise ValueError("qa_generation prompt not found in prompts config")
    
    console.print(f"[dim]→ Using {prompt_type} prompt template[/dim]")

    all_qa_pairs = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Resume from intermediate file if exists
    intermediate_file = output_file.parent / f"{output_file.stem}_intermediate.json"
    processed_chunk_ids = set()
    if intermediate_file.exists():
        try:
            import json
            with open(intermediate_file) as f:
                existing_pairs = json.load(f)
            if existing_pairs:
                all_qa_pairs = existing_pairs
                processed_chunk_ids = {p.get("chunk_id") for p in existing_pairs if p.get("chunk_id")}
                console.print(f"[yellow]⚡ Resuming: Found {len(all_qa_pairs)} existing pairs from {len(processed_chunk_ids)} chunks[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load intermediate file: {e}[/yellow]")

    # Process chunks - choose sequential or parallel based on workers parameter
    if workers == 1:
        # Sequential processing (original behavior)
        all_qa_pairs = _process_chunks_sequential(
            table=table,
            total_chunks=total_chunks,
            chunk_ids=chunk_ids,
            batch_size=batch_size,
            processed_chunk_ids=processed_chunk_ids,
            skip_patterns=skip_patterns,
            min_length=min_length,
            llm=llm,
            qa_prompt_template=qa_prompt_template,
            n_pairs_per_chunk=n_pairs_per_chunk,
            max_retries=max_retries,
            backoff_base=backoff_base,
            delay=delay,
            save_freq=save_freq,
            output_file=output_file,
            all_qa_pairs=all_qa_pairs,
        )
    else:
        # Parallel processing
        console.print(f"[yellow]⚡ Using parallel processing with {workers} workers[/yellow]")
        all_qa_pairs = _process_chunks_parallel(
            table=table,
            total_chunks=total_chunks,
            chunk_ids=chunk_ids,
            processed_chunk_ids=processed_chunk_ids,
            skip_patterns=skip_patterns,
            min_length=min_length,
            llm=llm,
            qa_prompt_template=qa_prompt_template,
            n_pairs_per_chunk=n_pairs_per_chunk,
            max_retries=max_retries,
            backoff_base=backoff_base,
            save_freq=save_freq,
            output_file=output_file,
            all_qa_pairs=all_qa_pairs,
            workers=workers,
        )

    # Final save
    _save_results(all_qa_pairs, output_file)

    # Calculate total time
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    console.print(f"\n[bold green]✓ Generated {len(all_qa_pairs)} QA pairs[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]")
    console.print(f"[bold cyan]⏱️  Total time: {time_str}[/bold cyan]\n")

    return all_qa_pairs


def _generate_pairs_for_chunk(
    content: str,
    chunk_id: str,
    source_file: str,
    llm: BaseLLMClient,
    prompt_template: str,
    n_pairs: int,
    max_retries: int = 3,
    backoff_base: int = 60,
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
                # Exponential backoff using config or default (60s, 120s, 240s...)
                wait_time = backoff_base * (2 ** attempt)
                console.print(f"[yellow]⚠ Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...[/yellow]")
                time.sleep(wait_time)
            else:
                # Non-rate-limit error, don't retry
                raise
    else:
        # All retries exhausted
        raise last_error

    # Check for empty or invalid response
    if not response or not response.strip():
        raise ValueError(f"Empty response from LLM for chunk {chunk_id}")
    
    # Parse JSON response (lenient with json5)
    try:
        pairs = json5.loads(response)
    except Exception as e:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            if not json_str:
                raise ValueError(f"Empty JSON in markdown code block for chunk {chunk_id}")
            try:
                pairs = json5.loads(json_str)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON from markdown: {e2}\nJSON: {json_str[:300]}")
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            if not json_str:
                raise ValueError(f"Empty JSON in code block for chunk {chunk_id}")
            try:
                pairs = json5.loads(json_str)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON from code block: {e2}\nJSON: {json_str[:300]}")
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
                    # Last attempt: try standard json parser which is stricter
                    import json
                    try:
                        pairs = json.loads(json_str)
                    except Exception as e3:
                        raise ValueError(f"Failed to parse JSON response: {e}\nExtraction failed: {e2}\nStandard JSON failed: {e3}\nExtracted JSON: {json_str[:300]}")
            else:
                raise ValueError(f"Failed to parse JSON response: {e}\nNo JSON array found in response: {response[:300]}")

    # Ensure pairs is a list (sometimes LLM returns a single dict)
    if isinstance(pairs, dict):
        pairs = [pairs]
    elif not isinstance(pairs, list):
        raise ValueError(f"Invalid response format: expected list or dict, got {type(pairs).__name__}")
    
    # Flatten if nested list (some LLMs return [[{...}]] instead of [{...}])
    if pairs and isinstance(pairs[0], list):
        pairs = pairs[0]
    
    # Validate we got some pairs
    if not pairs or len(pairs) == 0:
        raise ValueError(f"No QA pairs generated for chunk {chunk_id}")

    # Add metadata to each pair
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"Invalid pair format: expected dict, got {type(pair).__name__}. Pairs: {pairs[:2]}")
        pair["chunk_id"] = chunk_id
        pair["source_file"] = source_file
        pair["generated_at"] = datetime.now().isoformat()
        pair["model"] = llm.model

    # NOTE: Filtering disabled - let all generated pairs through
    # pairs = _filter_infrastructure_pairs(pairs)
    # pairs = _filter_build_questions(pairs)

    return pairs  # type: ignore[no-any-return]


def _filter_infrastructure_pairs(pairs: List[Dict]) -> List[Dict]:
    """
    Filter out infrastructure/CI/CD focused pairs.

    Removes pairs that heavily focus on GitHub workflows, build systems,
    CI/CD configurations, or repository structure.

    Args:
        pairs: List of QA pairs to filter

    Returns:
        Filtered list with infrastructure pairs removed
    """
    infrastructure_keywords = [
        "github actions", "workflow", "ci/cd", "pipeline job",
        "devcontainer", "cmake", "build system", "test script",
        "operating system", "runs on", "docker", "container",
        "ubuntu-latest", "windows-latest", "macos-latest",
        ".yml file", "yaml file", "workflow trigger", "job depend",
        "build step", "test framework", "makefile", "ctest"
    ]

    filtered = []
    for pair in pairs:
        question_lower = pair.get("question", "").lower()
        answer_lower = pair.get("answer", "").lower()

        # Count infrastructure keyword mentions
        infra_mentions = sum(
            1 for kw in infrastructure_keywords
            if kw in question_lower or kw in answer_lower
        )

        # Skip if too infrastructure-heavy (more than 1 mention)
        if infra_mentions <= 1:
            filtered.append(pair)

    return filtered


def _filter_build_questions(pairs: List[Dict]) -> List[Dict]:
    """
    Filter out build/compilation/installation focused pairs.

    Removes pairs about compiling source code, installation steps,
    repository cloning, or version management without technical context.

    Args:
        pairs: List of QA pairs to filter

    Returns:
        Filtered list with build/installation pairs removed
    """
    build_keywords = [
        "compile", "compilation", "build the", "building the",
        "install the", "installation of", "installing the",
        "clone the repository", "pull the branch", "checkout the",
        "commit hash", "version number", "release tag",
        "from the repository", "from the source code",
        "what happens when you compile", "what happens when you build",
        "how do you install", "how to install", "steps to install",
        "download the source", "extract the archive",
        "gcc", "clang", "toolchain", "compiler flag"
    ]

    filtered = []
    for pair in pairs:
        question_lower = pair.get("question", "").lower()
        answer_lower = pair.get("answer", "").lower()

        # Check if question is build/installation focused
        is_build_question = any(
            kw in question_lower for kw in build_keywords
        )

        # Additional check: if answer mentions build without technical API/library context
        has_build_without_context = False
        if any(kw in answer_lower for kw in ["compile", "compilation", "build"]):
            # Check if it's technical (has function calls, API references, etc.)
            technical_indicators = ["function", "api", "method", "class", "()", "parameter"]
            has_technical_context = any(ind in answer_lower for ind in technical_indicators)
            if not has_technical_context:
                has_build_without_context = True

        # Skip if it's a build/installation question without technical merit
        if not (is_build_question or has_build_without_context):
            filtered.append(pair)

    return filtered


def _save_intermediate(qa_pairs: List[Dict], output_path: Path):
    """Save intermediate results."""
    intermediate_path = output_path.parent / f"{output_path.stem}_intermediate.json"
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]💾 Saved {len(qa_pairs)} pairs to intermediate file[/dim]")


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


def _process_chunks_sequential(
    table, total_chunks, chunk_ids, batch_size, processed_chunk_ids,
    skip_patterns, min_length, llm, qa_prompt_template, n_pairs_per_chunk,
    max_retries, backoff_base, delay, save_freq, output_file, all_qa_pairs
):
    """Sequential processing (original behavior)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating QA pairs...", total=total_chunks)
        processed = 0
        offset = 0
        
        while offset < total_chunks:
            limit = min(batch_size, total_chunks - offset)
            
            if chunk_ids:
                df = table.to_pandas()
                df = df[df["id"].isin(chunk_ids[offset : offset + limit])]
            else:
                df = table.to_pandas()[offset : offset + limit]

            for idx, row in df.iterrows():
                chunk_id = row.get("id", f"chunk_{offset + idx}")
                content = row.get("content", "")
                source_file = row.get("source_file", "unknown")

                if chunk_id in processed_chunk_ids:
                    progress.advance(task)
                    processed += 1
                    continue

                if any(pattern in source_file for pattern in skip_patterns):
                    progress.advance(task)
                    continue

                metadata_patterns = ['readme', 'github.com', 'gitlab.com', 'bitbucket.org',
                                   'license', 'contributing', 'changelog', 'authors']
                content_preview = content[:300].lower()
                if any(pattern in source_file.lower() or pattern in content_preview
                       for pattern in metadata_patterns):
                    repo_keywords = ['repository', 'clone', 'branch', 'commit', 'pull request', 'fork']
                    if any(kw in content_preview for kw in repo_keywords):
                        progress.advance(task)
                        continue

                if not content or len(content.strip()) < min_length:
                    progress.advance(task)
                    continue

                try:
                    pairs = _generate_pairs_for_chunk(
                        content=content,
                        chunk_id=chunk_id,
                        source_file=source_file,
                        llm=llm,
                        prompt_template=qa_prompt_template,
                        n_pairs=n_pairs_per_chunk,
                        max_retries=max_retries,
                        backoff_base=backoff_base,
                    )
                    all_qa_pairs.extend(pairs)
                    time.sleep(delay)
                    if len(all_qa_pairs) % save_freq == 0:
                        _save_intermediate(all_qa_pairs, output_file)
                except Exception as e:
                    console.print(f"[red]✗ FAILED: {chunk_id} - {str(e)[:100]}[/red]")

                processed += 1
                remaining = total_chunks - processed
                progress.update(task, description=f"[cyan]Generating QA pairs... ({len(all_qa_pairs)} generated, {remaining} chunks left)")
                progress.advance(task)

            offset += limit
    
    return all_qa_pairs


def _process_chunks_parallel(
    table, total_chunks, chunk_ids, processed_chunk_ids,
    skip_patterns, min_length, llm, qa_prompt_template, n_pairs_per_chunk,
    max_retries, backoff_base, save_freq, output_file, all_qa_pairs, workers
):
    """Parallel processing with ThreadPoolExecutor."""
    # Load all chunks at once for parallel processing
    if chunk_ids:
        df = table.to_pandas()
        df = df[df["id"].isin(chunk_ids)]
    else:
        df = table.to_pandas()
    
    # Filter chunks
    chunks_to_process = []
    for idx, row in df.iterrows():
        chunk_id = row.get("id", f"chunk_{idx}")
        content = row.get("content", "")
        source_file = row.get("source_file", "unknown")
        
        # Apply same filters as sequential
        if chunk_id in processed_chunk_ids:
            continue
        if any(pattern in source_file for pattern in skip_patterns):
            continue
        
        metadata_patterns = ['readme', 'github.com', 'gitlab.com', 'bitbucket.org',
                           'license', 'contributing', 'changelog', 'authors']
        content_preview = content[:300].lower()
        if any(pattern in source_file.lower() or pattern in content_preview
               for pattern in metadata_patterns):
            repo_keywords = ['repository', 'clone', 'branch', 'commit', 'pull request', 'fork']
            if any(kw in content_preview for kw in repo_keywords):
                continue
        
        if not content or len(content.strip()) < min_length:
            continue
        
        chunks_to_process.append({
            'content': content,
            'chunk_id': chunk_id,
            'source_file': source_file
        })
    
    console.print(f"[cyan]Processing {len(chunks_to_process)} chunks with {workers} workers...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating QA pairs...", total=len(chunks_to_process))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for chunk in chunks_to_process:
                future = executor.submit(
                    _generate_pairs_for_chunk,
                    content=chunk['content'],
                    chunk_id=chunk['chunk_id'],
                    source_file=chunk['source_file'],
                    llm=llm,
                    prompt_template=qa_prompt_template,
                    n_pairs=n_pairs_per_chunk,
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                )
                futures[future] = chunk['chunk_id']
            
            completed_chunks = 0
            batch_count = 0
            
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    pairs = future.result()
                    all_qa_pairs.extend(pairs)
                    completed_chunks += 1
                    batch_count += 1
                    
                    # Save periodically - check if we crossed a save boundary
                    current_count = len(all_qa_pairs)
                    previous_count = current_count - len(pairs)
                    if (current_count // save_freq) > (previous_count // save_freq):
                        _save_intermediate(all_qa_pairs, output_file)
                except Exception as e:
                    error_msg = str(e)[:200] if len(str(e)) > 200 else str(e)
                    console.print(f"[red]✗ FAILED: {chunk_id} - {type(e).__name__}: {error_msg}[/red]")
                    completed_chunks += 1
                    batch_count += 1
                
                # Update progress in batches of workers count (show parallel completion)
                if batch_count >= workers:
                    remaining_chunks = len(chunks_to_process) - completed_chunks
                    progress.update(task, description=f"[cyan]Generating QA pairs... ({len(all_qa_pairs)} pairs | {completed_chunks}/{len(chunks_to_process)} chunks | {remaining_chunks} left)")
                    progress.advance(task, advance=batch_count)
                    batch_count = 0
            
            # Final batch update for remaining chunks
            if batch_count > 0:
                remaining_chunks = len(chunks_to_process) - completed_chunks
                progress.update(task, description=f"[cyan]Generating QA pairs... ({len(all_qa_pairs)} pairs | {completed_chunks}/{len(chunks_to_process)} chunks | {remaining_chunks} left)")
                progress.advance(task, advance=batch_count)
    
    return all_qa_pairs
