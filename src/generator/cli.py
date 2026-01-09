"""
Command-line interface for Generator.

Commands:
- generate: Generate QA pairs from LanceDB
- curate: Filter QA pairs by quality (LLM-as-Judge)
- export: Convert to training formats
"""

import click
import yaml
from pathlib import Path
from rich.console import Console
from .qa_generator import generate_qa_from_lancedb
from .curate import curate_qa_pairs
from .formatters import export_to_format
from .prompt_loader import load_prompts
from .enrich import enrich_qa_pairs, load_qa_pairs, save_qa_pairs
from .cot_generator import generate_cot_pairs
from .cot_enhancer import enhance_with_cot
from .compare import compare_datasets
from .clients import get_client

console = Console()


def _extract_llm_config(cfg, provider=None, model=None):
    """
    Extract and prepare LLM config from full config.
    
    Args:
        cfg: Full config dict
        provider: Optional provider override
        model: Optional model override
    
    Returns:
        LLM config dict ready for client initialization
    """
    llm_config = cfg["llm"].copy()
    
    # Get the active provider
    active_provider = provider if provider else llm_config.get("provider", "ollama")
    
    # Normalize provider names (backward compatibility)
    provider_map = {
        "adk": "gemini",
        "claude-sdk": "claude",
        "claude_sdk": "claude",
    }
    active_provider = provider_map.get(active_provider, active_provider)
    
    # Extract provider-specific config and merge
    provider_config = llm_config.get(active_provider, {})
    llm_config = {**provider_config, "provider": active_provider}
    
    # Override model if provided
    if model:
        llm_config["model"] = model
    
    return llm_config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Synthetic QA pair generator for LLM fine-tuning."""
    pass


@main.command()
def list_providers():
    """List all available LLM providers and their status."""
    console.print("\n[bold]📋 Available LLM Providers[/bold]\n")

    providers = [
        {
            "name": "ollama",
            "description": "Local Ollama (FREE, unlimited)",
            "status": "✅ RECOMMENDED",
            "setup": "ollama pull mistral:latest",
            "requires": "Ollama server running",
        },
        {
            "name": "claude-sdk",
            "description": "Claude Code Agent SDK (FREE)",
            "status": "✅ RECOMMENDED",
            "setup": "pip install claude-agent-sdk && claude auth login",
            "requires": "Claude Code authentication",
        },
        {
            "name": "adk",
            "description": "Google ADK with Gemini (FREE tier)",
            "status": "✅ RECOMMENDED",
            "setup": "pip install google-adk && export GOOGLE_API_KEY=...",
            "requires": "Google API key (free at aistudio.google.com)",
        },
        {
            "name": "vllm",
            "description": "Local vLLM server (FREE)",
            "status": "⚙️  Advanced",
            "setup": "vllm serve <model>",
            "requires": "vLLM server running",
        },
        {
            "name": "openai",
            "description": "OpenAI API (PAID)",
            "status": "💰 Paid",
            "setup": "export OPENAI_API_KEY=...",
            "requires": "OpenAI API key",
        },
        {
            "name": "anthropic",
            "description": "Anthropic API (PAID)",
            "status": "💰 Paid",
            "setup": "export ANTHROPIC_API_KEY=...",
            "requires": "Anthropic API key",
        },
    ]

    for p in providers:
        console.print(f"[bold cyan]{p['name']}[/bold cyan] - {p['description']}")
        console.print(f"  Status: {p['status']}")
        console.print(f"  Setup: [dim]{p['setup']}[/dim]")
        console.print(f"  Requires: {p['requires']}\n")

    console.print("[bold]Usage Examples:[/bold]")
    console.print("  # Use Ollama (default)")
    console.print("  uv run generator generate lancedb/ -o output.json --provider ollama")
    console.print("\n  # Use Claude")
    console.print("  uv run generator generate lancedb/ -o output.json --provider claude")
    console.print("\n  # Use Google Gemini")
    console.print("  uv run generator generate lancedb/ -o output.json --provider gemini --model gemini-2.0-flash-exp\n")


@main.command()
@click.argument("lancedb_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--table", default="text_chunks", multiple=True, help="LanceDB table name(s) - can specify multiple times")
@click.option("--n-pairs", type=int, help="QA pairs per chunk (fixed)")
@click.option("--target-pairs", type=int, help="Total target pairs (calculates per-chunk)")
@click.option("--batch-size", type=int, default=50, help="Chunks per batch")
@click.option("--max-chunks", type=int, help="Max chunks to process (for testing)")
@click.option("--topic", help="Topic filter (e.g., 'HDF5') - removes off-topic pairs after generation")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def generate(lancedb_path, output, config, table, n_pairs, target_pairs, batch_size, max_chunks, topic, provider, model):
    """Generate QA pairs from LanceDB chunks."""
    console.print("\n[bold]🚀 Generating QA pairs from LanceDB[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts from individual files
    prompts = load_prompts(Path(config_path).parent)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Add filtering configuration to llm_config
    filtering_config = cfg.get("filtering", {})
    if filtering_config.get("enabled", True):
        llm_config["skip_source_patterns"] = filtering_config.get("skip_source_patterns", [])
        llm_config["min_chunk_length"] = filtering_config.get("min_chunk_length", 200)
    
    # Add rate limiting config
    if "rate_limit" in cfg:
        llm_config["rate_limit"] = cfg["rate_limit"]

    # Handle multiple tables
    tables = table if table else ("text_chunks",)
    all_qa_pairs = []
    
    for table_name in tables:
        console.print(f"\n[bold cyan]📊 Processing table: {table_name}[/bold cyan]")
        
        # Generate QA pairs from this table
        qa_pairs = generate_qa_from_lancedb(
            db_path=lancedb_path,
            output_path=output,
            prompts=prompts,
            llm_config=llm_config.copy(),  # Copy to avoid mutation
            table_name=table_name,
            n_pairs_per_chunk=n_pairs,
            target_pairs=target_pairs // len(tables) if target_pairs else None,  # Split target across tables
            batch_size=batch_size,
            max_chunks=max_chunks,
        )
        
        all_qa_pairs.extend(qa_pairs)
    
    console.print(f"\n[bold green]✓ Total pairs from all tables: {len(all_qa_pairs)}[/bold green]\n")

    # Apply topic filtering if requested
    if topic:
        console.print(f"\n[bold cyan]🔍 Filtering pairs by topic: {topic}[/bold cyan]\n")
        from .curate import _rate_batch
        
        # Get rating prompt for topic filtering
        rating_prompt = prompts.get("qa_rating")
        if not rating_prompt:
            console.print("[yellow]⚠️  Warning: qa_rating prompt not found, skipping topic filter[/yellow]\n")
        else:
            # Initialize LLM for filtering (extract provider without modifying llm_config)
            provider = llm_config.get('provider', 'gemini')
            filter_llm = get_client(provider, llm_config)
            
            # Rate pairs with topic filter
            original_count = len(all_qa_pairs)
            filtered_pairs = []
            batch_size_filter = 10
            
            for i in range(0, len(all_qa_pairs), batch_size_filter):
                batch = all_qa_pairs[i:i + batch_size_filter]
                rated_batch = _rate_batch(
                    pairs=batch,
                    llm=filter_llm,
                    prompt_template=rating_prompt,
                    temperature=0.1,
                    topic_filter=topic
                )
                
                # Keep only topic-relevant pairs
                for pair in rated_batch:
                    if pair.get('topic_relevant', True):
                        filtered_pairs.append(pair)
            
            all_qa_pairs = filtered_pairs
            removed = original_count - len(all_qa_pairs)
            console.print(f"[green]✓ Filtered: kept {len(all_qa_pairs)}/{original_count} pairs (removed {removed} off-topic)[/green]\n")
            
            # Save filtered pairs
            import json
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✨ Success! Generated {len(all_qa_pairs)} QA pairs[/bold green]\n")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--threshold", type=float, default=7.0, help="Minimum rating (1-10)")
@click.option("--batch-size", type=int, default=5, help="Pairs rated per LLM call")
@click.option("--topic", help="Topic filter (e.g., 'HDF5') - removes off-topic pairs")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def curate(input_file, output, config, threshold, batch_size, topic, provider, model):
    """
    Filter QA pairs or CoT examples by quality using LLM-as-Judge.
    
    Automatically detects input format (QA or CoT), converts to conversation
    format for rating, and restores to original format after curation.
    """
    console.print("\n[bold]🎯 Curating QA pairs with LLM-as-Judge[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts from individual files
    prompts = load_prompts(Path(config_path).parent)

    # Extract LLM config - use curate.provider if specified, otherwise fallback to main provider
    curate_config = cfg.get("curate", {})
    curate_provider = curate_config.get("provider") or provider  # curate.provider > --provider flag > main llm.provider
    llm_config = _extract_llm_config(cfg, curate_provider, model)
    
    # Merge curate temperature into llm_config
    if "temperature" in curate_config:
        llm_config["temperature"] = curate_config["temperature"]
    
    # Get filtering patterns from config
    filtering_config = cfg.get("filtering", {})
    skip_patterns = filtering_config.get("skip_question_patterns", []) if filtering_config.get("enabled", True) else []

    # Curate QA pairs
    metrics = curate_qa_pairs(
        input_path=input_file,
        output_path=output,
        prompts=prompts,
        llm_config=llm_config,
        threshold=threshold,
        batch_size=batch_size,
        topic_filter=topic,
        skip_question_patterns=skip_patterns,
    )

    console.print(
        f"[bold green]✨ Success! Filtered {metrics['filtered_pairs']} / "
        f"{metrics['total_pairs']} pairs ({metrics['retention_rate']:.1%})[/bold green]\n"
    )


@main.command("select-coverage")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--target-count", type=int, help="Exact number of examples to select")
@click.option("--reduction-ratio", type=float, default=0.4, help="Target size as ratio (default: 0.4 = 40%)")
@click.option("--strategy", type=click.Choice(["centroid", "diverse"]), default="centroid", 
              help="Selection strategy: centroid (closest to cluster center) or diverse (spread)")
@click.option("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model for embeddings")
def select_coverage(input_file, output, target_count, reduction_ratio, strategy, model):
    """
    Select diverse, representative examples using semantic coverage.
    
    Based on TOUCAN (Oct 2025): Reduces dataset size by ~60% with no 
    performance degradation by selecting semantically diverse examples.
    
    Uses clustering to group similar examples and selects representatives
    from each cluster, ensuring broad coverage of topics/patterns.
    
    Requires: uv pip install -e ".[coverage]"
    
    Examples:
        # Keep top 40% most diverse (default)
        generator select-coverage curated.json -o diverse.json
        
        # Select exactly 500 diverse examples
        generator select-coverage curated.json -o diverse.json --target-count 500
        
        # Use diverse strategy (maximize spread)
        generator select-coverage curated.json -o diverse.json --strategy diverse
    """
    console.print("\n[bold]🎯 Selecting diverse examples by coverage[/bold]\n")
    
    try:
        from .coverage_selector import CoverageSelector
    except ImportError as e:
        console.print(
            "[red]Error: Coverage selection requires additional dependencies.[/red]\n"
            "[yellow]Install with: uv pip install -e \".[coverage]\"[/yellow]"
        )
        raise click.Abort()
    
    # Load input
    import json
    with open(input_file, "r") as f:
        examples = json.load(f)
    
    if not isinstance(examples, list):
        console.print("[red]Error: Input must be a JSON array of examples[/red]")
        raise click.Abort()
    
    console.print(f"[cyan]Loaded {len(examples)} examples from {input_file}[/cyan]")
    
    # Select by coverage
    selector = CoverageSelector(model_name=model)
    selected = selector.select_by_coverage(
        examples,
        target_count=target_count,
        reduction_ratio=reduction_ratio,
        strategy=strategy,
    )
    
    # Compute coverage score
    coverage_score = selector.compute_coverage_score(selected, examples)
    
    # Save output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    
    console.print(f"\n[green]✓ Saved {len(selected)} examples to {output}[/green]")
    console.print(f"[dim]Coverage score: {coverage_score:.3f} (1.0 = perfect)[/dim]")
    console.print(
        f"[bold green]✨ Success! Reduced {len(examples)} → {len(selected)} "
        f"({100*len(selected)/len(examples):.1f}%)[/bold green]\n"
    )


@main.command("multi-score")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--min-score", type=float, default=5.0, help="Minimum combined score (0-10)")
@click.option("--top-k", type=int, help="Select top-k pairs instead of threshold")
@click.option("--strategy", type=click.Choice(["combined", "complexity", "quality", "diversity"]),
              default="combined", help="Sorting/selection strategy")
@click.option("--complexity-weight", type=float, default=0.3, help="Complexity weight (default: 0.3)")
@click.option("--quality-weight", type=float, default=0.5, help="Quality weight (default: 0.5)")
@click.option("--diversity-weight", type=float, default=0.2, help="Diversity weight (default: 0.2)")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def multi_score(input_file, output, config, min_score, top_k, strategy, 
                complexity_weight, quality_weight, diversity_weight, provider, model):
    """
    Score QA pairs on multiple dimensions (DEITA-style).
    
    Based on DEITA (2024): 3D scoring achieves 10x data efficiency by
    selecting examples that are complex, high-quality, AND diverse.
    
    Dimensions:
      - Complexity: How challenging is the question? (reasoning depth)
      - Quality: How accurate/helpful is the answer?
      - Diversity: How different from other pairs?
    
    \b
    Examples:
      # Filter by combined score threshold
      uv run generator multi-score qa.json -o scored.json --min-score 6.0
      
      # Select top 100 by combined score
      uv run generator multi-score qa.json -o top100.json --top-k 100
      
      # Select top 50 most complex questions
      uv run generator multi-score qa.json -o complex.json --top-k 50 --strategy complexity
      
      # Custom weights (prefer complexity over diversity)
      uv run generator multi-score qa.json -o weighted.json --complexity-weight 0.5 --diversity-weight 0.1
    """
    console.print("\n[bold]📊 Multi-Dimensional Scoring (DEITA)[/bold]\n")
    
    from .multi_scorer import MultiDimensionalScorer, ScoreWeights
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Load input
    import json
    with open(input_file, "r") as f:
        pairs = json.load(f)
    
    if not isinstance(pairs, list):
        console.print("[red]Error: Input must be a JSON array of QA pairs[/red]")
        raise click.Abort()
    
    console.print(f"[cyan]Loaded {len(pairs)} pairs from {input_file}[/cyan]")
    console.print(f"[cyan]Weights: complexity={complexity_weight}, quality={quality_weight}, diversity={diversity_weight}[/cyan]")
    
    # Create scorer
    weights = ScoreWeights(
        complexity=complexity_weight,
        quality=quality_weight,
        diversity=diversity_weight,
    )
    
    scorer = MultiDimensionalScorer(
        llm_config=llm_config.copy(),
        weights=weights,
    )
    
    # Score and filter
    if top_k:
        selected = scorer.select_top_k(pairs, k=top_k, strategy=strategy)
    else:
        selected = scorer.filter_by_combined_score(pairs, min_score=min_score)
    
    # Save output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]✓ Saved {len(selected)} scored pairs to {output}[/green]")
    console.print(
        f"[bold green]✨ Success! Selected {len(selected)}/{len(pairs)} pairs[/bold green]\n"
    )


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
@click.option("--batch-size", type=int, default=5, help="Pairs to process per batch")
@click.option("--no-preserve-original", is_flag=True, help="Don't keep original answer")
def enrich(input_file, output, config, provider, model, batch_size, no_preserve_original):
    """
    Enrich QA pairs by rewriting answers for better clarity and structure.
    
    This implements response rewriting to improve answer quality while preserving
    all information. Useful for improving generated pairs before curation.
    
    Example:
        generator enrich output/qa_raw.json -o output/qa_enriched.json --config configs/config.yaml
    """
    console.print("\n[bold]✨ Enriching QA pairs with improved responses[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)

    # Load prompts
    prompts_dir = Path(config_path).parent

    # Get enrichment config
    enrich_config = cfg.get("enrich", {})

    # Load QA pairs
    qa_pairs = load_qa_pairs(Path(input_file))
    console.print(f"[cyan]Loaded {len(qa_pairs)} QA pairs[/cyan]")

    # Enrich pairs
    enriched_pairs = enrich_qa_pairs(
        qa_pairs=qa_pairs,
        llm_config=llm_config,
        prompts_dir=prompts_dir,
        batch_size=batch_size,
        preserve_original=not no_preserve_original,
        temperature=enrich_config.get("temperature", 0.3),
    )

    # Save enriched pairs
    save_qa_pairs(enriched_pairs, Path(output))

    console.print(f"[bold green]✨ Success! Enriched {len(enriched_pairs)} pairs[/bold green]\n")


@main.command()
@click.argument("lancedb_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--table", default="text_chunks", help="LanceDB table name")
@click.option("--n-pairs", type=int, help="CoT pairs per chunk (fixed)")
@click.option("--target-pairs", type=int, help="Total target pairs (calculates per-chunk)")
@click.option("--batch-size", type=int, default=50, help="Chunks per batch")
@click.option("--max-chunks", type=int, help="Max chunks to process (for testing)")
@click.option("--topic", help="Topic filter (e.g., 'HDF5') - removes off-topic pairs after generation")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def generate_cot(lancedb_path, output, config, table, n_pairs, target_pairs, batch_size, max_chunks, topic, provider, model):
    """
    Generate CoT (Chain-of-Thought) pairs from LanceDB chunks.
    
    Creates QA pairs with step-by-step reasoning from documents.
    Based on "Distilling Step-by-Step" methodology (Google, 2023).
    
    Example:
        generator generate-cot lancedb/ -o cot_pairs.json --target-pairs 100
    """
    console.print("\n[bold]🧠 Generating CoT reasoning pairs from LanceDB[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Add filtering configuration
    filtering_config = cfg.get("filtering", {})
    if filtering_config.get("enabled", True):
        llm_config["filtering"] = filtering_config

    # Generate CoT pairs
    result = generate_cot_pairs(
        lancedb_path=lancedb_path,
        output_path=output,
        llm_config=llm_config,
        table_name=table,
        n_pairs=n_pairs,
        target_pairs=target_pairs,
        batch_size=batch_size,
        max_chunks=max_chunks,
    )

    # Apply topic filtering if requested
    if topic:
        console.print(f"\n[bold cyan]🔍 Filtering CoT pairs by topic: {topic}[/bold cyan]\n")
        from .curate import _rate_batch, load_prompts
        import json
        
        # Load prompts
        config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        prompts = load_prompts(Path(config_path).parent)
        
        # Get rating prompt for topic filtering
        rating_prompt = prompts.get("qa_rating")
        if not rating_prompt:
            console.print("[yellow]⚠️  Warning: qa_rating prompt not found, skipping topic filter[/yellow]\n")
        else:
            # Load generated pairs
            with open(output, 'r', encoding='utf-8') as f:
                cot_pairs = json.load(f)
            
            # Initialize LLM for filtering (extract provider without modifying llm_config)
            provider = llm_config.get('provider', 'gemini')
            filter_llm = get_client(provider, llm_config)
            
            # Convert CoT to QA format for rating
            original_count = len(cot_pairs)
            filtered_pairs = []
            batch_size_filter = 10
            
            for i in range(0, len(cot_pairs), batch_size_filter):
                batch = cot_pairs[i:i + batch_size_filter]
                # Convert to QA format for rating
                qa_batch = [{'question': p['question'], 'answer': p['answer']} for p in batch]
                
                rated_batch = _rate_batch(
                    pairs=qa_batch,
                    llm=filter_llm,
                    prompt_template=rating_prompt,
                    temperature=0.1,
                    topic_filter=topic
                )
                
                # Keep only topic-relevant pairs (preserve original CoT structure)
                for j, pair in enumerate(rated_batch):
                    if pair.get('topic_relevant', True):
                        filtered_pairs.append(batch[j])
            
            removed = original_count - len(filtered_pairs)
            console.print(f"[green]✓ Filtered: kept {len(filtered_pairs)}/{original_count} pairs (removed {removed} off-topic)[/green]\n")
            
            # Save filtered pairs
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(filtered_pairs, f, indent=2, ensure_ascii=False)
            
            result['cot_pairs_generated'] = len(filtered_pairs)

    console.print(f"[bold green]✨ Success! Generated {result['cot_pairs_generated']} CoT pairs[/bold green]\n")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
@click.option("--batch-size", type=int, default=5, help="Pairs to enhance per batch")
def enhance_cot(input_file, output, config, provider, model, batch_size):
    """
    Add CoT reasoning to existing QA pairs.
    
    Takes plain QA pairs and enhances them with step-by-step reasoning,
    converting them to CoT format: {"question": "...", "reasoning": "...", "answer": "..."}.
    
    Example:
        generator enhance-cot qa_pairs.json -o cot_enhanced.json
    """
    console.print("\n[bold]🧠 Adding Chain-of-Thought reasoning to QA pairs[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)

    # Enhance with CoT
    result = enhance_with_cot(
        input_path=input_file,
        output_path=output,
        llm_config=llm_config,
        batch_size=batch_size,
    )

    console.print(f"[bold green]✨ Success! Enhanced {result['enhanced_pairs']} pairs with CoT reasoning[/bold green]\n")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output file path")
@click.option(
    "-f",
    "--format",
    type=click.Choice(["chatml", "alpaca", "sharegpt", "jsonl"]),
    default="chatml",
    help="Output format",
)
@click.option("--system-prompt", help="System prompt for conversation formats")
def export(input_file, output, format, system_prompt):
    """Export QA pairs to training format."""
    console.print(f"\n[bold]📦 Exporting to {format} format[/bold]\n")

    count = export_to_format(
        input_path=input_file, output_path=output, format_type=format, system_prompt=system_prompt
    )

    console.print(f"[bold green]✨ Success! Exported {count} examples[/bold green]\n")


@main.command()
@click.argument("lancedb_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Final training data file")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--threshold", type=float, default=7.0, help="Curation threshold")
@click.option(
    "-f",
    "--format",
    type=click.Choice(["chatml", "alpaca", "sharegpt", "jsonl"]),
    default="chatml",
    help="Output format",
)
@click.option("--max-chunks", type=int, help="Max chunks (for testing)")
@click.option("--skip-enrichment", is_flag=True, help="Skip enrichment step")
def pipeline(lancedb_path, output, config, threshold, format, max_chunks, skip_enrichment):
    """Run full pipeline: generate → enrich → curate → export."""
    console.print("\n[bold]🚀 Starting Full Pipeline[/bold]\n")

    output_path = Path(output)
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Step 1: Generate
    step_label = "1/4" if not skip_enrichment else "1/3"
    console.print(f"[bold cyan]Step {step_label}: Generating QA pairs...[/bold cyan]\n")
    qa_raw = temp_dir / "qa_raw.json"

    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts from individual files
    prompts_dir = Path(config_path).parent
    prompts = load_prompts(prompts_dir)

    generate_qa_from_lancedb(
        db_path=lancedb_path,
        output_path=str(qa_raw),
        prompts=prompts,
        llm_config=cfg["llm"],
        max_chunks=max_chunks,
    )

    # Step 2: Enrich (optional)
    if not skip_enrichment:
        step_label = "2/4" if not skip_enrichment else "2/3"
        console.print(f"\n[bold cyan]Step {step_label}: Enriching with response rewriting...[/bold cyan]\n")
        qa_enriched = temp_dir / "qa_enriched.json"

        qa_pairs = load_qa_pairs(qa_raw)
        enrich_config = cfg.get("enrich", {})
        enriched_pairs = enrich_qa_pairs(
            qa_pairs=qa_pairs,
            llm_config=cfg["llm"],
            prompts_dir=prompts_dir,
            batch_size=enrich_config.get("batch_size", 5),
            preserve_original=enrich_config.get("preserve_original", True),
            temperature=enrich_config.get("temperature", 0.3),
        )
        save_qa_pairs(enriched_pairs, qa_enriched)

        # Use enriched pairs for next step
        next_input = qa_enriched
        step_offset = 1
    else:
        console.print("\n[yellow]⏭️  Skipping enrichment step[/yellow]\n")
        next_input = qa_raw
        step_offset = 0

    # Step 3: Curate
    step_label = f"{2 + step_offset}/4" if not skip_enrichment else "2/3"
    console.print(f"\n[bold cyan]Step {step_label}: Curating with LLM-as-Judge...[/bold cyan]\n")
    qa_curated = temp_dir / "qa_curated.json"
    
    # Get filtering patterns from config
    filtering_config = cfg.get("filtering", {})
    skip_patterns = filtering_config.get("skip_question_patterns", []) if filtering_config.get("enabled", True) else []

    curate_qa_pairs(
        input_path=str(next_input),
        output_path=str(qa_curated),
        prompts=prompts,
        llm_config=cfg["llm"],
        threshold=threshold,
        skip_question_patterns=skip_patterns,
    )

    # Step 4: Export
    step_label = f"{3 + step_offset}/4" if not skip_enrichment else "3/3"
    console.print(f"\n[bold cyan]Step {step_label}: Exporting to {format}...[/bold cyan]\n")

    export_to_format(
        input_path=str(qa_curated), output_path=str(output_path), format_type=format
    )

    console.print("\n[bold green]✨ Pipeline Complete![/bold green]")
    console.print(f"[bold green]📁 Training data: {output_path}[/bold green]\n")


@main.command()
@click.argument("datasets", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output", type=click.Path(), required=True, help="Output comparison report path")
@click.option("--sample-size", type=int, default=10, help="Number of samples to judge per dataset")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.option("--provider", type=str, help="LLM provider override")
@click.option("--model", type=str, help="Model override")
def compare(datasets, output, sample_size, config, provider, model):
    """
    Compare multiple QA datasets using LLM judge.
    
    Can accept multiple files or use glob pattern:
    
    \b
    Compare two files:
      uv run generator compare file1.json file2.json -o report.json
    
    \b
    Compare all files in folder:
      uv run generator compare phase4_curate/*.json -o comparison.json
    """
    console.print("\n[bold cyan]🔍 Comparing datasets with LLM judge[/bold cyan]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Convert paths
    dataset_paths = [Path(d) for d in datasets]
    output_path = Path(output)
    
    console.print(f"📊 Datasets to compare: {len(dataset_paths)}")
    for p in dataset_paths:
        console.print(f"  - {p.name}")
    console.print(f"📝 Sample size per dataset: {sample_size}")
    console.print(f"🤖 Using: {llm_config['provider']} / {llm_config['model']}\n")
    
    # Run comparison
    report = compare_datasets(
        dataset_paths=dataset_paths,
        output_path=output_path,
        llm_config=llm_config,
        sample_size=sample_size
    )
    
    # Display winner
    winner = report.get("final_decision", {}).get("winner")
    if winner:
        console.print(f"\n[bold green]🏆 Recommended Winner: {winner}[/bold green]")
        console.print(f"[bold]📄 Full report: {output_path}[/bold]\n")
    else:
        console.print("\n[yellow]⚠️  Could not determine clear winner. Check report.[/yellow]\n")


# ==============================================================================
# PHASE 3: TOOL USE COMMANDS
# ==============================================================================

@main.command("tool-parse")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--format", "fmt", type=click.Choice(["auto", "openapi", "json", "python"]), 
              default="auto", help="Input format (auto-detect if not specified)")
def tool_parse(input_path, output, fmt):
    """
    Parse tool definitions from OpenAPI, JSON Schema, or Python modules.
    
    \b
    Examples:
      uv run generator tool-parse api.json -o tools.json
      uv run generator tool-parse openapi.yaml -o tools.json --format openapi
      uv run generator tool-parse tools.py -o tools.json --format python
    """
    from pathlib import Path
    from .tool_parser import ToolParser
    from .tool_schemas import save_tools
    
    console.print("\n[bold]🔧 Parsing tool definitions[/bold]\n")
    
    parser = ToolParser()
    input_file = Path(input_path)
    
    # Auto-detect format
    if fmt == "auto":
        if input_file.suffix in [".yaml", ".yml"]:
            fmt = "openapi"
        elif input_file.suffix == ".py":
            fmt = "python"
        else:
            fmt = "json"
    
    console.print(f"[cyan]Input: {input_path} (format: {fmt})[/cyan]")
    
    # Parse based on format
    if fmt == "openapi":
        tools = parser.parse_openapi(input_path)
    elif fmt == "python":
        tools = parser.parse_python_module(input_path)
    else:
        tools = parser.parse_json_schema(input_path)
    
    # Validate
    errors = parser.validate_tools(tools)
    if errors:
        console.print("[yellow]⚠️  Validation warnings:[/yellow]")
        for tool_id, errs in errors.items():
            for err in errs:
                console.print(f"  {tool_id}: {err}")
    
    # Save
    save_tools(tools, output)
    
    console.print(f"\n[bold green]✨ Parsed {len(tools)} tools → {output}[/bold green]\n")


@main.command("tool-deps")
@click.argument("tools_path", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output JSON file path for graph data")
@click.option("--tool", "tool_id", help="Show connections for specific tool")
@click.option("--chains", is_flag=True, help="List valid tool chains")
@click.option("--max-chain-length", type=int, default=4, help="Max chain length to search")
@click.option("--validate", "chain_to_validate", help="Validate a specific chain (comma-separated tool IDs)")
def tool_deps(tools_path, output, tool_id, chains, max_chain_length, chain_to_validate):
    """
    Analyze tool parameter dependencies.
    
    Based on In-N-Out (Sep 2025): Build parameter-level dependency graphs
    showing which tool outputs can feed into which tool inputs.
    
    Use this to:
    - Understand which tools can chain together
    - Validate proposed tool chains
    - Find compatible tool sequences
    
    \b
    Examples:
      # Show dependency summary
      uv run generator tool-deps configs/hdf5_tools.json
      
      # Show connections for specific tool
      uv run generator tool-deps configs/hdf5_tools.json --tool open_file
      
      # Export graph to JSON
      uv run generator tool-deps configs/hdf5_tools.json -o deps.json
      
      # List all valid chains
      uv run generator tool-deps configs/hdf5_tools.json --chains
      
      # Validate a chain
      uv run generator tool-deps configs/hdf5_tools.json --validate "open_file,read_full_dataset,close_file"
    """
    from .tool_schemas import load_tools
    from .dependency_graph import DependencyGraph
    
    console.print("\n[bold]🔗 Tool Dependency Analysis (In-N-Out)[/bold]\n")
    
    # Load tools
    tools = load_tools(tools_path)
    console.print(f"[green]✓ Loaded {len(tools)} tools[/green]\n")
    
    # Build graph
    graph = DependencyGraph(tools)
    
    # Handle specific operations
    if chain_to_validate:
        # Validate a specific chain
        chain_ids = [t.strip() for t in chain_to_validate.split(",")]
        is_valid, issues = graph.validate_chain(chain_ids)
        
        if is_valid:
            console.print(f"[green]✓ Chain is valid: {' → '.join(chain_ids)}[/green]")
        else:
            console.print(f"[red]✗ Chain is invalid:[/red]")
            for issue in issues:
                console.print(f"  - {issue}")
        return
    
    if tool_id:
        # Show connections for specific tool
        graph.print_tool_connections(tool_id)
        return
    
    if chains:
        # List valid chains
        console.print(f"[cyan]Finding valid chains (max length {max_chain_length})...[/cyan]\n")
        
        all_chains = []
        entry_points = graph.get_entry_points()
        
        for entry in entry_points[:5]:  # Limit to avoid explosion
            tool_chains = graph.get_compatible_chains(entry, max_length=max_chain_length)
            all_chains.extend(tool_chains[:10])  # Limit per entry
        
        # Show chains by length
        from collections import defaultdict
        by_length = defaultdict(list)
        for chain in all_chains:
            by_length[len(chain)].append(chain)
        
        for length in sorted(by_length.keys()):
            console.print(f"\n[bold]Chains of length {length}:[/bold]")
            for chain in by_length[length][:5]:  # Show max 5 per length
                chain_str = " → ".join(chain)
                console.print(f"  {chain_str}")
        
        console.print(f"\n[dim]Total: {len(all_chains)} valid chains found[/dim]")
        return
    
    # Default: show summary
    graph.print_summary()
    
    # Save if output specified
    if output:
        graph.save(output)


@main.command("tool-generate")
@click.argument("tools_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--single-step", "mode", flag_value="single", help="Generate only single-step examples")
@click.option("--multi-step", "mode", flag_value="multi", help="Generate only multi-step examples")
@click.option("--target-pairs", type=int, default=100, help="Total examples to generate")
@click.option("--max-steps", type=int, default=5, help="Max steps for multi-step")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def tool_generate(tools_path, output, config, mode, target_pairs, max_steps, provider, model):
    """
    Generate tool-use training examples.
    
    By default generates a balanced mix of single and multi-step examples
    based on instruction complexity (auto mode).
    
    \b
    Examples:
      uv run generator tool-generate tools.json -o examples.json
      uv run generator tool-generate tools.json -o examples.json --single-step
      uv run generator tool-generate tools.json -o examples.json --multi-step
      uv run generator tool-generate tools.json -o examples.json --target-pairs 500
    """
    import json
    from pathlib import Path
    from .tool_schemas import load_tools, save_examples
    from .tool_generator import ToolGenerator
    from .prompt_loader import load_prompts
    
    # Default to auto mode
    if mode is None:
        mode = "auto"
    
    console.print(f"\n[bold]🔧 Generating tool-use examples[/bold]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts (including tool_prompts.yaml)
    prompts = load_prompts(Path(config_path).parent)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    console.print(f"[cyan]Tools: {tools_path}[/cyan]")
    console.print(f"[cyan]Provider: {llm_config.get('provider', 'ollama')}[/cyan]")
    console.print(f"[cyan]Mode: {mode}[/cyan]")
    console.print(f"[cyan]Target: {target_pairs} examples[/cyan]\n")
    
    # Load tools
    tools = load_tools(tools_path)
    console.print(f"[green]✓ Loaded {len(tools)} tools[/green]")
    
    # Calculate per-tool count
    n_per_tool = max(1, target_pairs // len(tools))
    
    # Generate
    generator = ToolGenerator(llm_config.copy(), prompts)
    examples = generator.generate_examples(
        tools=tools,
        n_per_tool=n_per_tool,
        mode=mode,
        max_steps=max_steps,
    )
    
    # Save
    save_examples(examples, output)
    
    console.print(f"\n[bold green]✨ Generated {len(examples)} examples → {output}[/bold green]\n")


@main.command("tool-generate-chain")
@click.argument("tools_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--target-pairs", type=int, default=50, help="Total examples to generate")
@click.option("--min-steps", type=int, default=2, help="Min tools per chain")
@click.option("--max-steps", type=int, default=4, help="Max tools per chain")
@click.option("--hybrid/--no-hybrid", default=False, help="Use hybrid generation (chain-first + query-first)")
@click.option("--chain-ratio", type=float, default=0.4, help="Chain-first ratio for hybrid mode")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def tool_generate_chain(tools_path, output, config, target_pairs, min_steps, max_steps, 
                        hybrid, chain_ratio, provider, model):
    """
    Generate tool-use examples using chain-first approach.
    
    Based on ToolGrad (Aug 2025): Build valid tool chains first, then synthesize
    natural user queries. Reduces invalid samples by ~40% vs query-first.
    
    Use --hybrid for best results (combines chain-first and query-first).
    
    \b
    Examples:
      # Pure chain-first (complex multi-tool examples)
      uv run generator tool-generate-chain tools.json -o examples.json
      
      # Hybrid mode (recommended)
      uv run generator tool-generate-chain tools.json -o examples.json --hybrid
      
      # Customize chain length
      uv run generator tool-generate-chain tools.json -o examples.json --min-steps 3 --max-steps 5
    """
    import json
    from pathlib import Path
    from .tool_schemas import load_tools, save_examples
    from .tool_generator import ToolGenerator
    from .prompt_loader import load_prompts
    
    console.print(f"\n[bold]🔗 Chain-First Tool Generation (ToolGrad)[/bold]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts
    prompts = load_prompts(Path(config_path).parent)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    mode_str = "hybrid" if hybrid else "chain-first"
    console.print(f"[cyan]Tools: {tools_path}[/cyan]")
    console.print(f"[cyan]Provider: {llm_config.get('provider', 'ollama')}[/cyan]")
    console.print(f"[cyan]Mode: {mode_str}[/cyan]")
    console.print(f"[cyan]Target: {target_pairs} examples[/cyan]")
    console.print(f"[cyan]Chain steps: {min_steps}-{max_steps}[/cyan]\n")
    
    # Load tools
    tools = load_tools(tools_path)
    console.print(f"[green]✓ Loaded {len(tools)} tools[/green]")
    
    # Generate
    generator = ToolGenerator(llm_config.copy(), prompts)
    
    if hybrid:
        examples = generator.generate_examples_hybrid(
            tools=tools,
            n_total=target_pairs,
            chain_first_ratio=chain_ratio,
            max_steps=max_steps,
        )
    else:
        examples = generator.generate_chain_first(
            tools=tools,
            n_chains=target_pairs,
            min_steps=min_steps,
            max_steps=max_steps,
        )
    
    # Save
    save_examples(examples, output)
    
    console.print(f"\n[bold green]✨ Generated {len(examples)} examples → {output}[/bold green]\n")


@main.command("tool-execute")
@click.argument("examples_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--tools", "tools_path", type=click.Path(exists=True), help="Tools JSON file")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--mode", type=click.Choice(["simulated", "real", "multi_agent"]),
              default="simulated", help="Execution mode")
@click.option("--timeout", type=int, default=30, help="Timeout for real execution")
@click.option("--provider", help="LLM provider for simulation")
@click.option("--model", help="LLM model for simulation")
def tool_execute(examples_path, output, tools_path, config, mode, timeout, provider, model):
    """
    Execute and validate tool-use examples.
    
    \b
    Examples:
      uv run generator tool-execute examples.json -o validated.json --mode simulated
      uv run generator tool-execute examples.json -o validated.json --mode real --tools tools.json
    """
    from pathlib import Path
    from .tool_schemas import load_tools, load_examples, save_examples
    from .tool_executor import ToolExecutor, create_executor_with_builtins
    from .prompt_loader import load_prompts
    
    console.print(f"\n[bold]🔧 Executing tool examples ({mode})[/bold]\n")
    
    # Load config for LLM if needed
    llm_config = None
    prompts = {}
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    
    if mode in ["simulated", "multi_agent"]:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        llm_config = _extract_llm_config(cfg, provider, model)
        
        # Load prompts (includes tool_prompts.yaml)
        prompts = load_prompts(config_path.parent)
    
    console.print(f"[cyan]Examples: {examples_path}[/cyan]")
    console.print(f"[cyan]Mode: {mode}[/cyan]")
    
    # Load examples
    examples = load_examples(examples_path)
    console.print(f"[green]✓ Loaded {len(examples)} examples[/green]")
    
    # Load tools if provided
    tools = []
    if tools_path:
        tools = load_tools(tools_path)
        console.print(f"[green]✓ Loaded {len(tools)} tools[/green]")
    
    # Create executor
    if mode == "real":
        executor = create_executor_with_builtins(mode=mode, timeout=timeout)
    else:
        executor = ToolExecutor(mode=mode, timeout=timeout, llm_config=llm_config.copy() if llm_config else None, prompts=prompts)
    
    # Execute
    examples = executor.execute_examples(examples, tools)
    
    # Save
    save_examples(examples, output)
    
    console.print(f"\n[bold green]✨ Executed → {output}[/bold green]\n")


@main.command("tool-curate")
@click.argument("examples_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--tools", "tools_path", type=click.Path(exists=True), help="Tools JSON file")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--threshold", type=float, default=7.0, help="Minimum rating (1-10)")
@click.option("--min-success-rate", type=float, default=1.0, help="Min execution success rate")
@click.option("--balance-difficulty", is_flag=True, help="Balance by difficulty")
@click.option("--provider", help="LLM provider for rating")
@click.option("--model", help="LLM model for rating")
def tool_curate(examples_path, output, tools_path, config, threshold, min_success_rate, balance_difficulty, provider, model):
    """
    Curate tool-use examples by quality.
    
    \b
    Examples:
      uv run generator tool-curate examples.json -o curated.json --threshold 7.0
      uv run generator tool-curate examples.json -o curated.json --balance-difficulty
    """
    from pathlib import Path
    from .tool_schemas import load_tools, load_examples, save_examples
    from .tool_curator import ToolCurator
    from .prompt_loader import load_prompts
    
    console.print("\n[bold]🔧 Curating tool-use examples[/bold]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts (includes tool_prompts.yaml)
    prompts = load_prompts(Path(config_path).parent)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    console.print(f"[cyan]Examples: {examples_path}[/cyan]")
    console.print(f"[cyan]Threshold: {threshold}[/cyan]")
    
    # Load examples
    examples = load_examples(examples_path)
    console.print(f"[green]✓ Loaded {len(examples)} examples[/green]")
    
    # Load tools if provided
    tools = []
    if tools_path:
        tools = load_tools(tools_path)
    
    # Curate
    curator = ToolCurator(llm_config=llm_config.copy(), prompts=prompts)
    curated = curator.curate(
        examples=examples,
        tools=tools,
        min_success_rate=min_success_rate,
        rating_threshold=threshold,
        balance=balance_difficulty,
    )
    
    # Save
    save_examples(curated, output)
    
    console.print(f"\n[bold green]✨ Curated {len(curated)}/{len(examples)} examples → {output}[/bold green]\n")


@main.command("tool-evaluate")
@click.argument("examples_path", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output JSON file path (filtered examples)")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--min-score", type=float, default=0.7, help="Minimum outcome score (0.0-1.0)")
@click.option("--strict", is_flag=True, help="Require ALL requirements satisfied")
@click.option("--report-only", is_flag=True, help="Only report, don't filter")
@click.option("--provider", help="LLM provider")
@click.option("--model", help="LLM model")
def tool_evaluate(examples_path, output, config, min_score, strict, report_only, provider, model):
    """
    Evaluate tool-use examples by task completion (outcome-oriented).
    
    Goes beyond execution success to verify actual task completion.
    Based on MCP-AgentBench v2.
    
    \b
    Examples:
      uv run generator tool-evaluate examples.json -o verified.json --min-score 0.8
      uv run generator tool-evaluate examples.json --report-only
      uv run generator tool-evaluate examples.json -o strict.json --strict
    """
    from pathlib import Path
    from .tool_schemas import load_examples, save_examples
    from .outcome_evaluator import OutcomeEvaluator
    from .prompt_loader import load_prompts
    
    console.print("\n[bold]📊 Outcome-Oriented Evaluation[/bold]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts
    prompts = load_prompts(Path(config_path).parent)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    console.print(f"[cyan]Examples: {examples_path}[/cyan]")
    console.print(f"[cyan]Min Score: {min_score}[/cyan]")
    console.print(f"[cyan]Strict Mode: {strict}[/cyan]")
    
    # Load examples
    examples = load_examples(examples_path)
    console.print(f"[green]✓ Loaded {len(examples)} examples[/green]\n")
    
    # Evaluate
    evaluator = OutcomeEvaluator(
        llm_config=llm_config.copy(),
        prompts=prompts,
        strict_mode=strict,
    )
    
    if report_only:
        # Just evaluate and report
        results = evaluator.evaluate_examples(examples)
        console.print(f"\n[green]✓ Evaluated {len(results)} examples[/green]")
    else:
        # Filter by outcome
        if not output:
            console.print("[red]Error: -o/--output required unless --report-only is set[/red]")
            return
        
        filtered = evaluator.filter_by_outcome(
            examples,
            min_score=min_score,
            require_all_satisfied=strict,
        )
        
        # Save
        save_examples(filtered, output)
        
        console.print(f"\n[bold green]✨ Verified {len(filtered)}/{len(examples)} examples → {output}[/bold green]\n")


@main.command("tool-pipeline")
@click.argument("tools_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSONL file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--single-step", "mode", flag_value="single", help="Generate only single-step examples")
@click.option("--multi-step", "mode", flag_value="multi", help="Generate only multi-step examples")
@click.option("--target-pairs", type=int, default=100, help="Total examples to generate")
@click.option("--threshold", type=float, default=7.0, help="Quality threshold")
@click.option("--execution-mode", type=click.Choice(["simulated", "real"]),
              default="simulated", help="Execution mode")
@click.option("--format", "fmt", type=click.Choice(["chatml", "alpaca", "sharegpt"]),
              default="chatml", help="Output training format")
@click.option("--provider", help="LLM provider")
@click.option("--model", help="LLM model")
def tool_pipeline(tools_path, output, config, mode, target_pairs, threshold, execution_mode, fmt, provider, model):
    """
    Full tool-use training data pipeline.
    
    Runs: generate → execute → curate → export
    
    By default generates a balanced mix of single and multi-step examples.
    
    \b
    Examples:
      uv run generator tool-pipeline tools.json -o training.jsonl
      uv run generator tool-pipeline tools.json -o training.jsonl --single-step
      uv run generator tool-pipeline tools.json -o training.jsonl --target-pairs 1000
    """
    import json
    from pathlib import Path
    from .tool_schemas import load_tools, save_examples, ToolExample
    from .tool_generator import ToolGenerator
    from .tool_executor import ToolExecutor
    from .tool_curator import ToolCurator
    from .prompt_loader import load_prompts
    
    console.print("\n[bold]🔧 Running full tool-use pipeline[/bold]\n")
    
    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load prompts (includes tool_prompts.yaml)
    prompts = load_prompts(Path(config_path).parent)
    
    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Default to auto mode
    if mode is None:
        mode = "auto"
    
    console.print(f"[cyan]Tools: {tools_path}[/cyan]")
    console.print(f"[cyan]Mode: {mode}[/cyan]")
    console.print(f"[cyan]Target: {target_pairs} examples[/cyan]")
    console.print(f"[cyan]Execution: {execution_mode}[/cyan]")
    console.print(f"[cyan]Format: {fmt}[/cyan]\n")
    
    # Load tools
    tools = load_tools(tools_path)
    console.print(f"[green]✓ Loaded {len(tools)} tools[/green]\n")
    
    # Step 1: Generate
    console.print("[bold]Step 1/4: Generating examples...[/bold]")
    n_per_tool = max(1, target_pairs // len(tools))
    generator = ToolGenerator(llm_config.copy(), prompts)
    examples = generator.generate_examples(tools, n_per_tool, mode)
    
    # Step 2: Execute
    console.print("\n[bold]Step 2/4: Executing examples...[/bold]")
    executor = ToolExecutor(mode=execution_mode, llm_config=llm_config.copy(), prompts=prompts)
    examples = executor.execute_examples(examples, tools)
    
    # Step 3: Curate
    console.print("\n[bold]Step 3/4: Curating examples...[/bold]")
    curator = ToolCurator(llm_config=llm_config.copy(), prompts=prompts)
    examples = curator.curate(examples, tools, rating_threshold=threshold)
    
    # Step 4: Export
    console.print("\n[bold]Step 4/4: Exporting to training format...[/bold]")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for example in examples:
            training_data = example.to_training_format(fmt)
            f.write(json.dumps(training_data, ensure_ascii=False) + "\n")
    
    console.print(f"\n[bold green]✨ Pipeline complete! {len(examples)} examples → {output}[/bold green]\n")


if __name__ == "__main__":
    main()
