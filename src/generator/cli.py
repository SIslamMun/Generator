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
@click.option("--table", default="text_chunks", help="LanceDB table name")
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
    
    # Add rate limiting config
    if "rate_limit" in cfg:
        llm_config["rate_limit"] = cfg["rate_limit"]

    # Generate QA pairs
    qa_pairs = generate_qa_from_lancedb(
        db_path=lancedb_path,
        output_path=output,
        prompts=prompts,
        llm_config=llm_config,
        table_name=table,
        n_pairs_per_chunk=n_pairs,
        target_pairs=target_pairs,
        batch_size=batch_size,
        max_chunks=max_chunks,
    )

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
            original_count = len(qa_pairs)
            filtered_pairs = []
            batch_size_filter = 10
            
            for i in range(0, len(qa_pairs), batch_size_filter):
                batch = qa_pairs[i:i + batch_size_filter]
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
            
            qa_pairs = filtered_pairs
            removed = original_count - len(qa_pairs)
            console.print(f"[green]✓ Filtered: kept {len(qa_pairs)}/{original_count} pairs (removed {removed} off-topic)[/green]\n")
            
            # Save filtered pairs
            import json
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✨ Success! Generated {len(qa_pairs)} QA pairs[/bold green]\n")


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

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)
    
    # Merge curate temperature into llm_config
    curate_config = cfg.get("curate", {})
    if "temperature" in curate_config:
        llm_config["temperature"] = curate_config["temperature"]

    # Curate QA pairs
    metrics = curate_qa_pairs(
        input_path=input_file,
        output_path=output,
        prompts=prompts,
        llm_config=llm_config,
        threshold=threshold,
        batch_size=batch_size,
        topic_filter=topic,
    )

    console.print(
        f"[bold green]✨ Success! Filtered {metrics['filtered_pairs']} / "
        f"{metrics['total_pairs']} pairs ({metrics['retention_rate']:.1%})[/bold green]\n"
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

    curate_qa_pairs(
        input_path=str(next_input),
        output_path=str(qa_curated),
        prompts=prompts,
        llm_config=cfg["llm"],
        threshold=threshold,
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
