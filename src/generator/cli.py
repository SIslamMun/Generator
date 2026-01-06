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
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def generate(lancedb_path, output, config, table, n_pairs, target_pairs, batch_size, max_chunks, provider, model):
    """Generate QA pairs from LanceDB chunks."""
    console.print("\n[bold]🚀 Starting QA Generation[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts from individual files
    prompts = load_prompts(Path(config_path).parent)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)

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

    console.print(f"[bold green]✨ Success! Generated {len(qa_pairs)} QA pairs[/bold green]\n")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output JSON file path")
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--threshold", type=float, default=7.0, help="Minimum rating (1-10)")
@click.option("--batch-size", type=int, default=5, help="Pairs rated per LLM call")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def curate(input_file, output, config, threshold, batch_size, provider, model):
    """
    Filter QA pairs or CoT examples by quality using LLM-as-Judge.
    
    Automatically detects input format (QA or CoT), converts to conversation
    format for rating, and restores to original format after curation.
    """
    console.print("\n[bold]🎯 Starting QA Curation[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts from individual files
    prompts = load_prompts(Path(config_path).parent)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)

    # Curate QA pairs
    metrics = curate_qa_pairs(
        input_path=input_file,
        output_path=output,
        prompts=prompts,
        llm_config=llm_config,
        threshold=threshold,
        batch_size=batch_size,
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
    console.print("\n[bold]✨ Enriching QA pairs with better formatting[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract LLM config
    llm_config = _extract_llm_config(cfg, provider, model)

    # Load prompts
    prompts_dir = Path(config_path).parent

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
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def generate_cot(lancedb_path, output, config, table, n_pairs, target_pairs, batch_size, max_chunks, provider, model):
    """
    Generate CoT (Chain-of-Thought) pairs from LanceDB chunks.
    
    Creates QA pairs with step-by-step reasoning from documents.
    Based on "Distilling Step-by-Step" methodology (Google, 2023).
    
    Example:
        generator generate-cot lancedb/ -o cot_pairs.json --target-pairs 100
    """
    console.print("\n[bold]🧠 Starting CoT Generation[/bold]\n")

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
    console.print("\n[bold]🧠 Enhancing QA pairs with CoT reasoning[/bold]\n")

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
        console.print("\n[bold cyan]Step 2/4: Enriching with response rewriting...[/bold cyan]\n")
        qa_enriched = temp_dir / "qa_enriched.json"

        qa_pairs = load_qa_pairs(qa_raw)
        enriched_pairs = enrich_qa_pairs(
            qa_pairs=qa_pairs,
            llm_config=cfg["llm"],
            prompts_dir=prompts_dir,
            batch_size=5,
            preserve_original=True,
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


if __name__ == "__main__":
    main()
