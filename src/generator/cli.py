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

console = Console()


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
@click.option("--n-pairs", type=int, default=5, help="QA pairs per chunk")
@click.option("--batch-size", type=int, default=50, help="Chunks per batch")
@click.option("--max-chunks", type=int, help="Max chunks to process (for testing)")
@click.option("--provider", help="LLM provider (override config)")
@click.option("--model", help="LLM model (override config)")
def generate(lancedb_path, output, config, table, n_pairs, batch_size, max_chunks, provider, model):
    """Generate QA pairs from LanceDB chunks."""
    console.print("\n[bold]🚀 Starting QA Generation[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts
    prompts_path = Path(config_path).parent / cfg.get("prompts_file", "prompts.yaml")
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)

    # Override LLM settings if provided
    llm_config = cfg["llm"]
    if provider:
        # Normalize provider names (backward compatibility)
        provider_map = {
            "adk": "gemini",          # Old name -> new name
            "claude-sdk": "claude",   # Old name -> new name  
            "claude_sdk": "claude",   # Old name -> new name
        }
        normalized_provider = provider_map.get(provider, provider)
        llm_config["provider"] = normalized_provider
        
        # Try both old and new names for config lookup
        provider_config = llm_config.get(normalized_provider) or llm_config.get(provider)
        if provider_config:
            llm_config.update(provider_config)
    if model:
        llm_config["model"] = model

    # Generate QA pairs
    qa_pairs = generate_qa_from_lancedb(
        db_path=lancedb_path,
        output_path=output,
        prompts=prompts,
        llm_config=llm_config,
        table_name=table,
        n_pairs_per_chunk=n_pairs,
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
    """Filter QA pairs by quality using LLM-as-Judge."""
    console.print("\n[bold]🎯 Starting QA Curation[/bold]\n")

    # Load config
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load prompts
    prompts_path = Path(config_path).parent / cfg.get("prompts_file", "prompts.yaml")
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)

    # Override LLM settings if provided
    llm_config = cfg["llm"]
    if provider:
        llm_config["provider"] = provider
    if model:
        llm_config["model"] = model

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
def pipeline(lancedb_path, output, config, threshold, format, max_chunks):
    """Run full pipeline: generate → curate → export."""
    console.print("\n[bold]🚀 Starting Full Pipeline[/bold]\n")

    output_path = Path(output)
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Step 1: Generate
    console.print("[bold cyan]Step 1/3: Generating QA pairs...[/bold cyan]\n")
    qa_raw = temp_dir / "qa_raw.json"
    
    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    prompts_path = Path(config_path).parent / cfg.get("prompts_file", "prompts.yaml")
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    
    generate_qa_from_lancedb(
        db_path=lancedb_path,
        output_path=str(qa_raw),
        prompts=prompts,
        llm_config=cfg["llm"],
        max_chunks=max_chunks,
    )

    # Step 2: Curate
    console.print("\n[bold cyan]Step 2/3: Curating with LLM-as-Judge...[/bold cyan]\n")
    qa_curated = temp_dir / "qa_curated.json"
    
    curate_qa_pairs(
        input_path=str(qa_raw),
        output_path=str(qa_curated),
        prompts=prompts,
        llm_config=cfg["llm"],
        threshold=threshold,
    )

    # Step 3: Export
    console.print(f"\n[bold cyan]Step 3/3: Exporting to {format}...[/bold cyan]\n")
    
    export_to_format(
        input_path=str(qa_curated), output_path=str(output_path), format_type=format
    )

    console.print(f"\n[bold green]✨ Pipeline Complete![/bold green]")
    console.print(f"[bold green]📁 Training data: {output_path}[/bold green]\n")


if __name__ == "__main__":
    main()
