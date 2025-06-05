"""CLI interface for the RAG platform."""

import json
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table

from rag_platform.core.config import ExperimentConfig, Settings
from rag_platform.core.experiment import ExperimentRunner
from rag_platform.core.registry import registry

console = Console()


@click.group()
def cli():
    """Modular RAG Platform - Test and evaluate RAG pipelines."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--sample-size", type=int, help="Number of samples to evaluate")
@click.option("--output-dir", type=click.Path(), help="Override output directory")
def run(config_path: str, sample_size: Optional[int], output_dir: Optional[str]):
    """Run a RAG experiment from configuration file."""
    console.print(f"[bold green]Loading configuration from:[/bold green] {config_path}")
    
    config = ExperimentConfig.from_yaml(config_path)
    
    if sample_size:
        config.evaluation.sample_size = sample_size
    if output_dir:
        config.output.results_dir = output_dir
    
    console.print(f"[bold blue]Running experiment:[/bold blue] {config.name}")
    
    runner = ExperimentRunner(config)
    results = runner.run()
    
    console.print("[bold green]Experiment completed![/bold green]")
    console.print(f"Results saved to: {results['output_path']}")
    
    if "evaluation_results" in results:
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        
        for metric, score in results["evaluation_results"].items():
            if isinstance(score, (int, float)):
                table.add_row(metric, f"{score:.4f}")
            else:
                table.add_row(metric, str(score))
        
        console.print(table)


@cli.command()
def list_components():
    """List all available components."""
    components = registry.list_components()
    
    for component_type, names in components.items():
        console.print(f"\n[bold blue]{component_type.capitalize()}:[/bold blue]")
        for name in names:
            console.print(f"  - {name}")


@cli.command()
@click.argument("output_path", type=click.Path())
def create_config(output_path: str):
    """Create a sample configuration file."""
    sample_config = {
        "experiment": {
            "name": "sample_experiment",
            "description": "Sample RAG experiment configuration",
            "data": {
                "source": "local",
                "path": "./data/ru_rag_test_dataset.pkl",
                "loader": "pickle_loader"
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 1536
            },
            "vectorstore": {
                "type": "chroma",
                "collection_name": "sample_collection",
                "persist_directory": "./chroma_db"
            },
            "retriever": {
                "type": "vector",
                "k": 5,
                "rerank": False
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 500
            },
            "rag_chain": {
                "type": "vanilla",
                "prompt_template": "default"
            },
            "evaluation": {
                "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                "sample_size": 100
            },
            "output": {
                "results_dir": "./results",
                "format": "json"
            }
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[bold green]Sample configuration created at:[/bold green] {output_path}")


@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
def view_results(results_path: str):
    """View experiment results."""
    with open(results_path, "r") as f:
        if results_path.endswith(".json"):
            results = json.load(f)
        elif results_path.endswith(".yaml"):
            results = yaml.safe_load(f)
        else:
            console.print("[bold red]Unsupported file format. Use JSON or YAML.[/bold red]")
            return
    
    console.print(f"\n[bold blue]Experiment:[/bold blue] {results.get('experiment_name', 'Unknown')}")
    console.print(f"[bold blue]Timestamp:[/bold blue] {results.get('timestamp', 'Unknown')}")
    
    if "evaluation_results" in results:
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        
        for metric, score in results["evaluation_results"].items():
            if isinstance(score, (int, float)):
                table.add_row(metric, f"{score:.4f}")
            else:
                table.add_row(metric, str(score))
        
        console.print(table)
    
    if "sample_results" in results and len(results["sample_results"]) > 0:
        console.print("\n[bold blue]Sample Results:[/bold blue]")
        for i, sample in enumerate(results["sample_results"][:3]):
            console.print(f"\n[bold]Sample {i+1}:[/bold]")
            console.print(f"Q: {sample['query']}")
            console.print(f"A: {sample['answer']}")


@cli.command()
def check_env():
    """Check environment configuration."""
    settings = Settings()
    
    console.print("[bold blue]Environment Configuration:[/bold blue]\n")
    
    checks = [
        ("OpenAI API Key", bool(settings.openai_api_key)),
        ("Cohere API Key", bool(settings.cohere_api_key)),
        ("Anthropic API Key", bool(settings.anthropic_api_key)),
        ("HuggingFace API Key", bool(settings.huggingface_api_key)),
    ]
    
    table = Table(title="API Keys")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    
    for service, configured in checks:
        status = "✓ Configured" if configured else "✗ Not configured"
        table.add_row(service, status)
    
    console.print(table)
    
    console.print(f"\n[bold]Default Settings:[/bold]")
    console.print(f"  Embedding Model: {settings.default_embedding_model}")
    console.print(f"  LLM Model: {settings.default_llm_model}")
    console.print(f"  Temperature: {settings.default_temperature}")


if __name__ == "__main__":
    cli()