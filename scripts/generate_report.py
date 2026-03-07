#!/usr/bin/env python3
"""CLI script to generate the LLM morning research note.

Combines current price data, sentiment, and RAG context to produce
a professional energy market morning note.

Usage:
    python scripts/generate_report.py [OPTIONS]
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--output-dir", default="data/processed", help="Output directory for reports")
@click.option("--vector-store", default="./chroma_db", help="ChromaDB vector store path")
@click.option(
    "--llm-provider",
    default="openai",
    type=click.Choice(["openai", "ollama"]),
    help="LLM provider",
)
@click.option("--print-only", is_flag=True, help="Print to console instead of saving")
def main(
    output_dir: str,
    vector_store: str,
    llm_provider: str,
    print_only: bool,
) -> None:
    """Generate the daily energy market morning research note."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Generating Energy Market Morning Note[/bold green]\n")

    # Fetch current prices
    console.print("[cyan]Fetching current market data...[/cyan]")
    prices = {}
    price_changes = {}
    try:
        from src.data.price_fetcher import PriceFetcher

        fetcher = PriceFetcher()
        for instrument in ["wti", "natural_gas", "brent"]:
            df = fetcher.fetch_latest(instrument, days=5)
            if not df.empty:
                prices[instrument.upper()] = float(df["Close"].iloc[-1])
                price_changes[instrument.upper()] = float(df["Close"].pct_change().iloc[-1] * 100)
        console.print(f"  ✓ Retrieved prices for {len(prices)} instruments")
    except Exception as exc:
        console.print(f"  [yellow]⚠ Price fetch failed: {exc}[/yellow]")
        prices = {"WTI": 75.0, "BRENT": 78.0, "NATURAL_GAS": 2.5}
        price_changes = {"WTI": -0.5, "BRENT": -0.3, "NATURAL_GAS": 1.2}

    # Set up LLM
    from src.rag.llm_client import LLMClient

    llm = LLMClient(provider=llm_provider)

    # Set up RAG retriever (optional)
    retriever = None
    if Path(vector_store).exists():
        try:
            from src.rag.embeddings import EmbeddingGenerator
            from src.rag.retriever import Retriever
            from src.rag.vector_store import VectorStoreManager

            embedder = EmbeddingGenerator()
            store = VectorStoreManager(store_path=vector_store)
            retriever = Retriever(embedder=embedder, vector_store=store)
            console.print("  ✓ RAG vector store loaded")
        except Exception as exc:
            console.print(f"  [yellow]⚠ RAG unavailable: {exc}[/yellow]")
    else:
        console.print(
            f"  [yellow]⚠ No vector store at {vector_store} — generating without RAG[/yellow]"
        )

    # Generate note
    from src.reporting.morning_note import MorningNoteGenerator

    generator = MorningNoteGenerator(llm_client=llm, retriever=retriever)

    console.print("[cyan]Generating morning note...[/cyan]")
    try:
        note = generator.generate(
            prices=prices,
            price_changes=price_changes,
        )
    except Exception as exc:
        console.print(f"[red]Note generation failed: {exc}[/red]")
        return

    if print_only:
        console.print("\n")
        console.print(Markdown(note))
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_file = output_path / f"morning_note_{date_str}.md"
        with open(report_file, "w") as f:
            f.write(note)
        console.print(f"\n  ✓ Morning note saved to: {report_file}")
        console.print("\n")
        console.print(Markdown(note[:500] + "...\n*(truncated — see file for full note)*"))

    console.print("\n[bold green]Report generation complete![/bold green]")


if __name__ == "__main__":
    main()
