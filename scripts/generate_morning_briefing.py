#!/usr/bin/env python3
"""Generate the agentic power & gas morning trading briefing.

Orchestrates a multi-step agentic pipeline:
  1. Collect market data (prices, NG storage)
  2. Gather weather outlook (Open-Meteo)
  3. Run trained demand forecast model
  4. Query RAG knowledge base (energy trading docs)
  5. Analyse news sentiment
  6. Synthesise everything via LLM into a structured briefing

Usage:
    python scripts/generate_morning_briefing.py [OPTIONS]

Examples:
    # Full briefing with RAG and demand model
    python scripts/generate_morning_briefing.py

    # Save to file instead of printing
    python scripts/generate_morning_briefing.py --output reports/briefing_today.md

    # Skip demand model (if not yet trained)
    python scripts/generate_morning_briefing.py --no-demand-model
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--output", default=None, help="Save briefing to file (markdown)")
@click.option("--no-demand-model", is_flag=True, help="Skip demand forecast model")
@click.option("--no-rag", is_flag=True, help="Skip RAG knowledge base queries")
@click.option("--model-path", default="models/xgb_load.pkl",
              help="Path to trained load forecast model")
@click.option("--vector-store-dir", default="chroma_db",
              help="Path to ChromaDB vector store")
def main(
    output: str | None,
    no_demand_model: bool,
    no_rag: bool,
    model_path: str,
    vector_store_dir: str,
) -> None:
    """Generate the agentic morning trading briefing."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print("[bold green]Power & Gas Trading — Agentic Morning Briefing[/bold green]\n")

    # ── Initialise LLM ──────────────────────────────────────────────
    console.print("  [dim]Initialising LLM client...[/dim]")
    from src.rag.llm_client import LLMClient
    llm = LLMClient(provider="openai", model="gpt-4o")

    # ── Initialise RAG pipeline (if available) ───────────────────────
    qa_chain = None
    retriever = None
    if not no_rag:
        console.print("  [dim]Loading RAG knowledge base...[/dim]")
        try:
            from src.rag.embeddings import EmbeddingGenerator
            from src.rag.vector_store import VectorStoreManager
            from src.rag.retriever import Retriever
            from src.rag.qa_chain import QAChain

            embedder = EmbeddingGenerator()
            vs = VectorStoreManager(
                store_type="chroma",
                store_path=vector_store_dir,
            )
            # Check if vector store has documents
            try:
                vs._init_chroma()
                n_docs = vs._collection.count() if vs._collection else 0
            except Exception:
                n_docs = 0
            if n_docs > 0:
                retriever = Retriever(embedder=embedder, vector_store=vs)
                qa_chain = QAChain(retriever=retriever, llm=llm)
                console.print(f"  [green]OK[/green] RAG loaded: {n_docs:,} documents in knowledge base")
            else:
                console.print("  [yellow]WARN Vector store is empty - skipping RAG[/yellow]")
        except Exception as exc:
            console.print(f"  [yellow]WARN RAG init failed: {exc}[/yellow]")

    # ── Initialise price fetcher ─────────────────────────────────────
    prices_fetcher = None
    try:
        from src.data.price_fetcher import PriceFetcher
        prices_fetcher = PriceFetcher()
        console.print("  [green]OK[/green] Price fetcher ready")
    except Exception:
        console.print("  [yellow]WARN Price fetcher unavailable[/yellow]")

    # ── Load model path ──────────────────────────────────────────────
    load_path = None
    if not no_demand_model:
        mp = Path(model_path)
        if mp.exists():
            load_path = mp
            console.print(f"  [green]OK[/green] Demand model: {mp}")
        else:
            console.print(f"  [yellow]WARN Model not found: {mp} - skipping forecast[/yellow]")

    # ── Run the agent ────────────────────────────────────────────────
    console.print("\n[bold cyan]Running agentic briefing pipeline...[/bold cyan]\n")

    from src.agents.morning_briefing import MorningBriefingAgent

    agent = MorningBriefingAgent(
        llm_client=llm,
        qa_chain=qa_chain,
        retriever=retriever,
        load_model_path=load_path,
        prices_fetcher=prices_fetcher,
    )

    briefing = agent.run(date=datetime.now(timezone.utc))

    # ── Output ───────────────────────────────────────────────────────
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(briefing, encoding="utf-8")
        console.print(f"\n[green]OK - Briefing saved to {out_path}[/green]")
    else:
        console.print()
        console.print(Markdown(briefing))

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
