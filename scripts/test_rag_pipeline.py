#!/usr/bin/env python3
"""End-to-end RAG pipeline smoke test.

Validates the full retrieve → generate flow:
  1. Load existing ChromaDB store
  2. Embed a query
  3. Retrieve relevant chunks
  4. Generate an LLM answer (requires OPENAI_API_KEY)
  5. Generate a trading signal

Usage:
    python scripts/test_rag_pipeline.py [--skip-llm]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--skip-llm", is_flag=True, help="Skip LLM generation (test retrieval only)")
@click.option("--query", default=None, help="Custom query to test with")
def main(skip_llm: bool, query: str | None) -> None:
    """Run end-to-end RAG pipeline smoke test."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel("[bold green]Energy Trading AI — RAG Pipeline Smoke Test[/bold green]"))
    passed = 0
    failed = 0

    # ── Step 1: Vector store ────────────────────────────────────────
    console.print("\n[bold cyan]1. Loading vector store...[/bold cyan]")
    try:
        from src.rag.vector_store import VectorStoreManager

        vs = VectorStoreManager(store_path="./chroma_db")
        doc_count = vs.count()
        console.print(f"   ✓ ChromaDB loaded — {doc_count:,} documents")
        assert doc_count > 0, "Vector store is empty"
        passed += 1
    except Exception as exc:
        console.print(f"   ✗ Failed: {exc}")
        failed += 1
        console.print("[red]Cannot continue without vector store.[/red]")
        sys.exit(1)

    # ── Step 2: Embedding generator ─────────────────────────────────
    console.print("\n[bold cyan]2. Loading embedding model...[/bold cyan]")
    try:
        from src.rag.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator(provider="sentence-transformers")
        test_vec = embedder.embed_query("test")
        console.print(f"   ✓ Embedder ready — dim={test_vec.shape[0]}")
        passed += 1
    except Exception as exc:
        console.print(f"   ✗ Failed: {exc}")
        failed += 1
        console.print("[red]Cannot continue without embedder.[/red]")
        sys.exit(1)

    # ── Step 3: Retriever ───────────────────────────────────────────
    test_query = query or "What factors drive crude oil price volatility?"
    console.print(f"\n[bold cyan]3. Retrieving chunks for: '{test_query}'[/bold cyan]")
    try:
        from src.rag.retriever import Retriever

        retriever = Retriever(embedder=embedder, vector_store=vs, top_k=10, rerank_top_k=5)
        chunks = retriever.retrieve(test_query)
        console.print(f"   ✓ Retrieved {len(chunks)} chunks")

        if chunks:
            tbl = Table(title="Top retrieved chunks", show_lines=True)
            tbl.add_column("#", width=3)
            tbl.add_column("Source", max_width=40)
            tbl.add_column("Text preview", max_width=80)
            for i, c in enumerate(chunks[:5], 1):
                source = c.get("source", c.get("filename", "?"))
                text = c.get("text", "")[:150].replace("\n", " ")
                tbl.add_row(str(i), str(source), text)
            console.print(tbl)

        context = retriever.format_context(chunks)
        console.print(f"   Context length: {len(context)} chars")
        assert len(chunks) > 0, "No chunks retrieved"
        passed += 1
    except Exception as exc:
        console.print(f"   ✗ Failed: {exc}")
        failed += 1
        import traceback

        traceback.print_exc()

    if skip_llm:
        console.print("\n[yellow]--skip-llm flag set — skipping LLM generation tests[/yellow]")
    else:
        # ── Step 4: QA Chain ────────────────────────────────────────
        console.print(f"\n[bold cyan]4. Generating answer via QAChain...[/bold cyan]")
        try:
            from src.rag.llm_client import LLMClient
            from src.rag.qa_chain import QAChain

            llm = LLMClient(provider="openai")
            qa = QAChain(retriever=retriever, llm=llm)
            result = qa.ask(test_query, top_k=5)

            console.print(Panel(result["answer"], title="LLM Answer", border_style="green"))
            console.print(f"   Sources returned: {len(result.get('sources', []))}")
            assert "answer" in result and len(result["answer"]) > 10
            passed += 1
        except Exception as exc:
            console.print(f"   ✗ Failed: {exc}")
            failed += 1
            import traceback

            traceback.print_exc()

        # ── Step 5: Signal Generator ────────────────────────────────
        console.print(f"\n[bold cyan]5. Generating trading signal...[/bold cyan]")
        try:
            from src.rag.signal_generator import LLMSignalGenerator

            sig_gen = LLMSignalGenerator(qa_chain=qa)
            signal = sig_gen.generate(
                instrument="WTI Crude Oil",
                market_context=(
                    "WTI is trading at $72.50, down 1.2% on the day. "
                    "OPEC+ is considering extending production cuts. "
                    "US crude inventories fell by 3.2 million barrels last week."
                ),
            )
            signal_tbl = Table(title="Trading Signal", show_lines=True)
            signal_tbl.add_column("Field", style="bold")
            signal_tbl.add_column("Value")
            signal_tbl.add_row("Direction", signal.direction)
            signal_tbl.add_row("Confidence", f"{signal.confidence:.2f}")
            signal_tbl.add_row("Time Horizon", signal.time_horizon)
            signal_tbl.add_row("Reasoning", signal.reasoning)
            signal_tbl.add_row("Key Risks", signal.key_risks)
            signal_tbl.add_row("Numeric Signal", str(signal.numeric_signal))
            console.print(signal_tbl)
            assert signal.direction in ("BULLISH", "BEARISH", "NEUTRAL")
            passed += 1
        except Exception as exc:
            console.print(f"   ✗ Failed: {exc}")
            failed += 1
            import traceback

            traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────────
    console.print()
    total = passed + failed
    color = "green" if failed == 0 else "red"
    console.print(
        Panel(
            f"[bold {color}]{passed}/{total} checks passed[/bold {color}]",
            title="RAG Smoke Test Results",
        )
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
