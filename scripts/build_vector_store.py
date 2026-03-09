#!/usr/bin/env python3
"""CLI script to ingest documents and build the RAG vector store.

Loads PDFs, spreadsheets (xlsx/xls/csv), and optionally extracts and
describes plots/images from PDFs using GPT-4o vision.  Chunks, embeds,
and stores everything in ChromaDB ready for retrieval.

Usage:
    python scripts/build_vector_store.py [OPTIONS]
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
@click.option("--docs-dir", default="data/documents", help="Documents directory")
@click.option("--store-path", default="./chroma_db", help="ChromaDB storage path")
@click.option("--chunk-size", default=1000, help="Text chunk size (characters)")
@click.option("--chunk-overlap", default=200, help="Chunk overlap (characters)")
@click.option(
    "--embedding-provider",
    default="sentence-transformers",
    type=click.Choice(["sentence-transformers", "openai"]),
    help="Embedding model provider",
)
@click.option("--extract-images/--no-extract-images", default=True, help="Extract and describe PDF images with GPT-4o vision")
@click.option("--reset", is_flag=True, help="Reset the vector store before ingesting")
def main(
    docs_dir: str,
    store_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_provider: str,
    extract_images: bool,
    reset: bool,
) -> None:
    """Build the RAG vector store from PDFs, spreadsheets, and images."""
    from rich.console import Console

    from src.rag.document_loader import SUPPORTED_EXTENSIONS

    console = Console()
    docs_path = Path(docs_dir)

    console.print("[bold green]Building RAG Vector Store[/bold green]")
    console.print(f"  Documents: {docs_path.resolve()}")
    console.print(f"  Store: {store_path}")
    console.print(f"  Embeddings: {embedding_provider}")
    console.print(f"  Extract images: {extract_images}\n")

    # Discover supported files
    all_files = [
        f
        for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
    spreadsheet_files = [f for f in all_files if f.suffix.lower() in {".xlsx", ".xls", ".csv"}]

    if not all_files:
        console.print(f"[yellow]No supported files found in {docs_path}[/yellow]")
        console.print("Add PDFs, xlsx, xls, or csv files to data/documents/ and re-run.")
        console.print("\nRecommended documents (see README for full list):")
        console.print("  - EIA Short-Term Energy Outlook (free PDF)")
        console.print("  - EIA Annual Energy Outlook spreadsheets (xlsx)")
        console.print("  - OPEC Monthly Oil Market Report (free PDF)")
        console.print("  - IEA Oil Market Report (free PDF)")
        return

    console.print(f"  Found {len(pdf_files)} PDFs, {len(spreadsheet_files)} spreadsheets")

    from src.rag.chunker import TextChunker
    from src.rag.document_loader import DocumentLoader
    from src.rag.embeddings import EmbeddingGenerator
    from src.rag.vector_store import VectorStoreManager

    # 1. Load documents
    console.print("\n[cyan]Step 1/4: Loading documents...[/cyan]")
    loader = DocumentLoader(documents_dir=docs_path, extract_images=extract_images)
    pages = loader.load_directory()

    text_pages = [p for p in pages if p.get("content_type") != "image_description"]
    image_pages = [p for p in pages if p.get("content_type") == "image_description"]
    spreadsheet_pages = [p for p in pages if p.get("content_type") == "spreadsheet"]
    console.print(
        f"  Loaded {len(text_pages)} text pages, "
        f"{len(spreadsheet_pages)} spreadsheet sheets, "
        f"{len(image_pages)} image descriptions"
    )

    # 2. Chunk text
    console.print("[cyan]Step 2/4: Chunking text...[/cyan]")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_pages(pages)
    console.print(f"  Created {len(chunks)} text chunks")

    # 3. Generate embeddings
    console.print("[cyan]Step 3/4: Generating embeddings...[/cyan]")
    embedder = EmbeddingGenerator(provider=embedding_provider)
    chunks = embedder.embed_chunks(chunks)
    console.print(f"  Generated embeddings (dim={embedder.embedding_dim})")

    # 4. Build vector store
    console.print("[cyan]Step 4/4: Adding to vector store...[/cyan]")
    store = VectorStoreManager(store_path=store_path)
    if reset:
        console.print("  [yellow]Resetting existing vector store...[/yellow]")
    store.add_chunks(chunks)
    console.print(f"  Vector store contains {store.count()} documents")

    console.print("\n[bold green]Vector store built successfully![/bold green]")
    console.print(f"Store location: {Path(store_path).resolve()}")


if __name__ == "__main__":
    main()
