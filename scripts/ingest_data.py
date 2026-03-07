#!/usr/bin/env python3
"""CLI script to ingest all energy market data.

Downloads price data from yfinance, fundamental data from EIA/FRED,
weather data from NOAA, and news from NewsAPI/RSS feeds.
Saves all data to the data/raw/ directory.

Usage:
    python scripts/ingest_data.py [OPTIONS]

Options:
    --start TEXT    Start date (YYYY-MM-DD) [default: 2015-01-01]
    --end TEXT      End date (YYYY-MM-DD) [default: today]
    --output DIR    Output directory [default: data/raw]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--output", default="data/raw", help="Output directory")
@click.option("--skip-eia", is_flag=True, help="Skip EIA data download")
@click.option("--skip-fred", is_flag=True, help="Skip FRED data download")
@click.option("--skip-news", is_flag=True, help="Skip news download")
def main(
    start: str,
    end: str | None,
    output: str,
    skip_eia: bool,
    skip_fred: bool,
    skip_news: bool,
) -> None:
    """Download all energy market data to the raw data directory."""
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Energy Trading AI — Data Ingestion[/bold green]")
    console.print(f"Period: {start} → {end or 'today'}")
    console.print(f"Output: {output_dir.resolve()}\n")

    with Progress(console=console) as progress:
        # 1. Download price data
        task1 = progress.add_task("[cyan]Downloading price data...", total=1)
        try:
            from src.data.price_fetcher import PriceFetcher

            fetcher = PriceFetcher()
            prices = fetcher.fetch(start=start, end=end)
            for name, df in prices.items():
                df.to_csv(output_dir / f"prices_{name}.csv")
            console.print(f"  ✓ Downloaded price data for {len(prices)} instruments")
            progress.advance(task1)
        except Exception as exc:
            console.print(f"  [red]✗ Price data error: {exc}[/red]")

        # 2. EIA fundamental data
        if not skip_eia:
            task2 = progress.add_task("[cyan]Downloading EIA data...", total=1)
            try:
                from src.data.eia_client import EIAClient

                eia = EIAClient()
                crude_storage = eia.fetch_crude_storage(start=start, end=end)
                nat_gas_storage = eia.fetch_nat_gas_storage(start=start, end=end)
                crude_storage.to_csv(output_dir / "eia_crude_storage.csv")
                nat_gas_storage.to_csv(output_dir / "eia_natgas_storage.csv")
                console.print("  ✓ Downloaded EIA storage data")
                progress.advance(task2)
            except Exception as exc:
                console.print(f"  [yellow]⚠ EIA data unavailable: {exc}[/yellow]")

        # 3. FRED macro data
        if not skip_fred:
            task3 = progress.add_task("[cyan]Downloading FRED data...", total=1)
            try:
                from src.data.fred_client import FREDClient

                fred = FREDClient()
                macro = fred.fetch_macro_features(start=start, end=end)
                macro.to_csv(output_dir / "fred_macro.csv")
                console.print("  ✓ Downloaded FRED macro data")
                progress.advance(task3)
            except Exception as exc:
                console.print(f"  [yellow]⚠ FRED data unavailable: {exc}[/yellow]")

        # 4. News data
        if not skip_news:
            task4 = progress.add_task("[cyan]Downloading news articles...", total=1)
            try:
                from src.data.news_fetcher import NewsFetcher

                news = NewsFetcher()
                articles = news.fetch_all(start=start, end=end)
                if not articles.empty:
                    articles.to_csv(output_dir / "news_articles.csv", index=False)
                    console.print(f"  ✓ Downloaded {len(articles)} news articles")
                progress.advance(task4)
            except Exception as exc:
                console.print(f"  [yellow]⚠ News data unavailable: {exc}[/yellow]")

    console.print("\n[bold green]Data ingestion complete![/bold green]")


if __name__ == "__main__":
    main()
