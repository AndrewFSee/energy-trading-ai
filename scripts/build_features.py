#!/usr/bin/env python3
"""CLI script to build the feature matrix from raw data.

Reads raw price and fundamental data, applies all feature engineering
steps, and saves the feature matrix to data/processed/.

Usage:
    python scripts/build_features.py [OPTIONS]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--raw-dir", default="data/raw", help="Raw data directory")
@click.option("--output-dir", default="data/processed", help="Processed data directory")
@click.option("--instrument", default="wti", help="Instrument to process (e.g. wti, natural_gas)")
@click.option("--forecast-horizon", default=5, help="Forecast horizon in days")
def main(raw_dir: str, output_dir: str, instrument: str, forecast_horizon: int) -> None:
    """Build feature matrix from raw data."""
    from rich.console import Console

    console = Console()
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Building Feature Matrix for {instrument.upper()}[/bold green]\n")

    # Load price data
    price_file = raw_path / f"prices_{instrument}.csv"
    if not price_file.exists():
        console.print(f"[red]Price data not found: {price_file}[/red]")
        console.print("Run: python scripts/ingest_data.py first")
        return

    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    console.print(f"  Loaded price data: {len(price_df)} rows")

    # Load optional data
    storage_df = None
    storage_file = raw_path / "eia_crude_storage.csv"
    if storage_file.exists():
        storage_df = pd.read_csv(storage_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded storage data: {len(storage_df)} rows")

    macro_df = None
    macro_file = raw_path / "fred_macro.csv"
    if macro_file.exists():
        macro_df = pd.read_csv(macro_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded macro data: {len(macro_df)} rows")

    sentiment_df = None
    sentiment_file = raw_path / "sentiment_index.csv"
    if sentiment_file.exists():
        sentiment_df = pd.read_csv(sentiment_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded sentiment data: {len(sentiment_df)} rows")

    # Build features
    console.print("\n  Building feature matrix...")
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    feature_df = pipeline.build(
        price_df=price_df,
        storage_df=storage_df,
        macro_df=macro_df,
        sentiment_df=sentiment_df,
    )

    # Create targets
    target = pipeline.create_targets(feature_df, horizon=forecast_horizon)
    feature_df["target"] = target

    # Save
    output_file = out_path / f"features_{instrument}.csv"
    feature_df.to_csv(output_file)
    console.print(f"\n  ✓ Feature matrix saved: {output_file}")
    console.print(f"  Shape: {feature_df.shape}")
    console.print(f"  Features: {len(pipeline.feature_columns)}")
    console.print("\n[bold green]Feature engineering complete![/bold green]")


if __name__ == "__main__":
    main()
