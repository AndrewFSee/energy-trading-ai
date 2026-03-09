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

    # Load GDELT sentiment (2015+)
    gdelt_file = raw_path / "gdelt_sentiment.csv"
    if gdelt_file.exists():
        gdelt_df = pd.read_csv(gdelt_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded GDELT sentiment: {len(gdelt_df)} rows, {gdelt_df.shape[1]} features")
        # Merge GDELT into sentiment_df (or create it)
        if sentiment_df is not None:
            new_cols = [c for c in gdelt_df.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(gdelt_df[new_cols], how="outer")
        else:
            sentiment_df = gdelt_df

    # Load FRED sentiment proxies (EPU, credit spread, financial stress, etc.)
    fred_sent_file = raw_path / "fred_sentiment.csv"
    if fred_sent_file.exists():
        fred_sent_df = pd.read_csv(fred_sent_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded FRED sentiment proxies: {len(fred_sent_df)} rows, {fred_sent_df.shape[1]} features")
        # Merge FRED sentiment into sentiment_df
        if sentiment_df is not None:
            new_cols = [c for c in fred_sent_df.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(fred_sent_df[new_cols], how="outer")
        else:
            sentiment_df = fred_sent_df

    # Load GPR (Geopolitical Risk Index) — daily, 1985+
    gpr_file = raw_path / "gpr_daily.csv"
    if gpr_file.exists():
        gpr_df = pd.read_csv(gpr_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded GPR index: {len(gpr_df)} rows, {gpr_df.shape[1]} features")
        if sentiment_df is not None:
            new_cols = [c for c in gpr_df.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(gpr_df[new_cols], how="outer")
        else:
            sentiment_df = gpr_df

    # Load Gold futures as geopolitical safe-haven proxy
    gold_file = raw_path / "prices_gold.csv"
    if gold_file.exists():
        import numpy as np
        gold_df = pd.read_csv(gold_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded Gold data: {len(gold_df)} rows")
        gold_close = gold_df["Close"].rename("gold")
        gold_features = pd.DataFrame(index=gold_df.index)
        gold_features["gold"] = gold_close
        gold_features["gold_return_5d"] = gold_close.pct_change(5)
        gold_features["gold_return_20d"] = gold_close.pct_change(20)
        roll_g = gold_close.rolling(60, min_periods=10)
        gold_features["gold_zscore"] = (gold_close - roll_g.mean()) / roll_g.std()
        gold_features["gold_ma_ratio"] = gold_close / gold_close.rolling(20, min_periods=1).mean()
        # Gold/Oil ratio — classic geopolitical signal
        wti_close = price_df["Close"] if "Close" in price_df.columns else None
        if wti_close is not None:
            aligned_gold = gold_close.reindex(price_df.index).ffill()
            gold_features["gold_oil_ratio"] = aligned_gold / wti_close
            gor_roll = gold_features["gold_oil_ratio"].rolling(60, min_periods=10)
            gold_features["gold_oil_ratio_zscore"] = (
                gold_features["gold_oil_ratio"] - gor_roll.mean()
            ) / gor_roll.std()
        console.print(f"  Built Gold features: {gold_features.shape[1]} columns")
        if sentiment_df is not None:
            new_cols = [c for c in gold_features.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(gold_features[new_cols], how="outer")
        else:
            sentiment_df = gold_features

    # Load DXY (Dollar Index) as macro/geopolitical feature
    dxy_file = raw_path / "prices_dxy.csv"
    if dxy_file.exists():
        import numpy as np
        dxy_df = pd.read_csv(dxy_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded DXY data: {len(dxy_df)} rows")
        dxy_close = dxy_df["Close"].rename("dxy")
        dxy_features = pd.DataFrame(index=dxy_df.index)
        dxy_features["dxy"] = dxy_close
        dxy_features["dxy_return_5d"] = dxy_close.pct_change(5)
        dxy_features["dxy_return_20d"] = dxy_close.pct_change(20)
        roll_d = dxy_close.rolling(60, min_periods=10)
        dxy_features["dxy_zscore"] = (dxy_close - roll_d.mean()) / roll_d.std()
        dxy_features["dxy_ma_ratio"] = dxy_close / dxy_close.rolling(20, min_periods=1).mean()
        console.print(f"  Built DXY features: {dxy_features.shape[1]} columns")
        if sentiment_df is not None:
            new_cols = [c for c in dxy_features.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(dxy_features[new_cols], how="outer")
        else:
            sentiment_df = dxy_features

    # Load OVX (CBOE Crude Oil Volatility Index) as an oil-specific fear gauge
    ovx_file = raw_path / "prices_ovx.csv"
    if ovx_file.exists():
        ovx_df = pd.read_csv(ovx_file, index_col=0, parse_dates=True)
        console.print(f"  Loaded OVX data: {len(ovx_df)} rows")
        import numpy as np
        ovx_close = ovx_df["Close"].rename("ovx")
        ovx_features = pd.DataFrame(index=ovx_df.index)
        ovx_features["ovx"] = ovx_close
        ovx_features["ovx_change_5d"] = ovx_close.diff(5)
        ovx_features["ovx_change_20d"] = ovx_close.diff(20)
        roll = ovx_close.rolling(60, min_periods=10)
        ovx_features["ovx_zscore"] = (ovx_close - roll.mean()) / roll.std()
        ovx_features["ovx_high"] = (ovx_close > 40).astype(int)
        ovx_features["ovx_extreme"] = (ovx_close > 60).astype(int)
        ovx_features["ovx_ma_ratio"] = ovx_close / ovx_close.rolling(20, min_periods=1).mean()
        console.print(f"  Built OVX features: {ovx_features.shape[1]} columns")
        if sentiment_df is not None:
            new_cols = [c for c in ovx_features.columns if c not in sentiment_df.columns]
            if new_cols:
                sentiment_df = sentiment_df.join(ovx_features[new_cols], how="outer")
        else:
            sentiment_df = ovx_features

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
