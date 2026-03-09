#!/usr/bin/env python3
"""Ingest weather and electricity demand data for load forecasting.

Fetches data from two sources:
  1. **Open-Meteo API** — daily temperatures for 10 Eastern/Central US cities.
  2. **EIA API v2** — hourly electricity demand (PJM, MISO, NYIS, ISNE) → daily.

Saves raw CSVs to ``data/raw/`` and a merged feature-ready dataset
to ``data/processed/``.

Usage:
    python scripts/ingest_demand_data.py [OPTIONS]

Examples:
    # Full fetch from 2018 to present
    python scripts/ingest_demand_data.py

    # Only fetch weather (skip EIA — useful if EIA is slow)
    python scripts/ingest_demand_data.py --weather-only

    # Custom date range
    python scripts/ingest_demand_data.py --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: 5 days ago)")
@click.option("--output-dir", default="data", help="Base output directory")
@click.option("--weather-only", is_flag=True, help="Only fetch weather (skip EIA)")
@click.option("--demand-only", is_flag=True, help="Only fetch EIA demand (skip weather)")
@click.option("--regions", default="PJM,MISO,NYIS,ISNE", help="Comma-separated EIA regions")
def main(
    start: str,
    end: str | None,
    output_dir: str,
    weather_only: bool,
    demand_only: bool,
    regions: str,
) -> None:
    """Fetch and cache weather + electricity demand data."""
    from rich.console import Console

    console = Console()
    out = Path(output_dir)
    raw_dir = out / "raw"
    proc_dir = out / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Power/Gas Load Forecasting — Data Ingestion[/bold green]\n")
    console.print(f"  Date range : {start} — {end or 'present'}")
    console.print(f"  Output     : {out.resolve()}\n")

    weather_df = None
    demand_df = None

    # ── Weather ──────────────────────────────────────────────────────
    if not demand_only:
        console.print("[cyan]1. Fetching weather data (Open-Meteo)...[/cyan]")
        try:
            from src.data.openmeteo_client import OpenMeteoClient

            client = OpenMeteoClient()
            weather_df = client.fetch_weather(start=start, end=end)
            weather_path = raw_dir / "weather_daily.csv"
            weather_df.to_csv(weather_path)
            console.print(
                f"   ✓ {len(weather_df)} days, {len(weather_df.columns)} columns "
                f"→ {weather_path}"
            )
        except Exception as exc:
            console.print(f"   [red]✗ Weather fetch failed: {exc}[/red]")
            # Try loading cached
            cached = raw_dir / "weather_daily.csv"
            if cached.exists():
                weather_df = pd.read_csv(cached, index_col=0, parse_dates=True)
                console.print(f"   [yellow]  Loaded cached weather ({len(weather_df)} rows)[/yellow]")

    # ── Electricity Demand ───────────────────────────────────────────
    if not weather_only:
        console.print("\n[cyan]2. Fetching electricity demand (EIA)...[/cyan]")
        region_list = [r.strip() for r in regions.split(",")]
        try:
            from src.data.eia_demand_client import EIADemandClient

            client = EIADemandClient()
            demand_df = client.fetch_daily_demand(
                regions=region_list, start=start, end=end,
            )
            demand_path = raw_dir / "demand_daily.csv"
            demand_df.to_csv(demand_path)
            console.print(
                f"   ✓ {len(demand_df)} days, {len(demand_df.columns)} columns "
                f"→ {demand_path}"
            )
        except Exception as exc:
            console.print(f"   [red]✗ Demand fetch failed: {exc}[/red]")
            cached = raw_dir / "demand_daily.csv"
            if cached.exists():
                demand_df = pd.read_csv(cached, index_col=0, parse_dates=True)
                console.print(f"   [yellow]  Loaded cached demand ({len(demand_df)} rows)[/yellow]")

    # ── Merge & Build Features ───────────────────────────────────────
    if weather_df is not None and demand_df is not None:
        console.print("\n[cyan]3. Building load forecasting features...[/cyan]")
        try:
            from src.features.load_features import LoadFeatureBuilder

            builder = LoadFeatureBuilder(target_col="east_total_mwh")
            features = builder.build(demand_df, weather_df)
            feat_path = proc_dir / "load_features.csv"
            features.to_csv(feat_path)
            console.print(
                f"   ✓ {len(features)} rows × {len(features.columns) - 1} features "
                f"→ {feat_path}"
            )
        except Exception as exc:
            console.print(f"   [red]✗ Feature build failed: {exc}[/red]")
    elif weather_df is not None:
        console.print("\n[yellow]Demand data not available — saved weather only.[/yellow]")
    elif demand_df is not None:
        console.print("\n[yellow]Weather data not available — saved demand only.[/yellow]")
    else:
        console.print("\n[red]No data fetched.  Check API keys and network.[/red]")
        return

    console.print("\n[bold green]Data ingestion complete![/bold green]")


if __name__ == "__main__":
    main()
