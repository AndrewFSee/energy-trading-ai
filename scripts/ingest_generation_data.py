#!/usr/bin/env python
"""Ingest wind/solar generation, extended weather, and NG production data.

Fetches from three sources:
  1. EIA hourly generation by fuel type → daily aggregation
  2. Open-Meteo wind speed + solar radiation + temperature
  3. NG production fundamentals (rig counts + monthly production)

Saves raw data to ``data/raw/`` and processed features to ``data/processed/``.

Usage::

    python scripts/ingest_generation_data.py
    python scripts/ingest_generation_data.py --start 2020-01-01 --skip-generation
    python scripts/ingest_generation_data.py --skip-production
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from src.data.eia_generation_client import EIAGenerationClient
from src.data.eia_lmp_client import EIADemandClient
from src.data.ng_production_client import NGProductionClient
from src.data.openmeteo_client import OpenMeteoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ingest_generation(start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily generation by fuel type from EIA."""
    logger.info("=" * 60)
    logger.info("STEP 1: EIA Generation Data (all fuel types)")
    logger.info("=" * 60)

    try:
        client = EIAGenerationClient()
    except ValueError as exc:
        logger.error("Cannot create EIAGenerationClient: %s", exc)
        return None

    # Fetch ALL fuel types for complete generation mix
    gen_df = client.fetch_daily_generation(
        regions=["PJM", "MISO", "NYIS", "ISNE"],
        fuel_types=["WND", "SUN", "NG", "NUC", "COL", "WAT", "OTH"],
        start=start,
        end=end,
    )

    if gen_df.empty:
        logger.warning("No generation data retrieved")
        return None

    # Save raw
    out_path = RAW_DIR / "generation_daily.csv"
    gen_df.to_csv(out_path)
    logger.info("Saved generation data: %s (%d rows × %d cols)",
                out_path, len(gen_df), len(gen_df.columns))
    return gen_df


def ingest_lmp(start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily regional electricity demand from EIA."""
    logger.info("=" * 60)
    logger.info("STEP 1b: EIA Regional Demand Data")
    logger.info("=" * 60)

    try:
        client = EIADemandClient()
    except ValueError as exc:
        logger.error("Cannot create EIADemandClient: %s", exc)
        return None

    demand_df = client.fetch_daily_demand(
        regions=["PJM", "MISO", "NYIS", "ISNE"],
        start=start,
        end=end,
    )

    if demand_df.empty:
        logger.warning("No demand data retrieved")
        return None

    out_path = RAW_DIR / "demand_daily.csv"
    demand_df.to_csv(out_path)
    logger.info("Saved demand data: %s (%d rows × %d cols)",
                out_path, len(demand_df), len(demand_df.columns))
    return demand_df


def ingest_wind_solar_weather(start: str, end: str) -> pd.DataFrame | None:
    """Fetch extended weather variables (wind + solar + temp)."""
    logger.info("=" * 60)
    logger.info("STEP 2: Wind + Solar Weather Data (Open-Meteo)")
    logger.info("=" * 60)

    client = OpenMeteoClient()
    weather_df = client.fetch_wind_solar_weather(start=start, end=end)

    if weather_df.empty:
        logger.warning("No weather data retrieved")
        return None

    out_path = RAW_DIR / "weather_wind_solar.csv"
    weather_df.to_csv(out_path)
    logger.info("Saved weather data: %s (%d rows × %d cols)",
                out_path, len(weather_df), len(weather_df.columns))
    return weather_df


def ingest_ng_production(start: str, end: str | None = None) -> pd.DataFrame | None:
    """Fetch NG production fundamentals (rig counts + monthly production)."""
    logger.info("=" * 60)
    logger.info("STEP 3: NG Production Fundamentals")
    logger.info("=" * 60)

    try:
        client = NGProductionClient()
    except ValueError as exc:
        logger.error("Cannot create NGProductionClient: %s", exc)
        return None

    prod_df = client.fetch_production_fundamentals(start=start, end=end)

    if prod_df.empty:
        logger.warning("No production data retrieved")
        return None

    out_path = RAW_DIR / "ng_production.csv"
    prod_df.to_csv(out_path)
    logger.info("Saved NG production data: %s (%d rows × %d cols)",
                out_path, len(prod_df), len(prod_df.columns))
    return prod_df


def build_wind_features(
    gen_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build wind generation feature matrix."""
    logger.info("=" * 60)
    logger.info("STEP 4: Build Wind Generation Features")
    logger.info("=" * 60)

    from src.features.wind_gen_features import WindGenFeatureBuilder

    if "east_wind_total_mwh" not in gen_df.columns:
        logger.warning("No wind generation data — cannot build features")
        return None

    builder = WindGenFeatureBuilder(target_col="east_wind_total_mwh")
    features = builder.build(gen_df, weather_df)

    out_path = PROCESSED_DIR / "wind_gen_features.csv"
    features.to_csv(out_path)
    logger.info("Saved wind features: %s (%d rows × %d cols)",
                out_path, len(features), len(features.columns))
    return features


def build_ng_production_features(prod_df: pd.DataFrame) -> pd.DataFrame | None:
    """Build NG production direction features."""
    logger.info("=" * 60)
    logger.info("STEP 5: Build NG Production Trend Features")
    logger.info("=" * 60)

    from src.features.ng_production_features import NGProductionFeatureBuilder

    # Try to load NG prices for price-signal features
    price_df = None
    price_path = RAW_DIR / "prices_natural_gas.csv"
    if price_path.exists():
        try:
            price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
            logger.info("Loaded NG prices (%d rows) for production features", len(price_df))
        except Exception:
            logger.warning("Could not load NG prices — proceeding without")

    builder = NGProductionFeatureBuilder(forecast_horizon_days=90)
    features = builder.build(prod_df, price_df=price_df)

    out_path = PROCESSED_DIR / "ng_production_features.csv"
    features.to_csv(out_path)
    logger.info("Saved NG production features: %s (%d rows × %d cols)",
                out_path, len(features), len(features.columns))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest generation + production data")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip EIA generation fetch (use cached)")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Skip weather fetch (use cached)")
    parser.add_argument("--skip-production", action="store_true",
                        help="Skip NG production fetch (use cached)")
    parser.add_argument("--skip-lmp", action="store_true",
                        help="Skip LMP price fetch (use cached)")
    args = parser.parse_args()

    end = args.end or (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    # Ensure output directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generation data
    gen_df = None
    if not args.skip_generation:
        gen_df = ingest_generation(args.start, end)
    else:
        cached = RAW_DIR / "generation_daily.csv"
        if cached.exists():
            gen_df = pd.read_csv(cached, index_col=0, parse_dates=True)
            logger.info("Loaded cached generation data: %d rows", len(gen_df))

    # Step 1b: LMP prices
    if not getattr(args, 'skip_lmp', False):
        ingest_lmp(args.start, end)
    else:
        cached = RAW_DIR / "lmp_daily.csv"
        if cached.exists():
            logger.info("Using cached LMP data")

    # Step 2: Weather data
    weather_df = None
    if not args.skip_weather:
        weather_df = ingest_wind_solar_weather(args.start, end)
    else:
        cached = RAW_DIR / "weather_wind_solar.csv"
        if cached.exists():
            weather_df = pd.read_csv(cached, index_col=0, parse_dates=True)
            logger.info("Loaded cached weather data: %d rows", len(weather_df))

    # Step 3: NG production
    prod_df = None
    if not args.skip_production:
        prod_df = ingest_ng_production("2000-01-01", end)
    else:
        cached = RAW_DIR / "ng_production.csv"
        if cached.exists():
            prod_df = pd.read_csv(cached, index_col=0, parse_dates=True)
            logger.info("Loaded cached production data: %d rows", len(prod_df))

    # Step 4: Build wind generation features
    if gen_df is not None and weather_df is not None:
        build_wind_features(gen_df, weather_df)
    else:
        logger.warning("Skipping wind feature build (missing generation or weather data)")

    # Step 5: Build NG production features
    if prod_df is not None:
        build_ng_production_features(prod_df)
    else:
        logger.warning("Skipping production feature build (no production data)")

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
