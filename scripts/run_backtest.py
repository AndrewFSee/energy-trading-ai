#!/usr/bin/env python3
"""CLI script to run the full backtesting engine.

Loads price data and model predictions, runs the backtest,
and outputs a performance report.

Usage:
    python scripts/run_backtest.py [OPTIONS]
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
@click.option("--processed-dir", default="data/processed", help="Processed data directory")
@click.option("--models-dir", default="models", help="Trained models directory")
@click.option("--instrument", default="wti", help="Instrument to backtest")
@click.option("--start", default="2020-01-01", help="Backtest start date")
@click.option("--end", default=None, help="Backtest end date")
@click.option("--capital", default=1_000_000, help="Initial capital in USD")
@click.option("--cost-bps", default=5, help="Transaction cost in basis points")
@click.option("--output-dir", default="data/processed", help="Output directory for results")
def main(
    raw_dir: str,
    processed_dir: str,
    models_dir: str,
    instrument: str,
    start: str,
    end: str | None,
    capital: float,
    cost_bps: float,
    output_dir: str,
) -> None:
    """Run the backtesting engine on historical data."""
    from rich.console import Console

    console = Console()
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Energy Trading AI — Backtest[/bold green]")
    console.print(f"  Instrument: {instrument.upper()}")
    console.print(f"  Period: {start} → {end or 'latest'}")
    console.print(f"  Capital: ${capital:,.0f}")
    console.print(f"  Transaction cost: {cost_bps}bps\n")

    # Load price data
    price_file = raw_path / f"prices_{instrument}.csv"
    if not price_file.exists():
        console.print(f"[red]Price data not found: {price_file}[/red]")
        return

    prices_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    prices = prices_df["Close"].dropna()

    # Load feature data for signal generation
    feature_file = processed_path / f"features_{instrument}.csv"
    if not feature_file.exists():
        console.print("[yellow]Feature data not found — using simple MA crossover signals[/yellow]")
        # Fallback: simple momentum signal
        fast_ma = prices.rolling(20).mean()
        slow_ma = prices.rolling(50).mean()
        signals = pd.Series(
            (fast_ma > slow_ma).astype(int) * 2 - 1,
            index=prices.index,
        )
    else:
        # Use model predictions as signals (simplified)
        features_df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        # Use return_5d as a simple signal proxy
        signals = pd.Series(0, index=features_df.index)
        if "return_5d" in features_df.columns:
            from src.strategy.signals import SignalGenerator

            gen = SignalGenerator()
            signals = gen.generate(features_df["return_5d"].fillna(0))

    from src.backtesting.analysis import BacktestAnalysis
    from src.backtesting.engine import BacktestConfig, BacktestEngine
    from src.strategy.risk_manager import RiskManager

    config = BacktestConfig(
        initial_capital=float(capital),
        transaction_cost_bps=float(cost_bps),
        slippage_bps=2.0,
        start_date=start,
        end_date=end,
    )
    risk_mgr = RiskManager()
    engine = BacktestEngine(config=config, risk_manager=risk_mgr)

    console.print("[cyan]Running backtest...[/cyan]")
    result = engine.run(prices, signals)

    # Analysis
    analysis = BacktestAnalysis(result)
    analysis.print_summary()

    # Save results
    result_file = output_path / f"backtest_{instrument}.csv"
    result.to_csv(result_file)
    console.print(f"\n  Results saved to: {result_file}")
    console.print("\n[bold green]Backtest complete![/bold green]")


if __name__ == "__main__":
    main()
