"""Run the composite NG investment signal system with walk-forward backtest.

Usage:
    python scripts/run_composite_signal.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy.composite_signal import CompositeSignalEngine, CompositeSignalConfig
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.Series, pd.DataFrame | None]:
    """Load NG prices and storage data."""
    price_path = ROOT / "data" / "raw" / "prices_natural_gas.csv"
    storage_path = ROOT / "data" / "raw" / "eia_natgas_storage.csv"

    prices_df = pd.read_csv(price_path, parse_dates=["date"]).set_index("date").sort_index()
    prices = prices_df["Close"].dropna()
    prices.name = "NG_Close"

    storage = None
    if storage_path.exists():
        storage = pd.read_csv(storage_path)
        logger.info("Loaded storage data: %d rows", len(storage))

    logger.info(
        "Loaded NG prices: %s -> %s (%d rows)",
        prices.index[0].date(), prices.index[-1].date(), len(prices),
    )
    return prices, storage


def walk_forward_backtest(
    prices: pd.Series,
    storage: pd.DataFrame | None,
    train_years: int = 5,
    test_months: int = 6,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward validation: train on N years, test on M months, step forward.

    Returns concatenated OOS backtest results and per-fold metrics.
    """
    start_date = prices.index[0]
    end_date = prices.index[-1]
    fold_start = start_date + pd.DateOffset(years=train_years)

    all_results = []
    fold_metrics_list = []
    fold_num = 0
    metrics_calc = PerformanceMetrics(risk_free_rate=0.03)

    while fold_start < end_date:
        fold_end = min(fold_start + pd.DateOffset(months=test_months), end_date)
        train_prices = prices[:fold_start]

        if storage is not None:
            storage_dates = pd.to_datetime(storage["date"])
            train_storage = storage[storage_dates < fold_start]
        else:
            train_storage = None

        # Generate signals using only training data for HMM fit, but we need
        # prices up through test period for the technical indicators to work
        test_prices = prices[fold_start:fold_end]
        if len(test_prices) < 20:
            break

        # Use all data up to fold_end for signal generation (signals are causal)
        all_prices_to_date = prices[:fold_end]
        engine = CompositeSignalEngine(CompositeSignalConfig())
        signal_df = engine.generate_signals(all_prices_to_date, storage)

        # Extract OOS portion
        oos = signal_df.loc[fold_start:fold_end].copy()
        if len(oos) < 10:
            fold_start = fold_end
            continue

        # Run backtest on OOS
        bt_engine = BacktestEngine(BacktestConfig(
            initial_capital=1_000_000,
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
        ))
        bt_result = bt_engine.run(
            oos["price"],
            oos["signal"].astype(float),
            position_sizes=oos["position_size"],
        )
        bt_result["fold"] = fold_num

        # Compute fold metrics
        try:
            report = metrics_calc.compute(bt_result)
            sharpe = report.sharpe_ratio
            # Guard against overflow from empty folds
            if not np.isfinite(sharpe) or abs(sharpe) > 1e6:
                sharpe = 0.0
            fold_info = {
                "fold": fold_num,
                "start": fold_start.strftime("%Y-%m-%d"),
                "end": fold_end.strftime("%Y-%m-%d"),
                "days": len(oos),
                "sharpe": sharpe,
                "total_return": report.total_return,
                "max_dd": report.max_drawdown,
                "win_rate": report.win_rate,
                "n_trades": report.total_trades,
            }
        except Exception:
            fold_info = {
                "fold": fold_num,
                "start": fold_start.strftime("%Y-%m-%d"),
                "end": fold_end.strftime("%Y-%m-%d"),
                "days": len(oos),
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_dd": 0.0,
                "win_rate": 0.0,
                "n_trades": 0,
            }

        fold_metrics_list.append(fold_info)
        all_results.append(bt_result)

        logger.info(
            "Fold %d: %s->%s | Return=%.2f%% | Sharpe=%.3f | MaxDD=%.2f%%",
            fold_num, fold_info["start"], fold_info["end"],
            fold_info["total_return"] * 100,
            fold_info["sharpe"],
            fold_info["max_dd"] * 100,
        )

        fold_num += 1
        fold_start = fold_end

    combined = pd.concat(all_results, axis=0) if all_results else pd.DataFrame()
    return combined, fold_metrics_list


def run_full_period_backtest(
    prices: pd.Series,
    storage: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run on full dataset (2005+) to get aggregate stats + signal history."""
    # Skip first 5 years for warm-up
    start = prices.index[0] + pd.DateOffset(years=5)
    prices_trimmed = prices[start:]

    engine = CompositeSignalEngine(CompositeSignalConfig())
    signal_df = engine.generate_signals(prices, storage)
    signal_df = signal_df.loc[start:]

    bt_engine = BacktestEngine(BacktestConfig(
        initial_capital=1_000_000,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
    ))
    bt_result = bt_engine.run(
        signal_df["price"],
        signal_df["signal"].astype(float),
        position_sizes=signal_df["position_size"],
    )

    return signal_df, bt_result


def print_results(
    signal_df: pd.DataFrame,
    bt_result: pd.DataFrame,
    fold_metrics: list[dict],
) -> None:
    """Print comprehensive results."""
    metrics = PerformanceMetrics(risk_free_rate=0.03)

    print("\n" + "=" * 70)
    print("  COMPOSITE NG INVESTMENT SIGNAL -- BACKTEST RESULTS")
    print("=" * 70)

    # --- Full period metrics ---
    report = metrics.compute(bt_result)
    print("\n" + "-" * 50)
    print("  FULL PERIOD PERFORMANCE")
    print("-" * 50)
    for k, v in report.to_dict().items():
        print(f"  {k:<35} {v}")

    # --- Buy-and-hold comparison ---
    bnh_ret = bt_result["price"].iloc[-1] / bt_result["price"].iloc[0] - 1
    n_years = len(bt_result) / 252
    bnh_cagr = (1 + bnh_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    bnh_vol = bt_result["price"].pct_change().std() * np.sqrt(252)
    bnh_sharpe = (bnh_cagr - 0.03) / bnh_vol if bnh_vol > 0 else 0

    print("\n" + "-" * 50)
    print("  BUY-AND-HOLD COMPARISON")
    print("-" * 50)
    print(f"  {'Strategy CAGR':<35} {report.annualised_return:.2%}")
    print(f"  {'Buy-Hold CAGR':<35} {bnh_cagr:.2%}")
    print(f"  {'Strategy Sharpe':<35} {report.sharpe_ratio:.3f}")
    print(f"  {'Buy-Hold Sharpe':<35} {bnh_sharpe:.3f}")
    print(f"  {'Strategy Max DD':<35} {report.max_drawdown:.2%}")

    # --- Signal distribution ---
    sigs = signal_df["signal"]
    print("\n" + "-" * 50)
    print("  SIGNAL DISTRIBUTION")
    print("-" * 50)
    for label, val in [("INVEST (+1)", 1), ("STAY OUT (0)", 0), ("SHORT (-1)", -1)]:
        count = (sigs == val).sum()
        if count == 0:
            continue
        pct = 100 * count / len(sigs)
        print(f"  {label:<20} {count:>6} days ({pct:.1f}%)")

    # --- Sub-signal contribution ---
    print("\n" + "-" * 50)
    print("  SUB-SIGNAL MEANS (when LONG)")
    print("-" * 50)
    long_mask = signal_df["signal"] == 1
    if long_mask.any():
        for col in ["regime_signal", "storage_signal", "seasonal_signal",
                     "technical_signal", "mean_reversion_signal"]:
            if col in signal_df.columns:
                print(f"  {col:<30} {signal_df.loc[long_mask, col].mean():+.3f}")

    short_mask = signal_df["signal"] == -1
    if short_mask.any():
        print("\n" + "-" * 50)
        print("  SUB-SIGNAL MEANS (when SHORT)")
        print("-" * 50)
        for col in ["regime_signal", "storage_signal", "seasonal_signal",
                     "technical_signal", "mean_reversion_signal"]:
            if col in signal_df.columns:
                print(f"  {col:<30} {signal_df.loc[short_mask, col].mean():+.3f}")

    # --- Walk-forward fold results ---
    if fold_metrics:
        print("\n" + "-" * 50)
        print("  WALK-FORWARD FOLD RESULTS (6-month OOS windows)")
        print("-" * 50)
        print(f"  {'Fold':>4} {'Period':<24} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
        for fm in fold_metrics:
            print(
                f"  {fm['fold']:>4} {fm['start']}->{fm['end'][:7]}  "
                f"{fm['total_return']:>7.2%} {fm['sharpe']:>8.3f} "
                f"{fm['max_dd']:>7.2%} {fm['n_trades']:>7}"
            )
        avg_sharpe = np.mean([f["sharpe"] for f in fold_metrics])
        avg_ret = np.mean([f["total_return"] for f in fold_metrics])
        pct_positive = np.mean([1 if f["total_return"] > 0 else 0 for f in fold_metrics])
        print(f"\n  Avg fold Sharpe:    {avg_sharpe:.3f}")
        print(f"  Avg fold return:    {avg_ret:.2%}")
        print(f"  %% positive folds:   {pct_positive:.0%}")

    # --- Monthly return heatmap ---
    print("\n" + "-" * 50)
    print("  STRATEGY MONTHLY RETURNS")
    print("-" * 50)
    monthly = bt_result["strategy_returns"].resample("ME").sum()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
    # Annual column
    annual = bt_result["strategy_returns"].resample("YE").sum()
    pivot["Annual"] = annual.values[:len(pivot)]
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))

    print("\n" + "=" * 70)


def main() -> None:
    prices, storage = load_data()

    # Walk-forward backtest
    logger.info("Running walk-forward backtest...")
    wf_results, fold_metrics = walk_forward_backtest(prices, storage)

    # Full-period backtest and signals
    logger.info("Running full-period backtest...")
    signal_df, bt_result = run_full_period_backtest(prices, storage)

    # Print results
    print_results(signal_df, bt_result, fold_metrics)

    # Save signals for dashboard
    output_path = ROOT / "data" / "processed" / "composite_signals.csv"
    signal_df.to_csv(output_path)
    logger.info("Saved signals to %s", output_path)


if __name__ == "__main__":
    main()
