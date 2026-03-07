"""Backtest analysis and visualisation utilities.

Provides tools to analyse backtest results including trade analysis,
drawdown decomposition, rolling performance metrics, and benchmark comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtesting.metrics import PerformanceMetrics, PerformanceReport

logger = logging.getLogger(__name__)


class BacktestAnalysis:
    """Provides analysis tools for backtest results.

    Takes the raw backtest DataFrame and metrics and produces
    higher-level analytical summaries.

    Attributes:
        result: Backtest result DataFrame from ``BacktestEngine.run()``.
        report: Computed ``PerformanceReport``.
        metrics_calc: ``PerformanceMetrics`` calculator.
    """

    def __init__(
        self,
        result: pd.DataFrame,
        trade_log: pd.DataFrame | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        """Initialise the backtest analyser.

        Args:
            result: DataFrame from ``BacktestEngine.run()``.
            trade_log: Optional trade log DataFrame.
            risk_free_rate: Risk-free rate for performance calculations.
        """
        self.result = result
        self.trade_log = trade_log
        self.metrics_calc = PerformanceMetrics(risk_free_rate=risk_free_rate)
        self.report: PerformanceReport | None = None

    def compute_metrics(self) -> PerformanceReport:
        """Compute and cache the performance report.

        Returns:
            Populated ``PerformanceReport``.
        """
        self.report = self.metrics_calc.compute(self.result, self.trade_log)
        return self.report

    def print_summary(self) -> None:
        """Print a formatted performance summary to the console."""
        if self.report is None:
            self.compute_metrics()
        print("\n" + "=" * 50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 50)
        for key, value in self.report.to_dict().items():  # type: ignore[union-attr]
            print(f"  {key:<35} {value}")
        print("=" * 50 + "\n")

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        """Compute rolling Sharpe ratio.

        Args:
            window: Rolling window in trading days.

        Returns:
            ``pd.Series`` of rolling Sharpe ratios.
        """
        returns = self.result["strategy_returns"]
        daily_rf = self.metrics_calc.risk_free_rate / self.metrics_calc.trading_days
        excess = returns - daily_rf
        rolling_sharpe = (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(self.metrics_calc.trading_days)
        )
        rolling_sharpe.name = f"rolling_sharpe_{window}d"
        return rolling_sharpe

    def monthly_returns(self) -> pd.DataFrame:
        """Compute monthly return breakdown.

        Returns:
            Pivot table with months as rows and years as columns.
        """
        returns = self.result["strategy_returns"].copy()
        monthly = (1 + returns).resample("ME").prod() - 1
        monthly.index = monthly.index.to_period("M")
        table = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            }
        )
        pivot = table.pivot(index="month", columns="year", values="return")
        return pivot

    def drawdown_periods(self, min_duration: int = 5) -> pd.DataFrame:
        """Extract significant drawdown periods.

        Args:
            min_duration: Minimum drawdown duration in days to include.

        Returns:
            DataFrame with drawdown period details.
        """
        portfolio = self.result["portfolio_value"]  # noqa: F841
        drawdown = self.result["drawdown"]
        in_dd = drawdown < 0
        periods = []
        start = None
        for _i, (dt, is_dd) in enumerate(in_dd.items()):
            if is_dd and start is None:
                start = dt
            elif not is_dd and start is not None:
                dd_slice = drawdown[start:dt]
                duration = (dt - start).days
                if duration >= min_duration:
                    periods.append(
                        {
                            "start": start,
                            "end": dt,
                            "duration_days": duration,
                            "max_drawdown": float(dd_slice.min()),
                            "trough_date": dd_slice.idxmin(),
                        }
                    )
                start = None
        return pd.DataFrame(periods) if periods else pd.DataFrame()

    def compare_to_benchmark(
        self, benchmark_returns: pd.Series, benchmark_name: str = "Buy & Hold"
    ) -> pd.DataFrame:
        """Compare strategy performance to a benchmark.

        Args:
            benchmark_returns: Benchmark daily return series.
            benchmark_name: Human-readable benchmark label.

        Returns:
            DataFrame with side-by-side performance metrics.
        """
        strategy_returns = self.result["strategy_returns"]
        strat_report = self.metrics_calc.compute(self.result)

        # Build benchmark backtest result for metrics
        bench_portfolio = (
            1 + benchmark_returns.reindex(strategy_returns.index).fillna(0)
        ).cumprod()
        bench_dd = (bench_portfolio - bench_portfolio.cummax()) / bench_portfolio.cummax()
        bench_result = pd.DataFrame(
            {
                "strategy_returns": benchmark_returns.reindex(strategy_returns.index).fillna(0),
                "portfolio_value": bench_portfolio,
                "drawdown": bench_dd,
                "position": pd.Series(1, index=strategy_returns.index),
            }
        )
        bench_report = self.metrics_calc.compute(bench_result)

        comparison = pd.DataFrame(
            {
                "Strategy": [
                    f"{strat_report.annualised_return:.2%}",
                    f"{strat_report.annualised_volatility:.2%}",
                    f"{strat_report.sharpe_ratio:.3f}",
                    f"{strat_report.max_drawdown:.2%}",
                    f"{strat_report.calmar_ratio:.3f}",
                ],
                benchmark_name: [
                    f"{bench_report.annualised_return:.2%}",
                    f"{bench_report.annualised_volatility:.2%}",
                    f"{bench_report.sharpe_ratio:.3f}",
                    f"{bench_report.max_drawdown:.2%}",
                    f"{bench_report.calmar_ratio:.3f}",
                ],
            },
            index=["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar"],
        )
        return comparison
