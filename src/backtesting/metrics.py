"""Performance metrics for backtesting evaluation.

Computes standard quant finance performance metrics including Sharpe ratio,
Sortino ratio, Calmar ratio, maximum drawdown, win rate, and profit factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive backtest performance report.

    Attributes:
        total_return: Total strategy return (decimal).
        annualised_return: Annualised return (CAGR).
        annualised_volatility: Annualised return standard deviation.
        sharpe_ratio: Annualised Sharpe ratio.
        sortino_ratio: Annualised Sortino ratio (downside deviation).
        calmar_ratio: Annualised return / max drawdown.
        max_drawdown: Maximum peak-to-trough drawdown.
        max_drawdown_duration_days: Longest drawdown period in days.
        win_rate: Fraction of profitable trades.
        profit_factor: Gross profit / gross loss ratio.
        total_trades: Number of round-trip trades.
        avg_trade_return: Average return per trade.
        best_trade: Best single trade return.
        worst_trade: Worst single trade return.
        var_95: Historical 95% VaR (daily).
        var_99: Historical 99% VaR (daily).
    """

    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0

    def to_dict(self) -> dict[str, str | int]:
        """Serialise metrics to a flat dictionary."""
        return {
            "Total Return": f"{self.total_return:.2%}",
            "Annualised Return (CAGR)": f"{self.annualised_return:.2%}",
            "Annualised Volatility": f"{self.annualised_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max Drawdown Duration (days)": self.max_drawdown_duration_days,
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Total Trades": self.total_trades,
            "Avg Trade Return": f"{self.avg_trade_return:.4%}",
            "Best Trade": f"{self.best_trade:.2%}",
            "Worst Trade": f"{self.worst_trade:.2%}",
            "VaR 95% (daily)": f"{self.var_95:.4%}",
            "VaR 99% (daily)": f"{self.var_99:.4%}",
        }


class PerformanceMetrics:
    """Computes performance metrics from backtest results.

    Takes the backtest result DataFrame from ``BacktestEngine.run()``
    and computes the full set of performance statistics.
    """

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252) -> None:
        """Initialise the metrics calculator.

        Args:
            risk_free_rate: Annualised risk-free rate for Sharpe calculation.
            trading_days: Number of trading days per year (typically 252).
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def compute(
        self,
        backtest_result: pd.DataFrame,
        trade_log: pd.DataFrame | None = None,
    ) -> PerformanceReport:
        """Compute all performance metrics.

        Args:
            backtest_result: DataFrame from ``BacktestEngine.run()``.
            trade_log: Optional trade log from ``Portfolio.get_trade_log()``.

        Returns:
            Populated ``PerformanceReport`` dataclass.
        """
        returns = backtest_result["strategy_returns"].dropna()
        portfolio = backtest_result["portfolio_value"].dropna()

        report = PerformanceReport()

        # Return statistics
        report.total_return = self._total_return(portfolio)
        report.annualised_return = self._cagr(portfolio)
        report.annualised_volatility = self._annualised_vol(returns)

        # Risk-adjusted returns
        report.sharpe_ratio = self._sharpe(returns)
        report.sortino_ratio = self._sortino(returns)

        # Drawdown analysis
        drawdown_series = backtest_result["drawdown"].dropna()
        report.max_drawdown = float(drawdown_series.min())
        report.calmar_ratio = self._calmar(report.annualised_return, report.max_drawdown)
        report.max_drawdown_duration_days = self._max_dd_duration(portfolio)

        # VaR
        report.var_95 = float(np.percentile(returns, 5))
        report.var_99 = float(np.percentile(returns, 1))

        # Trade statistics
        if trade_log is not None and not trade_log.empty:
            report.total_trades = len(trade_log)
            pnl = trade_log["pnl_pct"]
            report.win_rate = float((pnl > 0).mean())
            gross_profit = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            report.avg_trade_return = float(pnl.mean())
            report.best_trade = float(pnl.max())
            report.worst_trade = float(pnl.min())
        else:
            # Estimate trade stats from position changes
            position_changes = backtest_result["position"].diff().abs()
            report.total_trades = int((position_changes > 0).sum())

        logger.info(
            "Performance report computed: Sharpe=%.3f, MaxDD=%.2f%%",
            report.sharpe_ratio,
            report.max_drawdown * 100,
        )
        return report

    def _total_return(self, portfolio: pd.Series) -> float:
        return float(portfolio.iloc[-1] / portfolio.iloc[0] - 1)

    def _cagr(self, portfolio: pd.Series) -> float:
        n_years = len(portfolio) / self.trading_days
        if n_years <= 0:
            return 0.0
        return float((portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / n_years) - 1)

    def _annualised_vol(self, returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(self.trading_days))

    def _sharpe(self, returns: pd.Series) -> float:
        daily_rf = self.risk_free_rate / self.trading_days
        excess = returns - daily_rf
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.trading_days))

    def _sortino(self, returns: pd.Series) -> float:
        daily_rf = self.risk_free_rate / self.trading_days
        excess = returns - daily_rf
        downside = excess[excess < 0].std()
        if downside == 0:
            return 0.0
        return float(excess.mean() / downside * np.sqrt(self.trading_days))

    def _calmar(self, cagr: float, max_dd: float) -> float:
        if max_dd == 0:
            return 0.0
        return abs(cagr / max_dd)

    def _max_dd_duration(self, portfolio: pd.Series) -> int:
        """Compute the maximum drawdown duration in trading days."""
        peak = portfolio.cummax()
        in_drawdown = portfolio < peak
        # Count consecutive drawdown days
        max_dur = 0
        current_dur = 0
        for dd in in_drawdown:
            if dd:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0
        return max_dur
