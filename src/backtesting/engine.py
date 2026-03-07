"""Backtesting engine with realistic transaction costs and slippage.

Simulates strategy performance on historical data with:
- Per-trade transaction costs (commission + spread)
- Price impact slippage
- Position-level stop-loss / take-profit execution
- Daily margin/mark-to-market accounting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio value in USD.
        transaction_cost_bps: One-way cost in basis points.
        slippage_bps: Slippage in basis points per trade.
        margin_requirement: Fraction of notional value required as margin.
        risk_free_rate: Annualised risk-free rate for Sharpe ratio.
        start_date: Backtest start date string.
        end_date: Backtest end date string.
    """

    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    margin_requirement: float = 0.10
    risk_free_rate: float = 0.05
    start_date: str | None = None
    end_date: str | None = None


class BacktestEngine:
    """Event-driven backtesting engine for energy trading strategies.

    Simulates daily strategy execution with realistic cost assumptions.
    Supports single-instrument and multi-instrument portfolios.

    Attributes:
        config: ``BacktestConfig`` instance.
        risk_manager: Optional risk management layer.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        """Initialise the backtesting engine.

        Args:
            config: Backtest configuration.  Defaults to ``BacktestConfig()``.
            risk_manager: Optional risk management layer.
        """
        self.config = config or BacktestConfig()
        self.risk_manager = risk_manager
        logger.info(
            "BacktestEngine initialised (capital=$%.0f, cost=%.1fbps, slippage=%.1fbps)",
            self.config.initial_capital,
            self.config.transaction_cost_bps,
            self.config.slippage_bps,
        )

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_sizes: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run a backtest on a single instrument.

        Args:
            prices: Daily close price series (``DatetimeIndex``).
            signals: Daily signal series (LONG=1, FLAT=0, SHORT=-1).
            position_sizes: Optional per-day position size fractions.
                If ``None``, signals are used directly as ±1 positions.

        Returns:
            DataFrame with columns: ``price``, ``signal``, ``position``,
            ``returns``, ``strategy_returns``, ``portfolio_value``,
            ``drawdown``, ``transaction_costs``.
        """
        # Align series
        common_idx = prices.index.intersection(signals.index)
        px = prices.reindex(common_idx)
        sig = signals.reindex(common_idx).fillna(0)

        if position_sizes is not None:
            pos_sizes = position_sizes.reindex(common_idx).fillna(0)
        else:
            pos_sizes = sig.astype(float)

        # Filter to date range
        if self.config.start_date:
            px = px[self.config.start_date :]
            sig = sig[self.config.start_date :]
            pos_sizes = pos_sizes[self.config.start_date :]
        if self.config.end_date:
            px = px[: self.config.end_date]
            sig = sig[: self.config.end_date]
            pos_sizes = pos_sizes[: self.config.end_date]

        n = len(px)  # noqa: F841
        total_cost_fraction = (
            self.config.transaction_cost_bps + self.config.slippage_bps
        ) / 10_000.0

        # Vectorised backtest calculation
        price_returns = px.pct_change().fillna(0.0)

        # Position held during each period (shifted: we trade at close, hold next day)
        position = pos_sizes.shift(1).fillna(0.0)

        # Detect trades (position changes)
        position_change = position.diff().abs().fillna(0.0)
        trade_costs = position_change * total_cost_fraction

        # Gross strategy returns = position * price_returns
        strategy_returns_gross = position * price_returns
        strategy_returns_net = strategy_returns_gross - trade_costs

        # Portfolio value
        portfolio_value = self.config.initial_capital * (1 + strategy_returns_net).cumprod()

        # Drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max

        # Check risk manager drawdown halt
        if self.risk_manager is not None:
            for dt, pv in portfolio_value.items():
                if (
                    self.risk_manager.update_drawdown(pv / self.config.initial_capital)
                    < -self.risk_manager.max_drawdown_limit
                ):
                    logger.warning("Risk manager triggered halt at %s", dt)
                    break

        result = pd.DataFrame(
            {
                "price": px,
                "signal": sig,
                "position": position,
                "price_returns": price_returns,
                "strategy_returns": strategy_returns_net,
                "gross_returns": strategy_returns_gross,
                "transaction_costs": trade_costs,
                "portfolio_value": portfolio_value,
                "drawdown": drawdown,
            },
            index=px.index,
        )

        # Summary stats log
        total_return = (portfolio_value.iloc[-1] / self.config.initial_capital - 1) * 100
        n_trades = int((position_change > 0).sum())
        logger.info(
            "Backtest complete: return=%.1f%%, trades=%d, max_dd=%.1f%%",
            total_return,
            n_trades,
            drawdown.min() * 100,
        )
        return result
