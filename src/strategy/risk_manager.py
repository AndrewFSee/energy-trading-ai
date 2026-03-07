"""Risk management module.

Enforces risk limits at the position and portfolio level:
- Maximum position size per instrument
- Maximum portfolio drawdown halt
- Value at Risk (VaR) monitoring
- Stop-loss and take-profit management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskReport:
    """Summary of current portfolio risk metrics.

    Attributes:
        current_drawdown: Current drawdown from peak (negative float).
        var_95: 95% historical Value at Risk (daily).
        var_99: 99% historical Value at Risk (daily).
        current_leverage: Current gross leverage.
        positions_at_limit: List of positions at the size limit.
        trading_halted: Whether risk limits have been breached.
        halt_reason: Reason for trading halt (if applicable).
    """

    current_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    current_leverage: float = 0.0
    positions_at_limit: list[str] = field(default_factory=list)
    trading_halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """Enforces risk management rules for the energy trading portfolio.

    Monitors drawdown, VaR, and position concentration limits.
    Can automatically halt trading when risk thresholds are breached.

    Attributes:
        max_position_size: Max position as fraction of portfolio.
        max_drawdown_limit: Max portfolio drawdown before halt.
        var_confidence: Confidence level for VaR calculation.
        var_lookback_days: Historical window for VaR.
        stop_loss_atr_multiple: ATR multiples for stop-loss level.
        take_profit_atr_multiple: ATR multiples for take-profit level.
        max_leverage: Maximum gross portfolio leverage.
    """

    def __init__(
        self,
        max_position_size: float = 0.20,
        max_drawdown_limit: float = 0.15,
        var_confidence: float = 0.99,
        var_lookback_days: int = 252,
        stop_loss_atr_multiple: float = 2.0,
        take_profit_atr_multiple: float = 4.0,
        max_leverage: float = 1.5,
    ) -> None:
        """Initialise the risk manager.

        Args:
            max_position_size: Maximum position size (fraction of capital).
            max_drawdown_limit: Portfolio drawdown level that halts trading.
            var_confidence: VaR confidence level (e.g. 0.99 = 99%).
            var_lookback_days: Rolling window for historical VaR.
            stop_loss_atr_multiple: Stop-loss placed at entry ± N * ATR.
            take_profit_atr_multiple: Take-profit at entry ± N * ATR.
            max_leverage: Maximum gross portfolio leverage.
        """
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        self.var_lookback_days = var_lookback_days
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.take_profit_atr_multiple = take_profit_atr_multiple
        self.max_leverage = max_leverage
        self._peak_value: float = 1.0
        self._trading_halted = False
        self._halt_reason = ""
        logger.info(
            "RiskManager initialised (max_dd=%.1f%%, max_pos=%.1f%%)",
            max_drawdown_limit * 100,
            max_position_size * 100,
        )

    def check_position_size(
        self,
        proposed_size: float,
        instrument: str = "unknown",
    ) -> float:
        """Enforce the maximum position size limit.

        Args:
            proposed_size: Proposed position as fraction of portfolio.
            instrument: Instrument identifier (for logging).

        Returns:
            Capped position size.
        """
        capped = np.clip(proposed_size, -self.max_position_size, self.max_position_size)
        if abs(capped) < abs(proposed_size):
            logger.warning(
                "Position capped for %s: %.3f → %.3f (max=%.3f)",
                instrument,
                proposed_size,
                capped,
                self.max_position_size,
            )
        return float(capped)

    def update_drawdown(self, portfolio_value: float) -> float:
        """Update peak value and compute current drawdown.

        Args:
            portfolio_value: Current portfolio value.

        Returns:
            Current drawdown (negative float, e.g. -0.05 = -5%).
        """
        self._peak_value = max(self._peak_value, portfolio_value)
        drawdown = (portfolio_value - self._peak_value) / self._peak_value
        if drawdown < -self.max_drawdown_limit and not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = (
                f"Max drawdown limit reached: {drawdown:.1%} < -{self.max_drawdown_limit:.1%}"
            )
            logger.warning("TRADING HALTED: %s", self._halt_reason)
        return drawdown

    def is_trading_halted(self) -> bool:
        """Check if trading has been halted due to risk breach.

        Returns:
            ``True`` if trading is currently halted.
        """
        return self._trading_halted

    def reset_halt(self) -> None:
        """Manually reset the trading halt flag (e.g. after review)."""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading halt reset manually")

    def compute_var(
        self,
        returns: pd.Series,
        confidence: float | None = None,
        lookback: int | None = None,
    ) -> float:
        """Compute historical Value at Risk.

        Args:
            returns: Daily return series.
            confidence: VaR confidence level (overrides instance default).
            lookback: Lookback window in days (overrides instance default).

        Returns:
            VaR value (positive number representing the maximum expected loss
            at the given confidence level over a 1-day horizon).
        """
        conf = confidence if confidence is not None else self.var_confidence
        lb = lookback if lookback is not None else self.var_lookback_days
        recent_returns = returns.dropna().tail(lb)
        if len(recent_returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0
        var = float(np.percentile(recent_returns, (1 - conf) * 100))
        return abs(var)  # Return as positive loss amount

    def compute_stop_loss(self, entry_price: float, atr: float, direction: int) -> float:
        """Compute ATR-based stop-loss price.

        Args:
            entry_price: Price at which the position was entered.
            atr: Average True Range at entry.
            direction: Position direction (+1 for long, -1 for short).

        Returns:
            Stop-loss price level.
        """
        return entry_price - direction * self.stop_loss_atr_multiple * atr

    def compute_take_profit(self, entry_price: float, atr: float, direction: int) -> float:
        """Compute ATR-based take-profit price.

        Args:
            entry_price: Price at which the position was entered.
            atr: Average True Range at entry.
            direction: Position direction (+1 for long, -1 for short).

        Returns:
            Take-profit price level.
        """
        return entry_price + direction * self.take_profit_atr_multiple * atr

    def generate_report(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: dict[str, float] | None = None,
    ) -> RiskReport:
        """Generate a comprehensive risk report.

        Args:
            portfolio_value: Current total portfolio value.
            returns: Historical daily returns series.
            positions: Optional dictionary of {instrument: position_size}.

        Returns:
            ``RiskReport`` dataclass with current risk metrics.
        """
        drawdown = self.update_drawdown(portfolio_value)
        var_95 = self.compute_var(returns, confidence=0.95)
        var_99 = self.compute_var(returns, confidence=0.99)

        current_leverage = 0.0
        positions_at_limit = []
        if positions:
            current_leverage = sum(abs(v) for v in positions.values())
            positions_at_limit = [
                k for k, v in positions.items() if abs(v) >= self.max_position_size
            ]

        return RiskReport(
            current_drawdown=drawdown,
            var_95=var_95,
            var_99=var_99,
            current_leverage=current_leverage,
            positions_at_limit=positions_at_limit,
            trading_halted=self._trading_halted,
            halt_reason=self._halt_reason,
        )
