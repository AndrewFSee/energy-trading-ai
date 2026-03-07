"""Position sizing algorithms for energy trading strategies.

Implements three position sizing methods:
- Volatility targeting: scales position so portfolio vol = target
- Kelly criterion: theoretically optimal fraction from edge/odds ratio
- Fixed fractional: simple constant fraction of capital per trade
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSizer:
    """Computes position sizes based on risk management rules.

    All methods return position sizes as a fraction of total portfolio
    capital (e.g. 0.15 = 15% of capital allocated to this position).

    Attributes:
        method: Sizing method (``"volatility_targeting"``, ``"kelly"``,
            or ``"fixed_fractional"``).
        target_volatility: Target annualised portfolio volatility.
        max_position_size: Hard cap on any single position.
        kelly_fraction: Fraction of full Kelly bet to use.
        fixed_fraction: Fixed fraction for ``"fixed_fractional"`` method.
    """

    METHODS = {"volatility_targeting", "kelly", "fixed_fractional"}

    def __init__(
        self,
        method: str = "volatility_targeting",
        target_volatility: float = 0.15,
        max_position_size: float = 0.20,
        kelly_fraction: float = 0.25,
        fixed_fraction: float = 0.02,
    ) -> None:
        """Initialise the position sizer.

        Args:
            method: Sizing algorithm to use.
            target_volatility: Annualised target volatility (e.g. 0.15 = 15%).
            max_position_size: Maximum fraction of capital in any single position.
            kelly_fraction: Multiplier on full Kelly (use <1 for fractional Kelly).
            fixed_fraction: Constant fraction of capital per trade.

        Raises:
            ValueError: If an unknown sizing method is specified.
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")
        self.method = method
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.fixed_fraction = fixed_fraction
        logger.info("PositionSizer initialised (method=%s)", method)

    def size(
        self,
        signal: float,
        price_volatility: float | None = None,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
    ) -> float:
        """Compute position size for a given signal.

        Args:
            signal: Trading signal in [-1, +1].
            price_volatility: Annualised return volatility of the instrument.
                Required for ``"volatility_targeting"`` method.
            win_rate: Historical win rate [0, 1] for ``"kelly"`` method.
            avg_win_loss_ratio: Average win / average loss ratio for Kelly.

        Returns:
            Position size as a fraction of capital in [-max, +max].
        """
        if signal == 0:
            return 0.0

        if self.method == "volatility_targeting":
            raw_size = self._volatility_targeting(signal, price_volatility)
        elif self.method == "kelly":
            raw_size = self._kelly(signal, win_rate, avg_win_loss_ratio)
        else:  # fixed_fractional
            raw_size = np.sign(signal) * self.fixed_fraction

        # Apply hard cap
        capped = np.clip(raw_size, -self.max_position_size, self.max_position_size)
        return float(capped)

    def _volatility_targeting(self, signal: float, price_volatility: float | None) -> float:
        """Compute position size using volatility targeting.

        Scales the position such that if the instrument had volatility
        ``price_volatility``, the contribution to portfolio vol would be
        ``target_volatility * |signal|``.

        Args:
            signal: Directional signal [-1, +1].
            price_volatility: Annualised return vol. Defaults to 0.30 if None.

        Returns:
            Signed position size fraction.
        """
        vol = price_volatility if price_volatility and price_volatility > 0 else 0.30
        size = (self.target_volatility / vol) * np.sign(signal)
        return float(size)

    def _kelly(
        self,
        signal: float,
        win_rate: float | None,
        avg_win_loss_ratio: float | None,
    ) -> float:
        """Compute fractional Kelly position size.

        Kelly fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio

        Args:
            signal: Directional signal [-1, +1].
            win_rate: Historical probability of profitable trade.
            avg_win_loss_ratio: Average win / average loss.

        Returns:
            Signed position size fraction.
        """
        wr = win_rate if win_rate is not None else 0.55
        b = avg_win_loss_ratio if avg_win_loss_ratio is not None else 1.5
        kelly = (wr * b - (1 - wr)) / b
        kelly = max(0.0, kelly)  # Never go negative (Kelly says don't trade)
        fractional_kelly = kelly * self.kelly_fraction
        return float(np.sign(signal) * fractional_kelly)

    def size_series(
        self,
        signals: pd.Series,
        returns: pd.Series | None = None,
        vol_lookback: int = 20,
    ) -> pd.Series:
        """Compute position sizes for a series of signals.

        Args:
            signals: Series of trading signals.
            returns: Price return series for volatility estimation.
            vol_lookback: Rolling window for volatility estimation.

        Returns:
            Series of position sizes.
        """
        if returns is not None and self.method == "volatility_targeting":
            rolling_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
            sizes = []
            for sig, vol in zip(signals, rolling_vol.reindex(signals.index), strict=False):
                sizes.append(
                    self.size(sig, price_volatility=float(vol) if not np.isnan(vol) else None)
                )
        else:
            sizes = [self.size(float(sig)) for sig in signals]

        return pd.Series(sizes, index=signals.index, name="position_size")
