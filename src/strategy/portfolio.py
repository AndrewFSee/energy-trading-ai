"""Portfolio-level management for multi-instrument energy trading.

Manages a portfolio of energy futures positions, tracks P&L, computes
portfolio-level risk metrics, and handles capital allocation across
correlated instruments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position.

    Attributes:
        instrument: Instrument identifier.
        direction: +1 for long, -1 for short, 0 for flat.
        size: Position size as fraction of portfolio capital.
        entry_price: Price at which the position was opened.
        entry_time: Timestamp of position entry.
        stop_loss: Stop-loss price level.
        take_profit: Take-profit price level.
        unrealised_pnl: Current mark-to-market P&L.
    """

    instrument: str
    direction: int
    size: float
    entry_price: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealised_pnl: float = 0.0

    def update_pnl(self, current_price: float) -> float:
        """Update unrealised P&L for the current price.

        Args:
            current_price: Current market price.

        Returns:
            Updated unrealised P&L.
        """
        self.unrealised_pnl = (
            self.direction * self.size * ((current_price - self.entry_price) / self.entry_price)
        )
        return self.unrealised_pnl


class Portfolio:
    """Portfolio manager for multi-instrument energy trading.

    Tracks positions, cash, P&L history, and provides portfolio-level
    metrics for the risk management and reporting layers.

    Attributes:
        initial_capital: Starting portfolio value in USD.
        capital: Current available cash.
        positions: Dictionary of open positions keyed by instrument.
        closed_trades: Log of completed trades.
        equity_curve: Daily portfolio value history.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        instruments: list[str] | None = None,
        max_instruments: int = 5,
    ) -> None:
        """Initialise the portfolio.

        Args:
            initial_capital: Starting capital in USD.
            instruments: List of tradeable instruments.
            max_instruments: Maximum number of concurrent open positions.
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.instruments = instruments or []
        self.max_instruments = max_instruments
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.equity_curve: list[dict] = []
        logger.info(
            "Portfolio initialised (capital=$%.0f, max_instruments=%d)",
            initial_capital,
            max_instruments,
        )

    @property
    def total_value(self) -> float:
        """Compute total portfolio value including unrealised P&L."""
        unrealised = sum(p.unrealised_pnl * self.capital for p in self.positions.values())
        return self.capital + unrealised

    @property
    def gross_leverage(self) -> float:
        """Compute current gross leverage."""
        return sum(abs(p.size) for p in self.positions.values())

    def open_position(
        self,
        instrument: str,
        direction: int,
        size: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        timestamp: datetime | None = None,
    ) -> Position | None:
        """Open a new position.

        Args:
            instrument: Instrument identifier.
            direction: +1 (long) or -1 (short).
            size: Position size as fraction of portfolio capital.
            price: Entry price.
            stop_loss: Stop-loss price level.
            take_profit: Take-profit price level.
            timestamp: Trade timestamp (defaults to now).

        Returns:
            Opened ``Position``, or ``None`` if portfolio limit is reached.
        """
        if len(self.positions) >= self.max_instruments:
            logger.warning(
                "Max positions (%d) reached — cannot open new position", self.max_instruments
            )
            return None

        if instrument in self.positions:
            logger.info("Closing existing position before reopening %s", instrument)
            self.close_position(instrument, price, timestamp)

        pos = Position(
            instrument=instrument,
            direction=direction,
            size=size,
            entry_price=price,
            entry_time=timestamp or datetime.utcnow(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions[instrument] = pos
        logger.info(
            "Opened %s position in %s: size=%.3f, entry=%.2f",
            "LONG" if direction == 1 else "SHORT",
            instrument,
            size,
            price,
        )
        return pos

    def close_position(
        self,
        instrument: str,
        exit_price: float,
        timestamp: datetime | None = None,
    ) -> dict | None:
        """Close an open position and record the trade.

        Args:
            instrument: Instrument identifier.
            exit_price: Price at which to close.
            timestamp: Close timestamp.

        Returns:
            Trade log dictionary, or ``None`` if no open position.
        """
        if instrument not in self.positions:
            logger.warning("No open position to close for %s", instrument)
            return None

        pos = self.positions.pop(instrument)
        pnl_pct = pos.direction * ((exit_price - pos.entry_price) / pos.entry_price)
        pnl_dollars = pnl_pct * pos.size * self.capital

        trade = {
            "instrument": instrument,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "size": pos.size,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_dollars": pnl_dollars,
            "entry_time": pos.entry_time,
            "exit_time": timestamp or datetime.utcnow(),
        }
        self.closed_trades.append(trade)
        self.capital += pnl_dollars
        logger.info(
            "Closed %s %s: P&L=%.2f%% ($%.0f)",
            trade["direction"],
            instrument,
            pnl_pct * 100,
            pnl_dollars,
        )
        return trade

    def update_prices(self, prices: dict[str, float]) -> float:
        """Update unrealised P&L for all open positions.

        Args:
            prices: Dictionary of {instrument: current_price}.

        Returns:
            Total portfolio value after update.
        """
        for instrument, pos in self.positions.items():
            if instrument in prices:
                pos.update_pnl(prices[instrument])
        return self.total_value

    def record_equity(self, timestamp: datetime | None = None) -> None:
        """Record the current portfolio value in the equity curve.

        Args:
            timestamp: Timestamp for this equity recording.
        """
        self.equity_curve.append(
            {
                "timestamp": timestamp or datetime.utcnow(),
                "portfolio_value": self.total_value,
                "capital": self.capital,
                "n_positions": len(self.positions),
            }
        )

    def get_equity_series(self) -> pd.DataFrame:
        """Return the equity curve as a DataFrame.

        Returns:
            DataFrame with ``timestamp``, ``portfolio_value``, ``capital``,
            and ``n_positions`` columns.
        """
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")

    def get_trade_log(self) -> pd.DataFrame:
        """Return all closed trades as a DataFrame.

        Returns:
            DataFrame of closed trades with P&L statistics.
        """
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_trades)
