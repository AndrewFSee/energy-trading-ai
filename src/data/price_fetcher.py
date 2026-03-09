"""Price fetcher module using yfinance for energy market data.

Provides a wrapper around the yfinance library to download historical
and real-time price data for WTI crude oil, Brent crude oil, natural gas,
VIX, DXY, and S&P 500.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetches OHLCV price data for energy-related instruments.

    Wraps yfinance to provide a consistent interface for downloading
    price history for commodity futures, indices, and ETFs.

    Attributes:
        default_tickers: Mapping of instrument names to yfinance ticker symbols.
    """

    default_tickers: dict[str, str] = {
        "wti": "CL=F",
        "brent": "BZ=F",
        "natural_gas": "NG=F",
        "heating_oil": "HO=F",
        "rbob_gasoline": "RB=F",
        "vix": "^VIX",
        "ovx": "^OVX",  # CBOE Crude Oil Volatility Index (oil-specific fear)
        "dxy": "DX-Y.NYB",
        "sp500": "^GSPC",
        "xle": "XLE",
        "xop": "XOP",
        "gold": "GC=F",  # Gold futures — safe-haven/geopolitical proxy
    }

    def __init__(self, tickers: dict[str, str] | None = None) -> None:
        """Initialise the price fetcher.

        Args:
            tickers: Optional custom ticker mapping. If not provided,
                ``default_tickers`` is used.
        """
        self.tickers = tickers or self.default_tickers
        logger.info("PriceFetcher initialised with %d tickers", len(self.tickers))

    def fetch(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Download price data for one or more instruments.

        Args:
            symbols: List of instrument keys from ``self.tickers``.  If
                ``None``, all default tickers are fetched.
            start: Start date string in ``YYYY-MM-DD`` format.  Defaults to
                2 years ago.
            end: End date string in ``YYYY-MM-DD`` format.  Defaults to today.
            interval: Data frequency (e.g. ``"1d"``, ``"1h"``).
            auto_adjust: Whether to apply corporate-action adjustments.

        Returns:
            Dictionary mapping instrument name to a ``pd.DataFrame`` with
            columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

        Raises:
            ValueError: If a requested symbol is not in ``self.tickers``.
        """
        if symbols is None:
            symbols = list(self.tickers.keys())

        if start is None:
            start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        results: dict[str, pd.DataFrame] = {}
        for name in symbols:
            if name not in self.tickers:
                raise ValueError(f"Unknown symbol key '{name}'. Available: {list(self.tickers)}")
            ticker = self.tickers[name]
            logger.debug("Downloading %s (%s) from %s to %s", name, ticker, start, end)
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False,
                )
                if df.empty:
                    logger.warning("No data returned for %s (%s)", name, ticker)
                    continue
                df.index.name = "date"
                # Flatten multi-level columns that yfinance may return
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                results[name] = df
                logger.info("Downloaded %d rows for %s (%s)", len(df), name, ticker)
            except Exception as exc:
                logger.error("Failed to download %s: %s", name, exc)

        return results

    def fetch_single(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download price data for a single instrument.

        Args:
            symbol: Instrument key from ``self.tickers``.
            start: Start date string in ``YYYY-MM-DD`` format.
            end: End date string in ``YYYY-MM-DD`` format.
            interval: Data frequency.

        Returns:
            ``pd.DataFrame`` with OHLCV columns.
        """
        data = self.fetch(symbols=[symbol], start=start, end=end, interval=interval)
        return data.get(symbol, pd.DataFrame())

    def fetch_latest(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Fetch the most recent N days of daily data for a single instrument.

        Args:
            symbol: Instrument key from ``self.tickers``.
            days: Number of calendar days to look back.

        Returns:
            ``pd.DataFrame`` with recent OHLCV data.
        """
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.fetch_single(symbol, start=start)

    def get_returns(self, prices: pd.DataFrame, method: str = "log") -> pd.Series:
        """Compute price returns from a Close price series.

        Args:
            prices: DataFrame with a ``Close`` column.
            method: ``"log"`` for log returns, ``"pct"`` for percentage returns.

        Returns:
            ``pd.Series`` of returns.
        """
        close = prices["Close"]
        if method == "log":
            import numpy as np

            return (np.log(close / close.shift(1))).dropna()
        return close.pct_change().dropna()
