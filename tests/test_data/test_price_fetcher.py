"""Tests for the PriceFetcher data ingestion module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.price_fetcher import PriceFetcher


class TestPriceFetcher:
    """Unit tests for PriceFetcher."""

    def test_default_tickers_present(self) -> None:
        """Default ticker mapping should contain key energy instruments."""
        fetcher = PriceFetcher()
        assert "wti" in fetcher.tickers
        assert "natural_gas" in fetcher.tickers
        assert "brent" in fetcher.tickers
        assert "vix" in fetcher.tickers
        assert "dxy" in fetcher.tickers
        assert "sp500" in fetcher.tickers

    def test_custom_tickers(self) -> None:
        """Custom ticker mapping overrides defaults."""
        custom = {"test_ticker": "AAPL"}
        fetcher = PriceFetcher(tickers=custom)
        assert "test_ticker" in fetcher.tickers
        assert "wti" not in fetcher.tickers

    def test_invalid_symbol_raises(self) -> None:
        """Requesting an unknown symbol should raise ValueError."""
        fetcher = PriceFetcher()
        with pytest.raises(ValueError, match="Unknown symbol key"):
            fetcher.fetch(symbols=["non_existent_instrument"])

    def test_get_returns_log(self) -> None:
        """Log returns should be computed correctly."""
        import numpy as np

        fetcher = PriceFetcher()
        prices = pd.DataFrame(
            {"Close": [100.0, 110.0, 105.0, 115.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )
        returns = fetcher.get_returns(prices, method="log")
        assert len(returns) == 3
        # First return: log(110/100) ≈ 0.0953
        assert abs(returns.iloc[0] - np.log(110 / 100)) < 1e-6

    def test_get_returns_pct(self) -> None:
        """Percentage returns should be computed correctly."""
        fetcher = PriceFetcher()
        prices = pd.DataFrame(
            {"Close": [100.0, 110.0, 99.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        returns = fetcher.get_returns(prices, method="pct")
        assert len(returns) == 2
        assert abs(returns.iloc[0] - 0.10) < 1e-6

    def test_ticker_symbol_values(self) -> None:
        """Verify expected ticker symbol strings."""
        fetcher = PriceFetcher()
        assert fetcher.tickers["wti"] == "CL=F"
        assert fetcher.tickers["natural_gas"] == "NG=F"
        assert fetcher.tickers["brent"] == "BZ=F"
