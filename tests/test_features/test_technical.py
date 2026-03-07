"""Tests for the TechnicalFeatures module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import TechnicalFeatures


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    close = 70.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)  # Ensure positive prices
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1000, 10000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.astype(float)},
        index=pd.date_range("2020-01-01", periods=n),
    )


class TestTechnicalFeatures:
    """Unit tests for TechnicalFeatures."""

    def test_add_all_returns_dataframe(self) -> None:
        """add_all() should return a DataFrame with more columns than input."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(120)
        result = tf.add_all(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(df.columns)

    def test_rsi_range(self) -> None:
        """RSI values should be in [0, 100]."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_rsi(df)
        assert "rsi" in result.columns
        rsi = result["rsi"].dropna()
        assert rsi.between(0, 100).all(), f"RSI out of range: {rsi.describe()}"

    def test_macd_columns_exist(self) -> None:
        """MACD calculation should add macd, macd_signal_line, macd_hist columns."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_macd(df)
        assert "macd" in result.columns
        assert "macd_signal_line" in result.columns
        assert "macd_hist" in result.columns

    def test_bollinger_bands_relationship(self) -> None:
        """Upper band should be >= middle >= lower band everywhere."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_bollinger_bands(df).dropna()
        assert (result["bb_upper"] >= result["bb_middle"]).all()
        assert (result["bb_middle"] >= result["bb_lower"]).all()

    def test_atr_positive(self) -> None:
        """ATR should be strictly positive."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_atr(df)
        atr = result["atr"].dropna()
        assert (atr > 0).all()

    def test_returns_correct_columns(self) -> None:
        """Return features for multiple horizons should be computed."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_returns(df)
        for n in [1, 2, 5]:
            assert f"return_{n}d" in result.columns
        assert "log_return_1d" in result.columns
        assert "realized_vol_20d" in result.columns

    def test_custom_ema_periods(self) -> None:
        """Custom EMA periods should be respected."""
        tf = TechnicalFeatures(ema_periods=[5, 15])
        df = _make_ohlcv(100)
        result = tf.add_moving_averages(df)
        assert "ema_5" in result.columns
        assert "ema_15" in result.columns
        assert "ema_8" not in result.columns  # Default, should not be present

    def test_original_columns_preserved(self) -> None:
        """Original OHLCV columns should be preserved in output."""
        tf = TechnicalFeatures()
        df = _make_ohlcv(100)
        result = tf.add_all(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns
