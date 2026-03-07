"""Technical indicators for energy price time series.

Computes a comprehensive set of technical analysis indicators using the
``ta`` library plus custom implementations.  Indicators include momentum,
trend, volatility, and volume features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import ta

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "'ta' library not installed — using fallback implementations"
    )

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Computes technical indicators from OHLCV price data.

    All methods accept a ``pd.DataFrame`` with columns
    ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` and return a
    DataFrame with additional indicator columns appended.

    Attributes:
        rsi_period: RSI lookback window.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal: MACD signal line period.
        bollinger_period: Bollinger Band window.
        bollinger_std: Bollinger Band standard deviation multiplier.
        atr_period: ATR lookback window.
        ema_periods: List of EMA periods to compute.
        sma_periods: List of SMA periods to compute.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        atr_period: int = 14,
        ema_periods: list[int] | None = None,
        sma_periods: list[int] | None = None,
    ) -> None:
        """Initialise with indicator parameters.

        Args:
            rsi_period: RSI calculation window.
            macd_fast: Fast EMA period for MACD.
            macd_slow: Slow EMA period for MACD.
            macd_signal: Signal line EMA period for MACD.
            bollinger_period: Window for Bollinger Bands.
            bollinger_std: Standard deviation multiplier for Bollinger Bands.
            atr_period: ATR calculation window.
            ema_periods: List of periods for EMA features.
            sma_periods: List of periods for SMA features.
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.atr_period = atr_period
        self.ema_periods = ema_periods or [8, 21, 50, 200]
        self.sma_periods = sma_periods or [10, 20, 50]

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append all technical indicators.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with all technical indicator columns appended.
        """
        df = df.copy()
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_stochastic(df)
        df = self.add_returns(df)
        df = self.add_volume_features(df)
        logger.info("Added %d technical features", len(df.columns) - 5)
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and SMA features.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with EMA and SMA columns.
        """
        df = df.copy()
        close = df["Close"]
        for p in self.ema_periods:
            df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
            df[f"ema_{p}_dist"] = (close - df[f"ema_{p}"]) / df[f"ema_{p}"]
        for p in self.sma_periods:
            df[f"sma_{p}"] = close.rolling(p).mean()
            df[f"sma_{p}_dist"] = (close - df[f"sma_{p}"]) / df[f"sma_{p}"]
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index (RSI).

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ``rsi`` column.
        """
        df = df.copy()
        if TA_AVAILABLE:
            df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=self.rsi_period).rsi()
        else:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD, MACD signal, and MACD histogram.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ``macd``, ``macd_signal``, ``macd_hist`` columns.
        """
        df = df.copy()
        if TA_AVAILABLE:
            macd_ind = ta.trend.MACD(
                close=df["Close"],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal,
            )
            df["macd"] = macd_ind.macd()
            df["macd_signal_line"] = macd_ind.macd_signal()
            df["macd_hist"] = macd_ind.macd_diff()
        else:
            ema_fast = df["Close"].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=self.macd_slow, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal_line"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal_line"]
        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands (upper, lower, middle) and %B indicator.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with Bollinger Band columns.
        """
        df = df.copy()
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(
                close=df["Close"],
                window=self.bollinger_period,
                window_dev=self.bollinger_std,
            )
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_pct"] = bb.bollinger_pband()
            df["bb_width"] = bb.bollinger_wband()
        else:
            close = df["Close"]
            sma = close.rolling(self.bollinger_period).mean()
            std = close.rolling(self.bollinger_period).std()
            df["bb_middle"] = sma
            df["bb_upper"] = sma + self.bollinger_std * std
            df["bb_lower"] = sma - self.bollinger_std * std
            df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range (ATR) volatility indicator.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ``atr`` and ``atr_pct`` columns.
        """
        df = df.copy()
        if TA_AVAILABLE:
            atr_series = ta.volatility.AverageTrueRange(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                window=self.atr_period,
            ).average_true_range()
            # Some versions of the ta library return 0 for the warmup period
            # instead of NaN.  Replace those zeros with NaN for consistency.
            df["atr"] = atr_series.replace(0, np.nan)
        else:
            hl = df["High"] - df["Low"]
            hc = (df["High"] - df["Close"].shift(1)).abs()
            lc = (df["Low"] - df["Close"].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr"] = tr.rolling(self.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["Close"]
        return df

    def add_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator (%K and %D).

        Args:
            df: OHLCV DataFrame.
            k_period: Lookback period for %K.
            d_period: Smoothing period for %D.

        Returns:
            DataFrame with ``stoch_k`` and ``stoch_d`` columns.
        """
        df = df.copy()
        if TA_AVAILABLE:
            stoch = ta.momentum.StochasticOscillator(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                window=k_period,
                smooth_window=d_period,
            )
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
        else:
            low_min = df["Low"].rolling(k_period).min()
            high_max = df["High"].rolling(k_period).max()
            df["stoch_k"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price return features at multiple horizons.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with return features.
        """
        df = df.copy()
        close = df["Close"]
        for n in [1, 2, 5, 10, 20]:
            df[f"return_{n}d"] = close.pct_change(n)
        df["log_return_1d"] = np.log(close / close.shift(1))
        df["realized_vol_20d"] = df["log_return_1d"].rolling(20).std() * np.sqrt(252)
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features including OBV.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with volume feature columns.
        """
        df = df.copy()
        if "Volume" not in df.columns:
            return df
        vol = df["Volume"].replace(0, np.nan)
        df["volume_sma_20"] = vol.rolling(20).mean()
        df["volume_ratio"] = vol / df["volume_sma_20"]
        # On-Balance Volume
        direction = np.sign(df["Close"].diff()).fillna(0)
        df["obv"] = (direction * vol.fillna(0)).cumsum()
        return df
