"""Macroeconomic features for energy price modelling.

Combines macroeconomic data from FRED and market proxies (USD index,
equity indices, interest rates) with energy price data to create
cross-asset correlation and macro regime features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroFeatures:
    """Computes macro-economic and cross-asset features.

    Energy prices are strongly correlated with the USD, interest rate
    expectations, and global growth proxies (equities, PMI).  This class
    builds those cross-asset features for use in the ML model.

    Attributes:
        correlation_window: Rolling window (days) for computing correlations.
    """

    def __init__(self, correlation_window: int = 60) -> None:
        """Initialise macro features calculator.

        Args:
            correlation_window: Number of trading days for rolling correlation.
        """
        self.correlation_window = correlation_window

    def add_usd_features(
        self,
        df: pd.DataFrame,
        dxy: pd.Series,
    ) -> pd.DataFrame:
        """Add US Dollar Index (DXY) features.

        Oil is priced in USD — a stronger dollar makes oil more expensive
        for foreign buyers, typically exerting downward price pressure.

        Args:
            df: Price DataFrame indexed by date.
            dxy: DXY price series (indexed by date).

        Returns:
            DataFrame with DXY-based features appended.
        """
        df = df.copy()
        dxy = dxy.reindex(df.index).ffill()
        df["dxy"] = dxy
        df["dxy_return_5d"] = dxy.pct_change(5)
        df["dxy_return_20d"] = dxy.pct_change(20)
        df["dxy_zscore_60d"] = (dxy - dxy.rolling(60).mean()) / dxy.rolling(60).std()

        # Rolling correlation between energy price and DXY
        if "Close" in df.columns:
            energy_ret = np.log(df["Close"] / df["Close"].shift(1))
            dxy_ret = np.log(dxy / dxy.shift(1))
            df["energy_dxy_corr"] = energy_ret.rolling(self.correlation_window).corr(dxy_ret)
        logger.debug("Added USD features")
        return df

    def add_equity_features(
        self,
        df: pd.DataFrame,
        sp500: pd.Series,
        vix: pd.Series,
    ) -> pd.DataFrame:
        """Add S&P 500 and VIX features.

        Global risk appetite (proxied by equities and VIX) influences energy
        demand expectations and speculative positioning.

        Args:
            df: Price DataFrame indexed by date.
            sp500: S&P 500 closing price series.
            vix: VIX closing price series.

        Returns:
            DataFrame with equity and volatility features appended.
        """
        df = df.copy()
        sp = sp500.reindex(df.index).ffill()
        vx = vix.reindex(df.index).ffill()

        df["sp500_return_5d"] = sp.pct_change(5)
        df["sp500_return_20d"] = sp.pct_change(20)
        df["vix"] = vx
        df["vix_change_5d"] = vx.diff(5)
        df["vix_regime_high"] = (vx > 25).astype(int)
        df["vix_regime_extreme"] = (vx > 35).astype(int)

        if "Close" in df.columns:
            energy_ret = np.log(df["Close"] / df["Close"].shift(1))
            sp_ret = np.log(sp / sp.shift(1))
            df["energy_sp500_corr"] = energy_ret.rolling(self.correlation_window).corr(sp_ret)
        logger.debug("Added equity and VIX features")
        return df

    def add_rate_features(
        self,
        df: pd.DataFrame,
        fed_funds_rate: pd.Series | None = None,
        ten_year_yield: pd.Series | None = None,
        yield_spread: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Add interest rate and yield curve features.

        Interest rates affect energy demand through their impact on economic
        growth expectations and the cost of carrying commodity inventories.

        Args:
            df: Price DataFrame indexed by date.
            fed_funds_rate: Fed funds rate series.
            ten_year_yield: 10-year Treasury yield series.
            yield_spread: Yield curve spread (10y - 2y) series.

        Returns:
            DataFrame with rate-based features appended.
        """
        df = df.copy()
        rate_series: dict[str, pd.Series | None] = {
            "fed_funds_rate": fed_funds_rate,
            "ten_year_yield": ten_year_yield,
            "yield_spread_10y2y": yield_spread,
        }
        for name, series in rate_series.items():
            if series is not None:
                s = series.reindex(df.index).ffill()
                df[name] = s
                df[f"{name}_change_20d"] = s.diff(20)
                # Recession indicator: inverted yield curve
                if name == "yield_spread_10y2y":
                    df["yield_curve_inverted"] = (s < 0).astype(int)
        logger.debug("Added interest rate features")
        return df

    def add_macro_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a simple macro regime label based on available features.

        Uses VIX level and yield curve shape to classify the macro environment
        into four regimes: risk-on growth, risk-on stress, risk-off recession,
        risk-off crisis.

        Args:
            df: DataFrame with ``vix``, ``yield_spread_10y2y`` columns.

        Returns:
            DataFrame with ``macro_regime`` integer column appended (0-3).
        """
        df = df.copy()
        if "vix" not in df.columns:
            logger.warning("VIX feature not found — skipping macro regime")
            return df

        conditions = [
            (df["vix"] <= 20),  # low vol = risk-on
            (df["vix"] <= 30),  # moderate vol
            (df["vix"] <= 40),  # elevated vol = risk-off
        ]
        regime = np.select(conditions, [0, 1, 2], default=3)
        df["macro_regime"] = regime.astype(int)
        logger.debug("Added macro regime feature")
        return df
