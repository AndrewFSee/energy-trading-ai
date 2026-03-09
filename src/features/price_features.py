"""Two-stage price feature engineering: demand forecast → price prediction.

Builds features for natural gas price direction / level prediction using
a demand-augmented approach:

  Stage 1: Weather → Demand forecast (from trained XGB load model)
  Stage 2: Demand forecast + fundamentals → Price features

The key hypothesis: predicted power demand is a causal driver of gas-for-power
demand, which constitutes ~40% of US natural gas consumption.  By including
demand forecasts as features, we expect to improve price prediction beyond
what is achievable from technical/calendar features alone.

This module produces TWO feature sets for ablation:
  - **baseline**: Technical + calendar + fundamental features (no demand)
  - **enhanced**: Same + demand forecast features

Comparing these measures the marginal information content of demand forecasts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PriceFeatureBuilder:
    """Build price prediction features with optional demand augmentation.

    The builder merges natural gas price data with weather, demand forecasts,
    and NG storage fundamentals to create a rich feature set.

    Attributes:
        horizons: Forecast horizons in trading days.
        technical_windows: Windows for technical indicators.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        technical_windows: list[int] | None = None,
    ) -> None:
        self.horizons = horizons or [1, 5]
        self.technical_windows = technical_windows or [5, 10, 20, 50]
        logger.info(
            "PriceFeatureBuilder (horizons=%s, windows=%s)",
            self.horizons, self.technical_windows,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def build(
        self,
        ng_prices: pd.DataFrame,
        weather_df: pd.DataFrame | None = None,
        demand_df: pd.DataFrame | None = None,
        demand_forecast: pd.Series | None = None,
        ng_storage: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build baseline and demand-enhanced feature sets.

        Args:
            ng_prices: NG futures OHLCV from PriceFetcher (index=date).
            weather_df: Daily weather from OpenMeteoClient.
            demand_df: Daily demand from EIADemandClient (actual values).
            demand_forecast: Series of predicted demand (from XGB backcast).
            ng_storage: Weekly NG storage from EIAClient.

        Returns:
            Dict with keys ``"baseline"`` and ``"enhanced"`` mapping to
            DataFrames. Each has feature columns + target columns
            (``target_ret_1d``, ``target_dir_1d``, etc.).
        """
        # Normalise price index
        df = ng_prices[["Close", "Open", "High", "Low", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()

        # Returns
        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["ret_1d"] = df["Close"].pct_change(1)

        # Technical features (always available)
        df = self._add_technical(df)
        df = self._add_calendar(df)

        # Fundamental features: NG storage
        if ng_storage is not None:
            df = self._add_storage(df, ng_storage)

        # Weather features (shared between baseline and enhanced)
        if weather_df is not None:
            df = self._add_weather_for_price(df, weather_df)

        # --- Build targets ---
        for h in self.horizons:
            df[f"target_ret_{h}d"] = df["Close"].pct_change(h).shift(-h)
            df[f"target_dir_{h}d"] = (df[f"target_ret_{h}d"] > 0).astype(int)

        # Baseline: everything except demand
        baseline_cols = [
            c for c in df.columns
            if not c.startswith("target_") and c not in ("Close", "Open", "High", "Low", "Volume", "log_ret")
        ]
        target_cols = [c for c in df.columns if c.startswith("target_")]

        baseline = df[baseline_cols + target_cols].copy()

        # Enhanced: add demand features
        enhanced = baseline.copy()
        if demand_forecast is not None and demand_df is not None:
            demand_feats = self._build_demand_features(
                enhanced, demand_df, demand_forecast,
            )
            enhanced = enhanced.join(demand_feats, how="left")
        elif demand_forecast is not None:
            # Even without actual demand, use forecast alone
            demand_feats = self._build_forecast_only_features(enhanced, demand_forecast)
            enhanced = enhanced.join(demand_feats, how="left")

        # Drop rows with NaN targets
        for key, feat_df in [("baseline", baseline), ("enhanced", enhanced)]:
            feat_df.dropna(subset=[f"target_dir_{self.horizons[0]}d"], inplace=True)

        logger.info(
            "Price features — baseline: %d rows × %d features, "
            "enhanced: %d rows × %d features",
            len(baseline),
            len([c for c in baseline.columns if not c.startswith("target_")]),
            len(enhanced),
            len([c for c in enhanced.columns if not c.startswith("target_")]),
        )

        return {"baseline": baseline, "enhanced": enhanced}

    # ------------------------------------------------------------------ #
    #  Technical Features
    # ------------------------------------------------------------------ #
    def _add_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical / price-derived features."""
        close = df["Close"]

        # Returns at multiple horizons
        for w in [1, 2, 3, 5, 10, 20]:
            df[f"ret_{w}d"] = close.pct_change(w)

        # Moving averages and crossovers
        for w in self.technical_windows:
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"sma_ratio_{w}"] = close / df[f"sma_{w}"]

        # EMA
        for w in [5, 12, 26]:
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df["ema_cross_12_26"] = df["ema_12"] - df["ema_26"]

        # Volatility
        for w in [5, 10, 20]:
            df[f"volatility_{w}d"] = df["log_ret"].rolling(w).std() * np.sqrt(252)

        # Bollinger bands
        bb_mean = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = bb_mean + 2 * bb_std
        df["bb_lower"] = bb_mean - 2 * bb_std
        df["bb_pct"] = (close - bb_mean) / (2 * bb_std)

        # RSI (14-day)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Volume features
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            df["volume_sma_20"] = df["Volume"].rolling(20).mean()
            df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + 1)
            df["volume_change"] = df["Volume"].pct_change()

        # Range / candle features
        df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["body_pct"] = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-10)

        # Momentum
        for w in [5, 10, 20]:
            df[f"momentum_{w}d"] = close / close.shift(w) - 1

        logger.debug("Added %d technical features", 40)
        return df

    # ------------------------------------------------------------------ #
    #  Calendar Features
    # ------------------------------------------------------------------ #
    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar features relevant to energy pricing."""
        idx = df.index
        df["day_of_week"] = idx.dayofweek
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["is_monday"] = (idx.dayofweek == 0).astype(int)
        df["is_friday"] = (idx.dayofweek == 4).astype(int)

        # Seasonal encoding
        doy = idx.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Season flags
        df["is_winter"] = idx.month.isin([12, 1, 2]).astype(int)
        df["is_summer"] = idx.month.isin([6, 7, 8]).astype(int)

        # Heating / injection season
        df["heating_season"] = idx.month.isin([11, 12, 1, 2, 3]).astype(int)
        df["injection_season"] = idx.month.isin([4, 5, 6, 7, 8, 9, 10]).astype(int)

        logger.debug("Added calendar features")
        return df

    # ------------------------------------------------------------------ #
    #  NG Storage Fundamental Features
    # ------------------------------------------------------------------ #
    def _add_storage(self, df: pd.DataFrame, storage: pd.DataFrame) -> pd.DataFrame:
        """Add natural gas storage features (weekly, forward-filled to daily)."""
        storage = storage.copy()
        storage.index = pd.to_datetime(storage.index)

        # Find the value column
        val_col = [c for c in storage.columns if c not in ("date",)][0]
        stor = storage[[val_col]].rename(columns={val_col: "ng_storage"})

        # Compute changes before resampling
        stor["ng_storage_change"] = stor["ng_storage"].diff()
        stor["ng_storage_change_4w"] = stor["ng_storage"].diff(4)

        # Rolling expectation for surprise
        stor["ng_expected_change"] = stor["ng_storage_change"].rolling(4).mean()
        stor["ng_storage_surprise"] = stor["ng_storage_change"] - stor["ng_expected_change"]

        # Year-over-year storage
        stor["ng_storage_yoy"] = stor["ng_storage"].pct_change(52)

        # Resample to daily (forward-fill weekly data)
        stor = stor.resample("D").ffill()

        # Join to price df
        df = df.join(stor, how="left")
        df[stor.columns] = df[stor.columns].ffill()

        logger.debug("Added NG storage features")
        return df

    # ------------------------------------------------------------------ #
    #  Weather → Price Features
    # ------------------------------------------------------------------ #
    def _add_weather_for_price(self, df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Add weather features relevant to gas pricing."""
        weather = weather.copy()
        weather.index = pd.to_datetime(weather.index)

        # Select key weather columns for price impact
        cols = []
        for c in ["tavg", "hdd", "cdd", "hdd_7d", "cdd_7d", "temp_range"]:
            if c in weather.columns:
                cols.append(c)

        if not cols:
            return df

        wdf = weather[cols].copy()
        # Prefix to avoid collision
        wdf.columns = [f"wx_{c}" for c in wdf.columns]

        df = df.join(wdf, how="left")
        df[wdf.columns] = df[wdf.columns].ffill()

        # Heating/cooling degree day extremes (gas demand drivers)
        if "wx_hdd" in df.columns:
            df["wx_hdd_extreme"] = (df["wx_hdd"] > df["wx_hdd"].rolling(90).quantile(0.9)).astype(int)
        if "wx_cdd" in df.columns:
            df["wx_cdd_extreme"] = (df["wx_cdd"] > df["wx_cdd"].rolling(90).quantile(0.9)).astype(int)

        logger.debug("Added weather-price features")
        return df

    # ------------------------------------------------------------------ #
    #  Demand Forecast Features (Enhanced only)
    # ------------------------------------------------------------------ #
    def _build_demand_features(
        self,
        df: pd.DataFrame,
        actual_demand: pd.DataFrame,
        forecast: pd.Series,
    ) -> pd.DataFrame:
        """Build demand-derived features for price prediction.

        This is the key value-add of the two-stage approach: using demand
        forecasts as a causal input to price prediction.

        Args:
            df: Price DataFrame (for date alignment).
            actual_demand: Daily actual demand data.
            forecast: Demand forecasts from XGB backcast.

        Returns:
            DataFrame with demand features, indexed like df.
        """
        # Align on dates
        actual = actual_demand.copy()
        actual.index = pd.to_datetime(actual.index)

        dem = pd.DataFrame(index=df.index)

        # -- Demand forecast level --
        fc = forecast.copy()
        fc.index = pd.to_datetime(fc.index)
        dem["demand_forecast"] = fc.reindex(df.index, method="ffill")

        # -- Actual demand (lagged by 1 to avoid look-ahead) --
        target_col = "east_total_mwh"
        if target_col in actual.columns:
            actual_vals = actual[target_col].reindex(df.index, method="ffill")
            dem["demand_actual_lag1"] = actual_vals.shift(1)

            # Demand surprise: forecast - lagged actual
            dem["demand_surprise"] = dem["demand_forecast"] - dem["demand_actual_lag1"]
            dem["demand_surprise_pct"] = dem["demand_surprise"] / (dem["demand_actual_lag1"] + 1)
        else:
            dem["demand_actual_lag1"] = np.nan
            dem["demand_surprise"] = np.nan
            dem["demand_surprise_pct"] = np.nan

        # -- Demand momentum --
        dem["demand_fc_change_1d"] = dem["demand_forecast"].pct_change(1)
        dem["demand_fc_change_5d"] = dem["demand_forecast"].pct_change(5)
        dem["demand_fc_change_20d"] = dem["demand_forecast"].pct_change(20)

        # -- Demand level relative to seasonal norm --
        dem["demand_fc_roll_30d"] = dem["demand_forecast"].rolling(30).mean()
        dem["demand_fc_deviation"] = (
            dem["demand_forecast"] - dem["demand_fc_roll_30d"]
        ) / (dem["demand_fc_roll_30d"] + 1)

        # -- Demand volatility (uncertainty = price risk) --
        dem["demand_fc_vol_7d"] = dem["demand_forecast"].rolling(7).std()
        dem["demand_fc_vol_20d"] = dem["demand_forecast"].rolling(20).std()

        # -- Demand percentile (where does today's demand sit historically?) --
        dem["demand_fc_percentile"] = dem["demand_forecast"].rolling(252).rank(pct=True)

        # -- Demand × storage interaction --
        if "ng_storage" in df.columns:
            # High demand + low storage = very bullish
            dem["demand_x_storage"] = (
                dem["demand_fc_deviation"] * (-df["ng_storage_surprise"].fillna(0))
            )
            # Normalised demand/storage ratio
            fc_z = (dem["demand_forecast"] - dem["demand_forecast"].rolling(60).mean()) / (
                dem["demand_forecast"].rolling(60).std() + 1
            )
            stor_z = (df["ng_storage"] - df["ng_storage"].rolling(60).mean()) / (
                df["ng_storage"].rolling(60).std() + 1
            )
            dem["demand_storage_zscore_ratio"] = fc_z / (stor_z + 0.01)

        # -- Demand × heating season interaction --
        if "heating_season" in df.columns:
            dem["demand_x_heating"] = dem["demand_forecast"] * df["heating_season"]
            dem["demand_surprise_x_winter"] = dem["demand_surprise"].fillna(0) * df.get("is_winter", 0)

        # -- Demand × weather interaction --
        if "wx_hdd" in df.columns:
            dem["demand_x_hdd"] = dem["demand_fc_deviation"] * df["wx_hdd"]

        logger.info(
            "Built %d demand-augmented features",
            len(dem.columns),
        )
        return dem

    def _build_forecast_only_features(
        self,
        df: pd.DataFrame,
        forecast: pd.Series,
    ) -> pd.DataFrame:
        """Minimal demand features when no actual demand is available."""
        dem = pd.DataFrame(index=df.index)
        fc = forecast.copy()
        fc.index = pd.to_datetime(fc.index)
        dem["demand_forecast"] = fc.reindex(df.index, method="ffill")
        dem["demand_fc_change_1d"] = dem["demand_forecast"].pct_change(1)
        dem["demand_fc_change_5d"] = dem["demand_forecast"].pct_change(5)
        dem["demand_fc_roll_30d"] = dem["demand_forecast"].rolling(30).mean()
        dem["demand_fc_deviation"] = (
            dem["demand_forecast"] - dem["demand_fc_roll_30d"]
        ) / (dem["demand_fc_roll_30d"] + 1)
        return dem
