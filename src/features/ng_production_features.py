"""Feature engineering for NG production trend forecasting.

Builds features to predict whether US natural gas production is
expected to increase or decrease over the coming months.

Key insight: Baker Hughes rig counts lead production by 4-6 months.
By modelling the rig-count → production lag relationship, we can
forecast production direction before it shows up in the data.

This is a medium-term fundamental signal — useful for:
  - NG strip pricing (forward curve shape)
  - Supply/demand balance analysis
  - Seasonal storage adequacy assessment
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NGProductionFeatureBuilder:
    """Builds features for NG production direction forecasting.

    The target is binary: will production be higher or lower in N months
    compared to today?  Features capture rig count dynamics, production
    momentum, seasonal patterns, and the historical rig→production lag.

    Attributes:
        forecast_horizon_days: How far ahead to forecast production direction.
        target_col: Production column to forecast.
    """

    def __init__(
        self,
        forecast_horizon_days: int = 90,
        target_col: str = "production_mmcf",
    ) -> None:
        self.forecast_horizon_days = forecast_horizon_days
        self.target_col = target_col
        logger.info(
            "NGProductionFeatureBuilder (horizon=%d days, target=%s)",
            forecast_horizon_days, target_col,
        )

    def build(
        self,
        production_df: pd.DataFrame,
        price_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the full feature matrix for production direction forecasting.

        Args:
            production_df: Daily DataFrame from ``NGProductionClient.fetch_production_fundamentals()``.
                Must contain rig count and production features.
            price_df: Optional daily NG price DataFrame (adds price-signal features).
                Should have a ``close`` or ``ng_close`` column.

        Returns:
            DataFrame with features + ``target`` column.
            Target = 1 if production increased over the forecast horizon, else 0.
        """
        if self.target_col not in production_df.columns:
            raise ValueError(
                f"Target '{self.target_col}' not in production data.  "
                f"Available: {list(production_df.columns)}"
            )

        df = production_df.copy()

        # Merge price data if provided
        if price_df is not None:
            df = df.join(price_df, how="left")
            df = df.ffill()

        # Add feature groups
        df = self._add_calendar_features(df)
        df = self._add_rig_features(df)
        df = self._add_production_features(df)
        if price_df is not None:
            df = self._add_price_features(df)
        df = self._add_interaction_features(df)

        # Target: production direction over the forecast horizon
        future_prod = df[self.target_col].shift(-self.forecast_horizon_days)
        df["target"] = (future_prod > df[self.target_col]).astype(float)

        # Select feature columns
        exclude_cols = {
            "target", self.target_col,
            "production_mmcf", "drilling_index",  # raw values excluded; use derived features
        }
        feature_cols = sorted([
            c for c in df.columns
            if c not in exclude_cols
            and c not in ("close", "ng_close", "open", "high", "low", "volume")
        ])

        result = df[feature_cols + ["target"]].dropna(subset=["target"])
        logger.info(
            "NG production features: %d rows × %d features + target (horizon=%d days)",
            len(result), len(feature_cols), self.forecast_horizon_days,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Calendar Features
    # ------------------------------------------------------------------ #
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seasonal patterns in NG production.

        Production has mild seasonality (maintenance schedules,
        freeze-offs in winter, associated gas from oil drilling in summer).
        """
        idx = df.index
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["day_of_year"] = idx.dayofyear

        doy_frac = df["day_of_year"] / 365.25
        for k in [1, 2]:
            df[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy_frac)
            df[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy_frac)

        # Winter freeze-off risk flag
        df["freeze_off_risk"] = df["month"].isin([12, 1, 2]).astype(int)

        # Hurricane season flag (Gulf of Mexico production disruption)
        df["hurricane_season"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)

        logger.debug("Added calendar features")
        return df

    # ------------------------------------------------------------------ #
    #  Drilling Activity Features
    # ------------------------------------------------------------------ #
    def _add_rig_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced drilling activity features for production forecasting.

        Uses the FRED Industrial Production: Drilling index as a proxy
        for Baker Hughes rig counts.  We capture:
        - Level relative to history
        - Rate of change (momentum)
        - Acceleration (second derivative)
        - Regime (up vs down trend)
        """
        if "drilling_index" not in df.columns:
            logger.warning("No drilling index data — skipping drilling features")
            return df

        di = df["drilling_index"]

        # Normalised drilling index (z-score relative to 2-year rolling)
        roll = di.rolling(730, min_periods=90)
        df["drill_zscore"] = (di - roll.mean()) / roll.std().clip(lower=1)

        # Rate of change at multiple horizons
        for months in [1, 3, 6, 12]:
            days = months * 30
            col = f"drill_roc_{months}m"
            df[col] = di.pct_change(days) * 100  # Percentage

        # Binary direction indicators
        df["drill_rising_3m"] = (di > di.shift(90)).astype(int)
        df["drill_rising_6m"] = (di > di.shift(180)).astype(int)

        # Drilling index as % of 5-year high/low
        five_yr_max = di.rolling(365 * 5, min_periods=365).max()
        five_yr_min = di.rolling(365 * 5, min_periods=365).min()
        df["drill_vs_5yr_high"] = di / five_yr_max.clip(lower=0.01)
        df["drill_vs_5yr_low"] = di / five_yr_min.clip(lower=0.01)

        logger.debug("Added drilling activity features")
        return df

    # ------------------------------------------------------------------ #
    #  Production Trend Features
    # ------------------------------------------------------------------ #
    def _add_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Production momentum and trend features."""
        if self.target_col not in df.columns:
            return df

        prod = df[self.target_col]

        # Normalised production (z-score)
        roll = prod.rolling(365, min_periods=60)
        df["prod_zscore"] = (prod - roll.mean()) / roll.std().clip(lower=0.001)

        # Month-over-month change
        df["prod_mom_1m"] = prod.pct_change(30)
        df["prod_mom_3m"] = prod.pct_change(90)
        df["prod_mom_6m"] = prod.pct_change(180)

        # Production smoothed (reduce monthly noise)
        df["prod_smooth_3m"] = prod.rolling(90).mean()
        df["prod_smooth_6m"] = prod.rolling(180).mean()

        # Year-over-year growth rate
        df["prod_yoy_growth"] = prod.pct_change(365)

        # Production acceleration (second derivative)
        df["prod_accel_3m"] = df["prod_mom_3m"].diff(90)

        logger.debug("Added production features")
        return df

    # ------------------------------------------------------------------ #
    #  Price Features (optional)
    # ------------------------------------------------------------------ #
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """NG price features — price signals that affect drilling decisions.

        High prices incentivise drilling → more rigs → more production.
        Low prices cause rig count declines → less production.
        """
        # Find price column
        price_col = None
        for candidate in ["close", "ng_close", "NG=F"]:
            if candidate in df.columns:
                price_col = candidate
                break

        if price_col is None:
            logger.debug("No price data found — skipping price features")
            return df

        price = pd.to_numeric(df[price_col], errors="coerce")

        # Price level relative to history
        roll = price.rolling(365, min_periods=60)
        df["price_zscore"] = (price - roll.mean()) / roll.std().clip(lower=0.01)

        # Price momentum
        df["price_mom_1m"] = price.pct_change(30)
        df["price_mom_3m"] = price.pct_change(90)

        # Price above/below breakeven proxy ($2.50/MMBtu approximate breakeven)
        df["price_above_breakeven"] = (price > 2.50).astype(int)

        # High price regime (incentivises drilling)
        df["price_high_regime"] = (price > price.rolling(365).quantile(0.75)).astype(int)

        logger.debug("Added price-signal features")
        return df

    # ------------------------------------------------------------------ #
    #  Interaction Features
    # ------------------------------------------------------------------ #
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-feature interactions."""
        # Drilling trend × production trend
        if "drill_rising_3m" in df.columns and "prod_mom_3m" in df.columns:
            df["drill_up_prod_up"] = (
                df["drill_rising_3m"] * (df["prod_mom_3m"] > 0).astype(int)
            )
            df["drill_down_prod_down"] = (
                (1 - df["drill_rising_3m"]) * (df["prod_mom_3m"] < 0).astype(int)
            )

        # Drilling × price regime
        if "drill_zscore" in df.columns and "price_high_regime" in df.columns:
            df["drill_x_price_high"] = df["drill_zscore"] * df["price_high_regime"]

        # Drilling × season (seasonal drilling patterns)
        if "drilling_index" in df.columns:
            di = df["drilling_index"]
            if "freeze_off_risk" in df.columns:
                df["drill_x_winter"] = di * df["freeze_off_risk"]
            if "hurricane_season" in df.columns:
                df["drill_x_hurricane"] = di * df["hurricane_season"]

        logger.debug("Added interaction features")
        return df

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_streak(condition: pd.Series) -> pd.Series:
        """Count consecutive True values in a boolean series."""
        groups = (~condition).cumsum()
        streaks = condition.groupby(groups).cumsum()
        return streaks.fillna(0).astype(int)
