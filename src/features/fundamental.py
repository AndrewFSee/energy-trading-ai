"""Fundamental energy market features.

Computes fundamental supply/demand features from EIA data including
storage surprises, rig count changes, futures curve shape (contango/backwardation),
and production/import balances.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """Computes fundamental energy market features.

    Transforms raw EIA storage and production data into model-ready features
    that capture supply/demand imbalances.

    Attributes:
        storage_surprise_window: Rolling window (weeks) for storage surprise.
        rig_count_lag: Lag (weeks) to apply to rig count data.
    """

    def __init__(
        self,
        storage_surprise_window: int = 4,
        rig_count_lag: int = 1,
    ) -> None:
        """Initialise fundamental features calculator.

        Args:
            storage_surprise_window: Weeks of history for computing expected
                storage change (baseline for surprise calculation).
            rig_count_lag: Number of weeks to lag rig count data (reflects
                delayed production impact).
        """
        self.storage_surprise_window = storage_surprise_window
        self.rig_count_lag = rig_count_lag

    def add_storage_features(
        self, price_df: pd.DataFrame, storage_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge storage level and surprise features onto the price DataFrame.

        Args:
            price_df: Daily price DataFrame (indexed by date).
            storage_df: Weekly storage DataFrame from ``EIAClient``.

        Returns:
            ``price_df`` with storage-related columns added and forward-filled
            to daily frequency.
        """
        df = price_df.copy()
        storage = storage_df.copy()

        # Compute storage surprise
        col = storage.columns[0]
        storage["storage_change"] = storage[col].diff()
        storage["storage_expected"] = (
            storage["storage_change"].rolling(self.storage_surprise_window).mean()
        )
        storage["storage_surprise"] = storage["storage_change"] - storage["storage_expected"]
        storage["storage_yoy"] = storage[col] - storage[col].shift(52)  # year-over-year

        # Normalise surprise by 5-year average absolute change
        abs_mean = storage["storage_change"].abs().rolling(260).mean()
        storage["storage_surprise_norm"] = storage["storage_surprise"] / abs_mean.replace(0, np.nan)

        # Merge and forward-fill to daily
        storage = storage.resample("D").last().ffill()
        df = df.join(
            storage[["storage_change", "storage_surprise", "storage_surprise_norm", "storage_yoy"]],
            how="left",
        )
        df[["storage_change", "storage_surprise", "storage_surprise_norm", "storage_yoy"]] = df[
            ["storage_change", "storage_surprise", "storage_surprise_norm", "storage_yoy"]
        ].ffill()
        logger.info("Added storage features to price DataFrame")
        return df

    def add_futures_curve_features(
        self,
        df: pd.DataFrame,
        spot: pd.Series,
        front_month: pd.Series,
        second_month: pd.Series,
    ) -> pd.DataFrame:
        """Add futures curve shape features (contango/backwardation).

        Backwardation (spot > futures) is typically bullish — indicates
        tight near-term supply.  Contango (futures > spot) is bearish.

        Args:
            df: Price DataFrame to merge features into.
            spot: Spot price series (indexed by date).
            front_month: Front-month futures price series.
            second_month: Second-month futures price series.

        Returns:
            DataFrame with curve shape features added.
        """
        df = df.copy()
        spread_1 = front_month - spot
        spread_2 = second_month - front_month
        df["curve_spread_m1"] = spread_1.reindex(df.index).ffill()
        df["curve_spread_m2"] = spread_2.reindex(df.index).ffill()
        df["in_backwardation"] = (df["curve_spread_m1"] < 0).astype(int)
        df["in_contango"] = (df["curve_spread_m1"] > 0).astype(int)
        logger.debug("Added futures curve features")
        return df

    def add_rig_count_features(
        self,
        df: pd.DataFrame,
        rig_count: pd.Series,
    ) -> pd.DataFrame:
        """Add Baker Hughes rig count change features.

        Rig count changes are a leading indicator of future production.
        Rising rig counts suggest future supply increases (bearish signal).

        Args:
            df: Price DataFrame.
            rig_count: Weekly Baker Hughes rig count series.

        Returns:
            DataFrame with rig count features added.
        """
        df = df.copy()
        rc = rig_count.copy()
        rc_df = pd.DataFrame({"rig_count": rc})
        rc_df["rig_count_chg"] = rc_df["rig_count"].diff()
        rc_df["rig_count_chg_4w"] = rc_df["rig_count"].diff(4)
        rc_df["rig_count_yoy"] = rc_df["rig_count"].diff(52)

        # Apply lag to capture delayed production impact
        if self.rig_count_lag > 0:
            rc_df = rc_df.shift(self.rig_count_lag)

        rc_daily = rc_df.resample("D").last().ffill()
        df = df.join(rc_daily, how="left")
        df[rc_daily.columns] = df[rc_daily.columns].ffill()
        logger.debug("Added rig count features with lag=%d", self.rig_count_lag)
        return df

    def add_production_balance(
        self,
        df: pd.DataFrame,
        production: pd.Series | None = None,
        imports: pd.Series | None = None,
        exports: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Add net supply balance features from production and trade data.

        Args:
            df: Price DataFrame.
            production: Weekly production series (thousand barrels/day).
            imports: Weekly imports series.
            exports: Weekly exports series.

        Returns:
            DataFrame with supply balance features added.
        """
        df = df.copy()
        supply_parts: dict[str, pd.Series | None] = {
            "production": production,
            "imports": imports,
            "exports": exports,
        }
        for name, series in supply_parts.items():
            if series is not None:
                daily = series.resample("D").last().ffill()
                df[f"supply_{name}"] = daily.reindex(df.index).ffill()

        # Compute net supply if all components available
        if production is not None and imports is not None and exports is not None:
            df["net_supply"] = (
                df.get("supply_production", 0)
                + df.get("supply_imports", 0)
                - df.get("supply_exports", 0)
            )

        logger.debug("Added production balance features")
        return df
