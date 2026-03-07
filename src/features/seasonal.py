"""Seasonal and calendar features for energy price modelling.

Energy markets exhibit strong seasonality patterns:
- Natural gas: high demand in heating (Oct-Mar) and cooling (Jun-Sep) seasons
- Crude oil: summer driving season (May-Sep) increases gasoline demand
- Seasonal refinery maintenance and OPEC meeting cycles

This module encodes these cyclical patterns as model features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Month groups for energy seasonality
HEATING_MONTHS = {10, 11, 12, 1, 2, 3}
COOLING_MONTHS = {6, 7, 8, 9}
DRIVING_SEASON_MONTHS = {5, 6, 7, 8, 9}
HURRICANE_MONTHS = {6, 7, 8, 9, 10, 11}
REFINERY_MAINTENANCE_MONTHS = {3, 4, 9, 10}  # Spring and fall turn-arounds


class SeasonalFeatures:
    """Encodes seasonal and calendar patterns for energy markets.

    Generates cyclical (sin/cos) encodings of calendar variables,
    binary season indicators, and time-to-event features.
    """

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all seasonal and calendar features.

        Args:
            df: DataFrame with a ``DatetimeIndex``.

        Returns:
            DataFrame with seasonal feature columns appended.

        Raises:
            TypeError: If the DataFrame index is not a ``DatetimeIndex``.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex for seasonal features")
        df = df.copy()
        df = self.add_calendar_features(df)
        df = self.add_cyclical_features(df)
        df = self.add_season_indicators(df)
        df = self.add_week_of_year(df)
        logger.info("Added seasonal features to DataFrame with shape %s", df.shape)
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic calendar features.

        Args:
            df: DataFrame with a ``DatetimeIndex``.

        Returns:
            DataFrame with ``month``, ``day_of_week``, ``day_of_year``,
            ``quarter``, ``is_month_end``, ``is_quarter_end`` columns.
        """
        df = df.copy()
        idx = df.index
        df["month"] = idx.month
        df["day_of_week"] = idx.dayofweek  # 0=Monday, 6=Sunday
        df["day_of_year"] = idx.dayofyear
        df["quarter"] = idx.quarter
        df["is_month_end"] = idx.is_month_end.astype(int)
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)
        df["year"] = idx.year
        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode cyclical calendar variables using sin/cos transformation.

        Sin/cos encoding preserves the circular nature of calendar variables
        (e.g. December is close to January, not far from it).

        Args:
            df: DataFrame with ``month``, ``day_of_week``, ``day_of_year`` columns.

        Returns:
            DataFrame with sin/cos encoded cyclical features.
        """
        df = df.copy()
        # Month cyclical features (period = 12)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        # Day of week cyclical features (period = 7)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        # Day of year cyclical features (period = 365)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        return df

    def add_season_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary season and demand-regime indicator features.

        Args:
            df: DataFrame with a ``DatetimeIndex`` and ``month`` column.

        Returns:
            DataFrame with season binary indicator columns.
        """
        df = df.copy()
        month = df.index.month
        df["heating_season"] = month.isin(HEATING_MONTHS).astype(int)
        df["cooling_season"] = month.isin(COOLING_MONTHS).astype(int)
        df["driving_season"] = month.isin(DRIVING_SEASON_MONTHS).astype(int)
        df["hurricane_season"] = month.isin(HURRICANE_MONTHS).astype(int)
        df["refinery_maintenance"] = month.isin(REFINERY_MAINTENANCE_MONTHS).astype(int)
        # EIA storage report day (typically Thursday)
        df["eia_report_day"] = (df.index.dayofweek == 3).astype(int)
        return df

    def add_week_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add week-of-year features useful for capturing weekly seasonality.

        Args:
            df: DataFrame with a ``DatetimeIndex``.

        Returns:
            DataFrame with ``week_of_year`` and its sin/cos encoding.
        """
        df = df.copy()
        woy = df.index.isocalendar().week.astype(int)
        df["week_of_year"] = woy.values
        df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        return df
