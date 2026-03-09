"""Feature engineering for electricity demand / load forecasting.

Builds a rich feature matrix for day-ahead load prediction by combining:
  1. **Weather features** — HDD, CDD, temperature lags & rolling stats.
  2. **Calendar features** — Day-of-week, month, US holidays, season.
  3. **Load lags** — Previous-day and rolling historical demand.
  4. **Interaction features** — Weather × calendar cross terms.

The physical relationship between temperature and electricity demand is
well-established: demand follows a U-shaped curve around ~65 °F, driven
by heating load in winter and cooling load in summer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── US Federal Holidays (fixed-date approximations) ─────────────────────
_US_HOLIDAYS_MMDD = {
    (1, 1),    # New Year's Day
    (1, 15),   # MLK Day (approx)
    (2, 19),   # Presidents' Day (approx)
    (5, 27),   # Memorial Day (approx)
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (9, 2),    # Labor Day (approx)
    (10, 14),  # Columbus Day (approx)
    (11, 11),  # Veterans Day
    (11, 28),  # Thanksgiving (approx)
    (12, 25),  # Christmas
}


def _is_holiday(dt: pd.Timestamp) -> bool:
    """Check whether a date is approximately a US federal holiday."""
    return (dt.month, dt.day) in _US_HOLIDAYS_MMDD


class LoadFeatureBuilder:
    """Builds features for day-ahead electricity demand forecasting.

    The feature set is designed to capture the key physical and behavioural
    drivers of regional power demand: weather sensitivity, calendar patterns,
    and demand persistence (autoregressive lags).

    Attributes:
        target_col: Name of the demand column to forecast.
        lag_days: List of lag offsets to include.
        rolling_windows: List of rolling-mean window sizes (days).
    """

    def __init__(
        self,
        target_col: str = "east_total_mwh",
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.target_col = target_col
        self.lag_days = lag_days or [1, 2, 3, 7, 14, 28]
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        logger.info(
            "LoadFeatureBuilder (target=%s, lags=%s, windows=%s)",
            target_col, self.lag_days, self.rolling_windows,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def build(
        self,
        demand_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full feature matrix from demand and weather data.

        Args:
            demand_df: Daily demand DataFrame (from ``EIADemandClient``).
                Must contain the column specified by ``target_col``.
            weather_df: Daily weather DataFrame (from ``OpenMeteoClient``).
                Expected columns: ``tavg``, ``tmax``, ``tmin``,
                ``hdd``, ``cdd``, and rolling variants.

        Returns:
            DataFrame with features + ``target`` column.  Rows with
            NaN targets are dropped.
        """
        if self.target_col not in demand_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in demand data.  "
                f"Available: {list(demand_df.columns)}"
            )

        # Merge demand and weather on date index
        df = demand_df.join(weather_df, how="inner")
        logger.info("Merged demand + weather: %d rows", len(df))

        # Add feature groups
        df = self._add_calendar_features(df)
        df = self._add_weather_features(df)
        df = self._add_load_lags(df)
        df = self._add_interaction_features(df)

        # Target: next-day total demand (shift for day-ahead forecasting)
        df["target"] = df[self.target_col].shift(-1)

        # Keep only feature columns + target
        feature_cols = [c for c in df.columns if c != "target" and c != self.target_col
                        and not c.endswith("_total_mwh") and not c.endswith("_peak_mw")
                        and not c.endswith("_min_mw") and not c.endswith("_avg_mw")
                        and c != "east_peak_mw" and c != "east_total_mwh"]
        # But keep some derived load columns that are features (lags, rolling)
        load_feature_cols = [c for c in df.columns
                             if c.startswith("load_lag") or c.startswith("load_roll")
                             or c.startswith("load_pct") or c.startswith("peak_lag")]
        feature_cols = list(set(feature_cols) | set(load_feature_cols))
        feature_cols = sorted(feature_cols)

        result = df[feature_cols + ["target"]].dropna(subset=["target"])
        logger.info(
            "Feature matrix: %d rows × %d features + target",
            len(result), len(feature_cols),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Calendar Features
    # ------------------------------------------------------------------ #
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based demand predictors.

        Power demand has strong weekly and seasonal patterns:
        - Weekdays > weekends (commercial/industrial load)
        - Summer and winter peaks (cooling and heating)
        - Holidays behave like weekends
        """
        idx = df.index

        df["day_of_week"] = idx.dayofweek                     # 0=Mon … 6=Sun
        df["day_of_year"] = idx.dayofyear
        df["month"] = idx.month
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["quarter"] = idx.quarter

        # Binary flags
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["is_monday"] = (idx.dayofweek == 0).astype(int)
        df["is_friday"] = (idx.dayofweek == 4).astype(int)
        df["is_holiday"] = idx.to_series().apply(_is_holiday).astype(int).values

        # Fourier encoding of day-of-year (captures smooth seasonality)
        doy_frac = df["day_of_year"] / 365.25
        for k in [1, 2, 3]:
            df[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy_frac)
            df[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy_frac)

        # Fourier encoding of day-of-week
        dow_frac = df["day_of_week"] / 7.0
        df["dow_sin"] = np.sin(2 * np.pi * dow_frac)
        df["dow_cos"] = np.cos(2 * np.pi * dow_frac)

        # Season buckets (meteorological)
        df["season_winter"] = df["month"].isin([12, 1, 2]).astype(int)
        df["season_spring"] = df["month"].isin([3, 4, 5]).astype(int)
        df["season_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["season_fall"] = df["month"].isin([9, 10, 11]).astype(int)

        logger.debug("Added %d calendar features", 20)
        return df

    # ------------------------------------------------------------------ #
    #  Weather Features
    # ------------------------------------------------------------------ #
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived weather features.

        The temperature–demand relationship is non-linear (U-shaped).
        We capture this with polynomial terms and extreme indicators.
        """
        if "tavg" not in df.columns:
            logger.warning("No weather data — skipping weather features")
            return df

        # Non-linear temperature terms
        df["tavg_sq"] = df["tavg"] ** 2
        df["tavg_cube"] = df["tavg"] ** 3

        # HDD/CDD powers (capture non-linear heating/cooling ramp)
        if "hdd" in df.columns:
            df["hdd_sq"] = df["hdd"] ** 2
        if "cdd" in df.columns:
            df["cdd_sq"] = df["cdd"] ** 2

        # Extreme weather indicators
        df["extreme_cold"] = (df["tavg"] < 20).astype(int)     # < 20°F
        df["extreme_heat"] = (df["tavg"] > 90).astype(int)     # > 90°F
        df["moderate_temp"] = ((df["tavg"] >= 55) & (df["tavg"] <= 75)).astype(int)

        # Temperature volatility (recent variability drives uncertain load)
        df["temp_volatility_7d"] = df["tavg"].rolling(7).std()
        df["temp_volatility_14d"] = df["tavg"].rolling(14).std()

        # Lagged weather (weather persistence)
        for lag in [1, 2, 3]:
            df[f"tavg_lag_{lag}d"] = df["tavg"].shift(lag)
            if "hdd" in df.columns:
                df[f"hdd_lag_{lag}d"] = df["hdd"].shift(lag)
            if "cdd" in df.columns:
                df[f"cdd_lag_{lag}d"] = df["cdd"].shift(lag)

        # Week-over-week temperature change
        df["temp_wow_change"] = df["tavg"] - df["tavg"].shift(7)

        logger.debug("Added weather-derived features")
        return df

    # ------------------------------------------------------------------ #
    #  Load Lag Features
    # ------------------------------------------------------------------ #
    def _add_load_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add autoregressive load features.

        Historical demand is the single best predictor of future demand
        (persistence forecast).  We capture this at multiple horizons.
        """
        load = df[self.target_col]

        # Point lags
        for lag in self.lag_days:
            df[f"load_lag_{lag}d"] = load.shift(lag)

        # Rolling averages
        for win in self.rolling_windows:
            df[f"load_roll_{win}d"] = load.rolling(win).mean()
            df[f"load_roll_{win}d_std"] = load.rolling(win).std()

        # Percentage change from lagged values
        df["load_pct_change_1d"] = load.pct_change(1)
        df["load_pct_change_7d"] = load.pct_change(7)

        # Same day last week (strong weekly autocorrelation)
        df["load_same_dow_1w"] = load.shift(7)
        df["load_same_dow_2w"] = load.shift(14)

        # Peak demand lags (if available)
        if "east_peak_mw" in df.columns:
            df["peak_lag_1d"] = df["east_peak_mw"].shift(1)
            df["peak_lag_7d"] = df["east_peak_mw"].shift(7)

        logger.debug("Added load lag features")
        return df

    # ------------------------------------------------------------------ #
    #  Interaction Features
    # ------------------------------------------------------------------ #
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather × calendar cross features.

        These capture the different demand responses to weather depending
        on the day type (e.g. weekday heating load vs weekend heating load).
        """
        if "hdd" not in df.columns:
            return df

        # Weather × weekday/weekend
        df["hdd_x_weekday"] = df["hdd"] * (1 - df["is_weekend"])
        df["cdd_x_weekday"] = df["cdd"] * (1 - df["is_weekend"])
        df["hdd_x_weekend"] = df["hdd"] * df["is_weekend"]
        df["cdd_x_weekend"] = df["cdd"] * df["is_weekend"]

        # Weather × season
        df["hdd_x_winter"] = df["hdd"] * df["season_winter"]
        df["cdd_x_summer"] = df["cdd"] * df["season_summer"]

        # Temperature × time of year  (peak cooling months)
        df["temp_x_jul_aug"] = df["tavg"] * df["month"].isin([7, 8]).astype(int)
        df["temp_x_jan_feb"] = df["tavg"] * df["month"].isin([1, 2]).astype(int)

        # Holiday × weather (holidays in extreme weather → higher residential load)
        df["hdd_x_holiday"] = df["hdd"] * df["is_holiday"]
        df["cdd_x_holiday"] = df["cdd"] * df["is_holiday"]

        logger.debug("Added interaction features")
        return df
