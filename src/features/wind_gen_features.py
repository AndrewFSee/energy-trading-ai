"""Feature engineering for wind & solar generation forecasting.

Builds feature matrices for predicting daily renewable generation from
weather variables, calendar effects, and autoregressive generation lags.

The wind power curve is highly non-linear:
  - Below ~3 m/s (cut-in): zero output
  - 3–12 m/s: approximately cubic relationship (P ∝ v³)
  - 12–25 m/s: rated output (flat)
  - Above 25 m/s (cut-out): zero (turbines shut down for safety)

Solar output depends primarily on:
  - Shortwave radiation (strongest driver)
  - Cloud cover / direct vs diffuse radiation ratio
  - Day length (seasonal)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# US Federal Holidays (same as load_features.py)
_US_HOLIDAYS_MMDD = {
    (1, 1), (1, 15), (2, 19), (5, 27), (6, 19), (7, 4),
    (9, 2), (10, 14), (11, 11), (11, 28), (12, 25),
}


def _is_holiday(dt: pd.Timestamp) -> bool:
    return (dt.month, dt.day) in _US_HOLIDAYS_MMDD


class WindGenFeatureBuilder:
    """Builds features for day-ahead wind generation forecasting.

    Combines weather variables (wind speed, gusts, direction) with
    calendar effects and autoregressive lags to predict total daily
    wind MWh across Eastern US regions.

    Attributes:
        target_col: Name of the wind generation column to forecast.
        lag_days: Autoregressive lag offsets.
        rolling_windows: Rolling statistic window sizes.
    """

    def __init__(
        self,
        target_col: str = "east_wind_total_mwh",
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.target_col = target_col
        self.lag_days = lag_days or [1, 2, 3, 7, 14, 28]
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        logger.info(
            "WindGenFeatureBuilder (target=%s, lags=%s, windows=%s)",
            target_col, self.lag_days, self.rolling_windows,
        )

    def build(
        self,
        generation_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full feature matrix for wind generation forecasting.

        Args:
            generation_df: Daily generation DataFrame (from ``EIAGenerationClient``).
                Must contain the column specified by ``target_col``.
            weather_df: Daily weather DataFrame (from ``OpenMeteoClient.fetch_wind_solar_weather``).
                Expected columns: wind_speed_max, wind_gusts_max,
                wind_dir_dominant, shortwave_rad, etc.

        Returns:
            DataFrame with features + ``target`` column.
        """
        if self.target_col not in generation_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not in generation data.  "
                f"Available: {list(generation_df.columns)}"
            )

        # Merge generation and weather on date index
        df = generation_df.join(weather_df, how="inner")
        logger.info("Merged generation + weather: %d rows", len(df))

        # Add feature groups
        df = self._add_calendar_features(df)
        df = self._add_wind_power_features(df)
        df = self._add_solar_features(df)
        df = self._add_generation_lags(df)
        df = self._add_interaction_features(df)

        # Target: next-day wind generation
        df["target"] = df[self.target_col].shift(-1)

        # Select feature columns (exclude raw generation targets)
        exclude_suffixes = ("_total_mwh", "_peak_mw", "_min_mw", "_avg_mw")
        feature_cols = [
            c for c in df.columns
            if c != "target"
            and not any(c.endswith(s) for s in exclude_suffixes)
        ]
        # But keep lag/rolling generation features
        gen_feature_cols = [
            c for c in df.columns
            if c.startswith("gen_lag") or c.startswith("gen_roll")
            or c.startswith("gen_pct") or c.startswith("cf_")
        ]
        feature_cols = sorted(set(feature_cols) | set(gen_feature_cols))

        result = df[feature_cols + ["target"]].dropna(subset=["target"])
        logger.info(
            "Wind gen features: %d rows × %d features + target",
            len(result), len(feature_cols),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Calendar Features
    # ------------------------------------------------------------------ #
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar features capturing seasonal wind patterns."""
        idx = df.index

        df["day_of_year"] = idx.dayofyear
        df["month"] = idx.month
        df["day_of_week"] = idx.dayofweek
        df["quarter"] = idx.quarter

        # Fourier encoding of day-of-year (wind has strong seasonal pattern)
        doy_frac = df["day_of_year"] / 365.25
        for k in [1, 2, 3]:
            df[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy_frac)
            df[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy_frac)

        # Season flags
        df["season_winter"] = df["month"].isin([12, 1, 2]).astype(int)
        df["season_spring"] = df["month"].isin([3, 4, 5]).astype(int)
        df["season_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["season_fall"] = df["month"].isin([9, 10, 11]).astype(int)

        logger.debug("Added calendar features")
        return df

    # ------------------------------------------------------------------ #
    #  Wind Power Curve Features
    # ------------------------------------------------------------------ #
    def _add_wind_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features capturing the non-linear wind power curve.

        Wind power ∝ v³ between cut-in and rated speed.  We model this
        with polynomial terms and physics-informed regime indicators.
        """
        if "wind_speed_max" not in df.columns:
            logger.warning("No wind speed data — skipping wind power features")
            return df

        ws = df["wind_speed_max"]

        # Raw polynomial terms (captures cubic power curve)
        df["wind_speed_sq"] = ws ** 2
        df["wind_speed_cube"] = ws ** 3

        # Log wind speed (compresses high values)
        df["wind_speed_log"] = np.log1p(ws)

        # Wind speed regimes (based on typical turbine thresholds)
        # Note: wind_speed_max here is in km/h from Open-Meteo
        # Typical cut-in ~11 km/h (3 m/s), rated ~43 km/h (12 m/s),
        # cut-out ~90 km/h (25 m/s)
        df["wind_below_cutin"] = (ws < 11).astype(int)
        df["wind_rated"] = ((ws >= 43) & (ws < 90)).astype(int)
        df["wind_above_cutout"] = (ws >= 90).astype(int)
        df["wind_productive"] = ((ws >= 11) & (ws < 90)).astype(int)

        # Effective wind speed (clipped to productive range)
        df["wind_effective"] = ws.clip(lower=11, upper=90)

        # Gust features
        if "wind_gusts_max" in df.columns:
            df["gust_speed"] = df["wind_gusts_max"]
            df["gust_diff"] = df["wind_gusts_max"] - ws
            # High gusts relative to speed = turbulence = reduced output
            df["turbulence_proxy"] = (
                df["gust_diff"] / ws.clip(lower=1)
            ).clip(0, 5)

        # Wind direction features (circular encoding)
        if "wind_dir_dominant" in df.columns:
            wd_rad = np.deg2rad(df["wind_dir_dominant"])
            df["wind_dir_sin"] = np.sin(wd_rad)
            df["wind_dir_cos"] = np.cos(wd_rad)
            # Directional wind speed components
            df["wind_u"] = ws * np.sin(wd_rad)  # East-west component
            df["wind_v"] = ws * np.cos(wd_rad)  # North-south component

        # Lagged wind speed
        for lag in [1, 2, 3, 7]:
            df[f"wind_lag_{lag}d"] = ws.shift(lag)

        # Wind persistence (autocorrelation feature)
        df["wind_change_1d"] = ws.diff(1)
        df["wind_change_3d"] = ws.diff(3)

        logger.debug("Added wind power features")
        return df

    # ------------------------------------------------------------------ #
    #  Solar Features
    # ------------------------------------------------------------------ #
    def _add_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Solar irradiance features for additional context.

        While the primary target is wind, solar generation is anti-correlated
        with cloud cover which also affects wind patterns.
        """
        if "shortwave_rad" not in df.columns:
            return df

        sr = df["shortwave_rad"]
        df["solar_rad_sq"] = sr ** 2

        # Cloud proxy: low radiation relative to seasonal norm
        df["solar_rad_anomaly_7d"] = sr - sr.rolling(7).mean()
        df["solar_rad_anomaly_30d"] = sr - sr.rolling(30).mean()

        # Sunshine duration features
        if "sunshine_hours" in df.columns:
            df["sunshine_frac"] = df["sunshine_hours"] / 14.0  # Normalise by ~max day length
            df["sunshine_frac"] = df["sunshine_frac"].clip(0, 1)

        logger.debug("Added solar features")
        return df

    # ------------------------------------------------------------------ #
    #  Generation Lag Features
    # ------------------------------------------------------------------ #
    def _add_generation_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Autoregressive generation features.

        Yesterday's wind generation is informative because weather
        patterns persist across days (frontal systems, jet stream).
        """
        gen = df[self.target_col]

        # Point lags
        for lag in self.lag_days:
            df[f"gen_lag_{lag}d"] = gen.shift(lag)

        # Rolling averages and volatility
        for win in self.rolling_windows:
            df[f"gen_roll_{win}d"] = gen.rolling(win).mean()
            df[f"gen_roll_{win}d_std"] = gen.rolling(win).std()

        # Percentage change
        df["gen_pct_1d"] = gen.pct_change(1)
        df["gen_pct_7d"] = gen.pct_change(7)

        # Same day last week (weekly pattern from grid scheduling)
        df["gen_same_dow_1w"] = gen.shift(7)
        df["gen_same_dow_2w"] = gen.shift(14)

        # Capacity factor lags (if available)
        cf_cols = [c for c in df.columns if c.endswith("_cf") and "wnd" in c]
        for col in cf_cols:
            df[f"cf_lag_1d_{col}"] = df[col].shift(1)
            df[f"cf_lag_7d_{col}"] = df[col].shift(7)

        logger.debug("Added generation lag features")
        return df

    # ------------------------------------------------------------------ #
    #  Interaction Features
    # ------------------------------------------------------------------ #
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weather × season cross features.

        Wind generation varies seasonally: stronger winds in spring/fall
        in the Great Plains, weaker in summer.
        """
        if "wind_speed_max" not in df.columns:
            return df

        ws = df["wind_speed_max"]

        # Wind speed × season
        df["wind_x_winter"] = ws * df["season_winter"]
        df["wind_x_spring"] = ws * df["season_spring"]
        df["wind_x_summer"] = ws * df["season_summer"]
        df["wind_x_fall"] = ws * df["season_fall"]

        # Wind speed × temperature (cold fronts bring high winds)
        if "tavg" in df.columns:
            df["wind_x_temp"] = ws * df["tavg"]
            df["wind_x_cold"] = ws * (df["tavg"] < 32).astype(int)

        # Wind cube × productive regime (main power generation signal)
        df["power_signal"] = (ws ** 3) * df.get("wind_productive", 1)

        logger.debug("Added interaction features")
        return df


class SolarGenFeatureBuilder:
    """Builds features for day-ahead solar generation forecasting.

    Solar generation is primarily driven by shortwave radiation,
    cloud cover, and day length (seasonal).

    Attributes:
        target_col: Name of the solar generation column.
        lag_days: Autoregressive lag offsets.
        rolling_windows: Rolling window sizes.
    """

    def __init__(
        self,
        target_col: str = "east_solar_total_mwh",
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.target_col = target_col
        self.lag_days = lag_days or [1, 2, 3, 7, 14]
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        logger.info("SolarGenFeatureBuilder (target=%s)", target_col)

    def build(
        self,
        generation_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build solar generation feature matrix.

        Args:
            generation_df: Daily generation DataFrame.
            weather_df: Daily weather with solar radiation variables.

        Returns:
            DataFrame with features + ``target`` column.
        """
        if self.target_col not in generation_df.columns:
            raise ValueError(
                f"Target '{self.target_col}' not in generation data.  "
                f"Available: {list(generation_df.columns)}"
            )

        df = generation_df.join(weather_df, how="inner")
        logger.info("Merged generation + weather for solar: %d rows", len(df))

        df = self._add_calendar_features(df)
        df = self._add_solar_power_features(df)
        df = self._add_generation_lags(df)

        # Target: next-day solar generation
        df["target"] = df[self.target_col].shift(-1)

        exclude_suffixes = ("_total_mwh", "_peak_mw", "_min_mw", "_avg_mw")
        feature_cols = [
            c for c in df.columns
            if c != "target"
            and not any(c.endswith(s) for s in exclude_suffixes)
        ]
        gen_features = [
            c for c in df.columns
            if c.startswith("solar_gen_lag") or c.startswith("solar_gen_roll")
            or c.startswith("solar_gen_pct")
        ]
        feature_cols = sorted(set(feature_cols) | set(gen_features))

        result = df[feature_cols + ["target"]].dropna(subset=["target"])
        logger.info(
            "Solar gen features: %d rows × %d features",
            len(result), len(feature_cols),
        )
        return result

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["day_of_year"] = idx.dayofyear
        df["month"] = idx.month

        doy_frac = df["day_of_year"] / 365.25
        for k in [1, 2]:
            df[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy_frac)
            df[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy_frac)

        df["season_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["season_winter"] = df["month"].isin([12, 1, 2]).astype(int)
        return df

    def _add_solar_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "shortwave_rad" not in df.columns:
            return df

        sr = df["shortwave_rad"]
        df["solar_rad_sq"] = sr ** 2
        df["solar_rad_log"] = np.log1p(sr)

        # Clear sky indicator (from cloud cover — archive API lacks direct_rad)
        if "clear_sky_index" in df.columns:
            df["clear_sky"] = df["clear_sky_index"]
        elif "cloud_cover" in df.columns:
            df["clear_sky"] = (1.0 - df["cloud_cover"] / 100.0).clip(0, 1)

        # Precipitation flag (rain/clouds reduce solar output)
        if "precipitation" in df.columns:
            df["rain_flag"] = (df["precipitation"] > 1.0).astype(int)

        if "sunshine_hours" in df.columns:
            df["sunshine_norm"] = df["sunshine_hours"].clip(lower=0) / 14.0

        # Lagged radiation
        for lag in [1, 2, 3, 7]:
            df[f"solar_rad_lag_{lag}d"] = sr.shift(lag)

        # Radiation anomaly from seasonal norm
        df["solar_anomaly_7d"] = sr - sr.rolling(7).mean()
        df["solar_anomaly_30d"] = sr - sr.rolling(30).mean()

        return df

    def _add_generation_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        gen = df[self.target_col]

        for lag in self.lag_days:
            df[f"solar_gen_lag_{lag}d"] = gen.shift(lag)
        for win in self.rolling_windows:
            df[f"solar_gen_roll_{win}d"] = gen.rolling(win).mean()
            df[f"solar_gen_roll_{win}d_std"] = gen.rolling(win).std()

        df["solar_gen_pct_1d"] = gen.pct_change(1)
        df["solar_gen_pct_7d"] = gen.pct_change(7)
        return df
