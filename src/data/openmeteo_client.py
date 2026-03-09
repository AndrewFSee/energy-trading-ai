"""Open-Meteo historical weather client for energy modelling.

Fetches daily weather data from the free Open-Meteo API for key
Eastern/Central US population centres.  Provides:
  1. **Temperature** — HDD, CDD for demand modelling.
  2. **Wind** — Hub-height (100 m) wind speed for wind generation forecasting.
  3. **Solar** — Shortwave radiation and cloud cover for solar forecasting.

No API key is required.  Rate limit: ~10,000 requests/day.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Major Eastern/Central US cities weighted by population & energy load
CITIES = {
    "new_york":     {"lat": 40.71, "lon": -74.01, "weight": 0.20},
    "chicago":      {"lat": 41.88, "lon": -87.63, "weight": 0.15},
    "houston":      {"lat": 29.76, "lon": -95.37, "weight": 0.12},
    "philadelphia": {"lat": 39.95, "lon": -75.17, "weight": 0.10},
    "dallas":       {"lat": 32.78, "lon": -96.80, "weight": 0.10},
    "atlanta":      {"lat": 33.75, "lon": -84.39, "weight": 0.08},
    "detroit":      {"lat": 42.33, "lon": -83.05, "weight": 0.07},
    "boston":        {"lat": 42.36, "lon": -71.06, "weight": 0.06},
    "miami":        {"lat": 25.76, "lon": -80.19, "weight": 0.06},
    "pittsburgh":   {"lat": 40.44, "lon": -79.99, "weight": 0.06},
}

BASE_TEMP_F = 65.0  # Reference for HDD/CDD


class OpenMeteoClient:
    """Fetches historical daily weather data for energy modelling.

    Uses the Open-Meteo Archive API for free, reliable historical weather data
    covering 1940–present at daily resolution.  Provides temperature (HDD/CDD),
    wind speed at 100 m hub height, solar radiation, and cloud cover.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Daily variables for temperature-based demand modelling
    TEMP_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"

    # Daily variables for wind generation modelling
    WIND_VARS = (
        "windspeed_10m_max,windgusts_10m_max,"
        "winddirection_10m_dominant"
    )

    # Daily variables for solar generation modelling
    # Note: direct_radiation_sum and diffuse_radiation_sum are NOT available
    # in the archive API — use cloud_cover_mean as a clear-sky proxy instead.
    SOLAR_VARS = (
        "shortwave_radiation_sum,sunshine_duration,"
        "cloud_cover_mean,precipitation_sum"
    )

    def __init__(self, cities: dict | None = None) -> None:
        self.cities = cities or CITIES
        self.session = requests.Session()
        logger.info("OpenMeteoClient initialised with %d cities", len(self.cities))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    def _fetch_city(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Fetch daily temperature for one location."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": self.TEMP_VARS,
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        resp = self.session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if not daily or not daily.get("time"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "tavg": daily.get("temperature_2m_mean"),
            "tmax": daily.get("temperature_2m_max"),
            "tmin": daily.get("temperature_2m_min"),
        }).set_index("date")
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    def _fetch_city_wind_solar(
        self, lat: float, lon: float, start: str, end: str,
    ) -> pd.DataFrame:
        """Fetch daily wind and solar variables for one location."""
        all_vars = f"{self.TEMP_VARS},{self.WIND_VARS},{self.SOLAR_VARS}"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": all_vars,
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        resp = self.session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if not daily or not daily.get("time"):
            return pd.DataFrame()

        df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})

        # Temperature
        df["tavg"] = daily.get("temperature_2m_mean")
        df["tmax"] = daily.get("temperature_2m_max")
        df["tmin"] = daily.get("temperature_2m_min")

        # Wind (10 m — Open-Meteo archive daily only has 10m, not 100m)
        df["wind_speed_max"] = daily.get("windspeed_10m_max")
        df["wind_gusts_max"] = daily.get("windgusts_10m_max")
        df["wind_dir_dominant"] = daily.get("winddirection_10m_dominant")

        # Solar radiation (MJ/m²) and cloud cover
        df["shortwave_rad"] = daily.get("shortwave_radiation_sum")
        df["cloud_cover"] = daily.get("cloud_cover_mean")
        df["precipitation"] = daily.get("precipitation_sum")
        df["sunshine_hours"] = daily.get("sunshine_duration")
        # Convert sunshine_duration from seconds to hours if present
        if df["sunshine_hours"] is not None:
            df["sunshine_hours"] = pd.to_numeric(df["sunshine_hours"], errors="coerce") / 3600.0

        return df.set_index("date")

    def fetch_weather(
        self,
        start: str = "2010-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch population-weighted daily temperatures for all cities.

        Returns a DataFrame indexed by date with columns:
        ``tavg``, ``tmax``, ``tmin`` (population-weighted averages),
        ``hdd``, ``cdd``, and per-city temps.
        """
        if end is None:
            end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        city_dfs = {}
        for name, info in self.cities.items():
            logger.info("Fetching weather for %s (%.2f, %.2f)", name, info["lat"], info["lon"])
            try:
                df = self._fetch_city(info["lat"], info["lon"], start, end)
                if len(df) > 0:
                    city_dfs[name] = df
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", name, exc)

        if not city_dfs:
            return pd.DataFrame()

        # Population-weighted average
        weights = {name: self.cities[name]["weight"] for name in city_dfs}
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        combined = None
        for name, df in city_dfs.items():
            w = weights[name]
            if combined is None:
                combined = df[["tavg", "tmax", "tmin"]] * w
            else:
                combined = combined.add(df[["tavg", "tmax", "tmin"]] * w, fill_value=0)

        combined = combined.dropna()

        # HDD and CDD
        combined["hdd"] = (BASE_TEMP_F - combined["tavg"]).clip(lower=0)
        combined["cdd"] = (combined["tavg"] - BASE_TEMP_F).clip(lower=0)

        # Rolling weather features
        combined["hdd_7d"] = combined["hdd"].rolling(7).mean()
        combined["cdd_7d"] = combined["cdd"].rolling(7).mean()
        combined["hdd_30d"] = combined["hdd"].rolling(30).mean()
        combined["cdd_30d"] = combined["cdd"].rolling(30).mean()
        combined["temp_range"] = combined["tmax"] - combined["tmin"]
        combined["temp_change_1d"] = combined["tavg"].diff()
        combined["temp_change_7d"] = combined["tavg"].diff(7)

        # Store per-city temps for regional analysis
        for name, df in city_dfs.items():
            combined[f"temp_{name}"] = df["tavg"]

        logger.info("Weather data: %d days, %d columns", len(combined), len(combined.columns))
        return combined

    def fetch_wind_solar_weather(
        self,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch population-weighted daily wind + solar + temperature weather.

        Returns a DataFrame indexed by date with columns for temperature,
        wind speed, wind gusts, wind direction, shortwave/direct/diffuse
        radiation, sunshine hours, HDD, CDD, and derived features.

        Args:
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.  Defaults to 5 days ago.

        Returns:
            DataFrame with all weather variables needed for wind/solar
            generation forecasting plus demand modelling.
        """
        if end is None:
            end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        city_dfs: dict[str, pd.DataFrame] = {}
        for name, info in self.cities.items():
            logger.info(
                "Fetching wind+solar weather for %s (%.2f, %.2f)",
                name, info["lat"], info["lon"],
            )
            try:
                df = self._fetch_city_wind_solar(info["lat"], info["lon"], start, end)
                if len(df) > 0:
                    city_dfs[name] = df
            except Exception as exc:
                logger.warning("Failed to fetch wind/solar weather for %s: %s", name, exc)
            time.sleep(2)  # Respect rate limits

        if not city_dfs:
            return pd.DataFrame()

        # Build population weights
        weights = {name: self.cities[name]["weight"] for name in city_dfs}
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        # Columns to population-weight
        weight_cols = [
            "tavg", "tmax", "tmin",
            "wind_speed_max", "wind_gusts_max",
            "shortwave_rad", "cloud_cover", "precipitation", "sunshine_hours",
        ]

        combined = None
        for name, df in city_dfs.items():
            w = weights[name]
            numerics = df[[c for c in weight_cols if c in df.columns]].apply(
                pd.to_numeric, errors="coerce"
            )
            if combined is None:
                combined = numerics * w
            else:
                combined = combined.add(numerics * w, fill_value=0)

        if combined is None:
            return pd.DataFrame()

        combined = combined.dropna(how="all")

        # Wind direction: circular mean (use first city with most weight as proxy)
        # Circular averaging is complex; use dominant city direction
        top_city = max(weights, key=weights.get)
        if "wind_dir_dominant" in city_dfs[top_city].columns:
            combined["wind_dir_dominant"] = pd.to_numeric(
                city_dfs[top_city]["wind_dir_dominant"], errors="coerce"
            )

        # HDD and CDD
        combined["hdd"] = (BASE_TEMP_F - combined["tavg"]).clip(lower=0)
        combined["cdd"] = (combined["tavg"] - BASE_TEMP_F).clip(lower=0)

        # Rolling temperature features
        combined["hdd_7d"] = combined["hdd"].rolling(7).mean()
        combined["cdd_7d"] = combined["cdd"].rolling(7).mean()
        combined["temp_range"] = combined["tmax"] - combined["tmin"]
        combined["temp_change_1d"] = combined["tavg"].diff()

        # Wind-derived features
        if "wind_speed_max" in combined.columns:
            combined["wind_speed_7d"] = combined["wind_speed_max"].rolling(7).mean()
            combined["wind_speed_14d"] = combined["wind_speed_max"].rolling(14).mean()
            combined["wind_speed_change_1d"] = combined["wind_speed_max"].diff()
            combined["wind_speed_change_7d"] = combined["wind_speed_max"].diff(7)
            combined["wind_volatility_7d"] = combined["wind_speed_max"].rolling(7).std()

        # Wind gusts features
        if "wind_gusts_max" in combined.columns:
            combined["gust_ratio"] = (
                combined["wind_gusts_max"] / combined["wind_speed_max"].clip(lower=0.1)
            )

        # Solar-derived features
        if "shortwave_rad" in combined.columns:
            combined["solar_rad_7d"] = combined["shortwave_rad"].rolling(7).mean()
            combined["solar_rad_14d"] = combined["shortwave_rad"].rolling(14).mean()
            combined["solar_change_1d"] = combined["shortwave_rad"].diff()
            # Clear sky indicator (low cloud cover = more direct radiation)
            if "cloud_cover" in combined.columns:
                combined["clear_sky_index"] = (
                    1.0 - combined["cloud_cover"] / 100.0
                ).clip(0, 1)
            # Rain flag (precipitation reduces solar output)
            if "precipitation" in combined.columns:
                combined["rain_flag"] = (combined["precipitation"] > 1.0).astype(int)

        # Store per-city wind speeds for regional analysis
        for name, df in city_dfs.items():
            if "wind_speed_max" in df.columns:
                combined[f"wind_{name}"] = pd.to_numeric(
                    df["wind_speed_max"], errors="coerce"
                )

        logger.info(
            "Wind+solar weather data: %d days, %d columns",
            len(combined), len(combined.columns),
        )
        return combined
