"""Open-Meteo historical weather client for energy demand modelling.

Fetches daily temperature data from the free Open-Meteo API for key
Eastern/Central US population centres.  Computes Heating Degree Days (HDD)
and Cooling Degree Days (CDD) which are the primary drivers of electricity
and natural gas demand.

No API key is required.  Rate limit: ~10,000 requests/day.
"""
from __future__ import annotations

import logging
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
    """Fetches historical daily temperatures and computes HDD/CDD.

    Uses the Open-Meteo Archive API for free, reliable historical weather data
    covering 1940–present at daily resolution.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
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
