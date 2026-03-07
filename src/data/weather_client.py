"""NOAA weather data client.

Fetches Heating Degree Day (HDD) and Cooling Degree Day (CDD) data from the
NOAA Climate Data Online (CDO) API.  Weather demand signals are important
drivers of natural gas and heating oil prices.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# Key US weather stations for energy demand modelling
DEFAULT_STATIONS = [
    "GHCND:USW00094728",  # New York (Central Park)
    "GHCND:USW00094846",  # Chicago O'Hare
    "GHCND:USW00013904",  # Houston Hobby
    "GHCND:USW00023174",  # Los Angeles
    "GHCND:USW00014739",  # Boston Logan
]


class WeatherClient:
    """Client for NOAA Climate Data Online (CDO) API.

    Retrieves daily temperature data and computes Heating/Cooling Degree Days
    (HDD/CDD) which are key demand indicators for natural gas and heating oil.

    Attributes:
        api_key: NOAA CDO API token.
        base_url: Base URL for the CDO API.
        stations: List of weather station IDs to aggregate.
    """

    BASE_TEMP_F = 65.0  # Reference temperature for HDD/CDD calculation (°F)

    def __init__(
        self,
        api_key: str | None = None,
        stations: list[str] | None = None,
        base_url: str = NOAA_BASE_URL,
    ) -> None:
        """Initialise the weather client.

        Args:
            api_key: NOAA CDO API token.  If ``None``, reads from
                ``NOAA_API_KEY`` environment variable.
            stations: List of NOAA station IDs.  Defaults to key US cities.
            base_url: Override for the NOAA CDO API base URL.

        Raises:
            ValueError: If no API key is provided.
        """
        self.api_key = api_key or os.environ.get("NOAA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NOAA API key is required. Set the NOAA_API_KEY environment variable "
                "or pass api_key to WeatherClient."
            )
        self.stations = stations or DEFAULT_STATIONS
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"token": self.api_key})
        logger.info("WeatherClient initialised with %d stations", len(self.stations))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a GET request to the NOAA CDO API.

        Args:
            endpoint: API path (e.g. ``"/data"``).
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses.
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug("NOAA GET %s params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def fetch_temperature(
        self,
        start: str | None = None,
        end: str | None = None,
        station_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily average temperature data (TAVG) for given stations.

        Args:
            start: Start date in ``YYYY-MM-DD`` format.  Defaults to 365 days ago.
            end: End date in ``YYYY-MM-DD`` format.  Defaults to today.
            station_ids: List of NOAA station IDs.  Uses ``self.stations`` if None.

        Returns:
            ``pd.DataFrame`` with columns ``station``, ``date``, ``tavg_f``.
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        stations = station_ids or self.stations
        all_records: list[dict] = []

        for station in stations:
            params = {
                "datasetid": "GHCND",
                "stationid": station,
                "datatypeid": "TAVG",
                "startdate": start,
                "enddate": end,
                "units": "standard",  # Fahrenheit
                "limit": 1000,
            }
            try:
                data = self._get("/data", params)
                records = data.get("results", [])
                for r in records:
                    all_records.append(
                        {
                            "station": station,
                            "date": pd.to_datetime(r["date"]),
                            "tavg_f": float(r["value"]) / 10.0,  # GHCND stores as tenths
                        }
                    )
                logger.debug("Fetched %d temperature records for station %s", len(records), station)
            except Exception as exc:
                logger.warning("Failed to fetch data for station %s: %s", station, exc)

        if not all_records:
            logger.warning("No temperature data fetched for any station")
            return pd.DataFrame(columns=["station", "date", "tavg_f"])

        return pd.DataFrame(all_records)

    def compute_hdd_cdd(
        self,
        temperature_df: pd.DataFrame,
        base_temp: float = BASE_TEMP_F,
    ) -> pd.DataFrame:
        """Compute daily Heating and Cooling Degree Days.

        HDD = max(0, base_temp - tavg)  — drives natural gas / heating oil demand.
        CDD = max(0, tavg - base_temp)  — drives electricity / summer cooling demand.

        Args:
            temperature_df: DataFrame from ``fetch_temperature`` with ``tavg_f`` column.
            base_temp: Reference temperature in °F (default 65°F).

        Returns:
            ``pd.DataFrame`` with daily ``hdd`` and ``cdd`` columns, averaged across
            all stations, indexed by date.
        """
        df = temperature_df.copy()
        df["hdd"] = (base_temp - df["tavg_f"]).clip(lower=0.0)
        df["cdd"] = (df["tavg_f"] - base_temp).clip(lower=0.0)
        # Average across all stations per day
        daily = df.groupby("date")[["hdd", "cdd"]].mean()
        daily.index.name = "date"
        logger.info("Computed HDD/CDD for %d days", len(daily))
        return daily

    def fetch_hdd_cdd(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Convenience method: fetch temperature and return computed HDD/CDD.

        Args:
            start: Start date string.
            end: End date string.

        Returns:
            ``pd.DataFrame`` with ``hdd`` and ``cdd`` columns indexed by date.
        """
        temps = self.fetch_temperature(start=start, end=end)
        if temps.empty:
            return pd.DataFrame(columns=["hdd", "cdd"])
        return self.compute_hdd_cdd(temps)
