"""EIA electricity generation client by fuel type.

Fetches hourly electricity generation from the EIA Open Data API v2,
broken down by fuel type (wind, solar, natural gas, nuclear, coal, hydro).
Covers Eastern/Central US grid operators: PJM, MISO, NYIS, ISNE.

Key use cases:
  - Wind / solar generation forecasting (weather → renewable MWh)
  - Net load calculation: demand − wind − solar
  - Fuel-mix analytics for gas-for-power demand estimation
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"

# Major US ISO/RTO regions (same as demand client)
REGIONS = {
    "PJM": "PJM Interconnection",
    "MISO": "Midcontinent ISO",
    "NYIS": "New York ISO",
    "ISNE": "ISO New England",
    "CISO": "California ISO",
    "ERCO": "Electric Reliability Council of Texas",
    "SWPP": "Southwest Power Pool",
}

# EIA fuel-type codes
FUEL_TYPES = {
    "WND": "Wind",
    "SUN": "Solar",
    "NG":  "Natural Gas",
    "NUC": "Nuclear",
    "COL": "Coal",
    "WAT": "Hydro",
    "OTH": "Other",
}


class EIAGenerationClient:
    """Fetches regional electricity generation by fuel type from EIA API v2.

    Provides hourly generation data aggregated to daily frequency with
    total MWh, peak MW, min MW, avg MW, capacity factor, ramp rate,
    and variability metrics for each region × fuel type.

    Attributes:
        api_key: EIA API key.
        base_url: EIA API v2 base URL.
        session: Persistent HTTP session with connection pooling.
    """

    ENDPOINT = "/electricity/rto/fuel-type-data/data/"
    PAGE_SIZE = 5000

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = EIA_BASE_URL,
    ) -> None:
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key required.  Set EIA_API_KEY env var or pass api_key."
            )
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        logger.info("EIAGenerationClient initialised (base=%s)", self.base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make a paginated GET request to the EIA API."""
        params["api_key"] = self.api_key
        url = f"{self.base_url}{self.ENDPOINT}"
        logger.debug("EIA GET %s", url)
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _build_params(
        self,
        region: str,
        fuel_type: str,
        start: str,
        end: str,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Build query parameters for a generation data request."""
        return {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[fueltype][]": fuel_type,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": self.PAGE_SIZE,
        }

    def fetch_hourly_generation(
        self,
        region: str,
        fuel_type: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch hourly generation for a region × fuel type.

        Handles pagination automatically — EIA API caps each response
        at 5,000 records.

        Args:
            region: ISO/RTO code (``"PJM"``, ``"MISO"``, ``"NYIS"``, ``"ISNE"``).
            fuel_type: EIA fuel code (``"WND"``, ``"SUN"``, ``"NG"``, etc.).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.

        Returns:
            DataFrame indexed by datetime with ``gen_mw`` column.
        """
        all_records: list[dict] = []
        offset = 0

        while True:
            params = self._build_params(region, fuel_type, start, end, offset)
            try:
                data = self._get(params)
            except Exception as exc:
                logger.error(
                    "EIA generation request failed (region=%s, fuel=%s, offset=%d): %s",
                    region, fuel_type, offset, exc,
                )
                break

            records = data.get("response", {}).get("data", [])
            if not records:
                break
            all_records.extend(records)
            total = int(data.get("response", {}).get("total", 0))
            offset += self.PAGE_SIZE
            logger.debug(
                "Fetched %d / %d records for %s/%s",
                len(all_records), total, region, fuel_type,
            )
            if offset >= total:
                break

        if not all_records:
            logger.warning(
                "No generation data for %s/%s (%s – %s)",
                region, fuel_type, start, end,
            )
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["datetime"] = pd.to_datetime(df["period"])
        df["gen_mw"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["datetime", "gen_mw"]].set_index("datetime").sort_index()
        df = df.dropna()
        logger.info(
            "Fetched %d hourly records for %s/%s",
            len(df), region, fuel_type,
        )
        return df

    @staticmethod
    def _aggregate_daily(
        hourly: pd.DataFrame,
        region: str,
        fuel_type: str,
    ) -> pd.DataFrame:
        """Aggregate hourly generation to daily metrics.

        Computes for each day:
            - ``{prefix}_total_mwh``: Total generation (MWh).
            - ``{prefix}_peak_mw``:   Maximum hourly output (MW).
            - ``{prefix}_min_mw``:    Minimum hourly output (MW).
            - ``{prefix}_avg_mw``:    Mean hourly output (MW).
            - ``{prefix}_ramp_mw``:   Max hour-over-hour ramp (MW) — volatility.
            - ``{prefix}_cf``:        Capacity factor proxy (avg/peak).
            - ``{prefix}_variability``: Std(hourly) / mean — intermittency.

        Args:
            hourly: Hourly generation DataFrame with ``gen_mw`` column.
            region: Region code (e.g. ``"PJM"``).
            fuel_type: Fuel code (e.g. ``"WND"``).

        Returns:
            Daily-indexed DataFrame.
        """
        prefix = f"{region.lower()}_{fuel_type.lower()}"

        daily = hourly.resample("D").agg(
            total=("gen_mw", "sum"),
            peak=("gen_mw", "max"),
            minimum=("gen_mw", "min"),
            avg=("gen_mw", "mean"),
            std=("gen_mw", "std"),
            hours=("gen_mw", "count"),
        )
        # Only keep days with ≥20 hours of data
        daily = daily[daily["hours"] >= 20]

        # Capacity factor proxy: avg / peak (0–1 range)
        daily[f"{prefix}_cf"] = (daily["avg"] / daily["peak"]).clip(0, 1)

        # Variability (coefficient of variation) — high for wind/solar
        daily[f"{prefix}_variability"] = (daily["std"] / daily["avg"]).clip(0, 10)

        # Hourly ramp rate: max absolute hour-over-hour change
        ramp = hourly["gen_mw"].diff().abs()
        daily_ramp = ramp.resample("D").max()
        daily[f"{prefix}_ramp_mw"] = daily_ramp

        daily = daily.rename(columns={
            "total": f"{prefix}_total_mwh",
            "peak": f"{prefix}_peak_mw",
            "minimum": f"{prefix}_min_mw",
            "avg": f"{prefix}_avg_mw",
        }).drop(columns=["hours", "std"])

        return daily

    def fetch_daily_generation(
        self,
        regions: list[str] | None = None,
        fuel_types: list[str] | None = None,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily generation for multiple regions × fuel types.

        Args:
            regions: List of ISO/RTO codes. Defaults to all four Eastern regions.
            fuel_types: List of fuel codes. Defaults to WND, SUN, NG.
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.  Defaults to 5 days ago.

        Returns:
            Daily DataFrame with per-region-per-fuel columns.
        """
        if regions is None:
            regions = list(REGIONS.keys())
        if fuel_types is None:
            fuel_types = ["WND", "SUN", "NG"]
        if end is None:
            end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        combined = None
        for region in regions:
            for fuel in fuel_types:
                logger.info(
                    "Fetching daily generation for %s/%s (%s — %s)",
                    region, fuel, start, end,
                )
                try:
                    hourly = self.fetch_hourly_generation(region, fuel, start, end)
                    if hourly.empty:
                        logger.warning("No data for %s/%s — skipping", region, fuel)
                        continue
                    daily = self._aggregate_daily(hourly, region, fuel)
                    if combined is None:
                        combined = daily
                    else:
                        combined = combined.join(daily, how="outer")
                except Exception as exc:
                    logger.error("Failed to fetch %s/%s: %s", region, fuel, exc)

        if combined is None:
            logger.warning("No generation data retrieved")
            return pd.DataFrame()

        # Compute aggregate wind and solar totals across all regions
        wind_cols = [c for c in combined.columns if "_wnd_total_mwh" in c]
        solar_cols = [c for c in combined.columns if "_sun_total_mwh" in c]
        gas_cols = [c for c in combined.columns if "_ng_total_mwh" in c]

        if wind_cols:
            combined["east_wind_total_mwh"] = combined[wind_cols].sum(axis=1)
        if solar_cols:
            combined["east_solar_total_mwh"] = combined[solar_cols].sum(axis=1)
        if gas_cols:
            combined["east_gas_gen_total_mwh"] = combined[gas_cols].sum(axis=1)

        # Forward-fill small gaps (missing hours in a day)
        combined = combined.sort_index()
        combined = combined.ffill(limit=2)

        logger.info(
            "Daily generation: %d days, %d columns, regions=%s, fuels=%s",
            len(combined), len(combined.columns), regions, fuel_types,
        )
        return combined

    def fetch_wind_generation(
        self,
        regions: list[str] | None = None,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch only wind generation data.

        Args:
            regions: List of ISO/RTO codes.
            start: Start date.
            end: End date.

        Returns:
            Daily wind generation DataFrame.
        """
        return self.fetch_daily_generation(
            regions=regions, fuel_types=["WND"], start=start, end=end,
        )

    def fetch_solar_generation(
        self,
        regions: list[str] | None = None,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch only solar generation data.

        Args:
            regions: List of ISO/RTO codes.
            start: Start date.
            end: End date.

        Returns:
            Daily solar generation DataFrame.
        """
        return self.fetch_daily_generation(
            regions=regions, fuel_types=["SUN"], start=start, end=end,
        )
