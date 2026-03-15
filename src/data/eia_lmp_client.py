"""EIA regional electricity demand client.

Fetches hourly electricity demand from the EIA Open Data API v2
for major US ISOs/RTOs and aggregates to daily metrics.

Endpoint: /electricity/rto/region-data/data/
Series: Demand (value-id ``"D"``).
Coverage: ~2019 to present, hourly → aggregated to daily.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"

# Regions that have demand data on EIA
DEMAND_REGIONS = {
    "PJM": "PJM Interconnection",
    "MISO": "Midcontinent ISO",
    "NYIS": "New York ISO",
    "ISNE": "ISO New England",
}


class EIADemandClient:
    """Fetches regional electricity demand from EIA API v2.

    Retrieves hourly demand data and aggregates to daily metrics
    (mean, peak, off-peak, max, min, volatility).
    """

    ENDPOINT = "/electricity/rto/region-data/data/"
    PAGE_SIZE = 5000

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key required. Set EIA_API_KEY env var or pass api_key."
            )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params["api_key"] = self.api_key
        url = f"{EIA_BASE_URL}{self.ENDPOINT}"
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def fetch_hourly_demand(
        self,
        region: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch hourly demand for a region.

        Args:
            region: ISO code (PJM, MISO, NYIS, ISNE).
            start: Start date YYYY-MM-DD.
            end: End date YYYY-MM-DD.

        Returns:
            DataFrame indexed by datetime with ``demand_mwh`` column.
        """
        all_records: list[dict] = []
        offset = 0

        while True:
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": region,
                "facets[type][]": "D",  # Day-ahead demand-weighted LMP
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": self.PAGE_SIZE,
            }
            try:
                data = self._get(params)
            except Exception as exc:
                logger.error("EIA demand request failed (%s, offset=%d): %s",
                             region, offset, exc)
                break

            records = data.get("response", {}).get("data", [])
            if not records:
                break
            all_records.extend(records)
            total = int(data.get("response", {}).get("total", 0))
            offset += self.PAGE_SIZE
            if offset >= total:
                break

        if not all_records:
            logger.warning("No demand data for %s (%s – %s)", region, start, end)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["datetime"] = pd.to_datetime(df["period"])
        df["demand_mwh"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["datetime", "demand_mwh"]].set_index("datetime").sort_index().dropna()
        logger.info("Fetched %d hourly demand records for %s", len(df), region)
        return df

    def fetch_daily_demand(
        self,
        regions: list[str] | None = None,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily demand for multiple regions.

        Aggregates hourly demand to daily: mean, peak, off-peak,
        max, min, volatility, and on-peak/off-peak spread.

        Args:
            regions: ISO codes. Defaults to all four Eastern ISOs.
            start: Start date YYYY-MM-DD.
            end: End date YYYY-MM-DD. Defaults to 5 days ago.

        Returns:
            Daily DataFrame with columns like ``PJM_demand_mean``,
            ``PJM_demand_peak``, ``PJM_demand_spread`` for each region.
        """
        if regions is None:
            regions = list(DEMAND_REGIONS.keys())
        if end is None:
            end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        combined = None
        for region in regions:
            logger.info("Fetching daily demand for %s (%s – %s)", region, start, end)
            try:
                hourly = self.fetch_hourly_demand(region, start, end)
                if hourly.empty:
                    continue
                daily = self._aggregate_daily(hourly, region)
                if combined is None:
                    combined = daily
                else:
                    combined = combined.join(daily, how="outer")
            except Exception as exc:
                logger.error("Failed to fetch demand for %s: %s", region, exc)

        if combined is None:
            return pd.DataFrame()

        combined = combined.sort_index().ffill(limit=2)
        logger.info("Daily demand: %d days, %d columns, regions=%s",
                     len(combined), len(combined.columns), regions)
        return combined

    @staticmethod
    def _aggregate_daily(hourly: pd.DataFrame, region: str) -> pd.DataFrame:
        """Aggregate hourly demand to daily metrics.

        On-peak hours: 7–22 (HE 7–22 inclusive).
        Off-peak hours: 0–6, 23.
        """
        prefix = region

        hourly = hourly.copy()
        hourly["hour"] = hourly.index.hour
        hourly["is_peak"] = hourly["hour"].between(7, 22)

        daily_mean = hourly["demand_mwh"].resample("D").mean()
        daily_peak = hourly.loc[hourly["is_peak"], "demand_mwh"].resample("D").mean()
        daily_offpeak = hourly.loc[~hourly["is_peak"], "demand_mwh"].resample("D").mean()
        daily_max = hourly["demand_mwh"].resample("D").max()
        daily_min = hourly["demand_mwh"].resample("D").min()
        daily_vol = hourly["demand_mwh"].resample("D").std()

        result = pd.DataFrame({
            f"{prefix}_demand_mean": daily_mean,
            f"{prefix}_demand_peak": daily_peak,
            f"{prefix}_demand_offpeak": daily_offpeak,
            f"{prefix}_demand_max": daily_max,
            f"{prefix}_demand_min": daily_min,
            f"{prefix}_demand_vol": daily_vol,
        })

        # Peak/off-peak spread
        result[f"{prefix}_demand_spread"] = (
            result[f"{prefix}_demand_peak"] - result[f"{prefix}_demand_offpeak"]
        )

        return result
