"""EIA electricity demand client for power load forecasting.

Fetches regional hourly electricity demand from the EIA Open Data API v2
and aggregates to daily frequency.  Covers key Eastern/Central US grid
operators: PJM, MISO, NYIS, ISNE.

These regions align with the major power trading hubs in the Eastern
Interconnection and are the primary markets for power & gas trading desks.
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

# Eastern/Central US ISO/RTO regions
REGIONS = {
    "PJM": "PJM Interconnection",      # Mid-Atlantic + Midwest, largest US RTO
    "MISO": "Midcontinent ISO",         # 15 US states, major gas-for-power market
    "NYIS": "New York ISO",             # New York state
    "ISNE": "ISO New England",          # 6 New England states
}


class EIADemandClient:
    """Fetches regional electricity demand from EIA Open Data API v2.

    Provides hourly demand data aggregated to daily frequency with
    peak, min, mean, and total metrics for each region.

    Attributes:
        api_key: EIA API key.
        base_url: EIA API v2 base URL.
        session: Persistent HTTP session with connection pooling.
    """

    ENDPOINT = "/electricity/rto/region-data/data/"
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
        logger.info("EIADemandClient initialised (base=%s)", self.base_url)

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
        start: str,
        end: str,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Build query parameters for a demand data request."""
        return {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",      # D = demand
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": self.PAGE_SIZE,
        }

    def fetch_hourly_demand(
        self,
        region: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch hourly electricity demand for a single region.

        Handles pagination automatically — the EIA API caps each response
        at 5 000 records.

        Args:
            region: ISO/RTO code (``"PJM"``, ``"MISO"``, ``"NYIS"``, ``"ISNE"``).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.

        Returns:
            DataFrame indexed by datetime with ``demand_mw`` column.
        """
        all_records: list[dict] = []
        offset = 0

        while True:
            params = self._build_params(region, start, end, offset)
            try:
                data = self._get(params)
            except Exception as exc:
                logger.error("EIA demand request failed (region=%s, offset=%d): %s",
                             region, offset, exc)
                break

            records = data.get("response", {}).get("data", [])
            if not records:
                break
            all_records.extend(records)
            total = int(data.get("response", {}).get("total", 0))
            offset += self.PAGE_SIZE
            logger.debug("Fetched %d / %d records for %s", len(all_records), total, region)
            if offset >= total:
                break

        if not all_records:
            logger.warning("No demand data for %s (%s – %s)", region, start, end)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["datetime"] = pd.to_datetime(df["period"])
        df["demand_mw"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["datetime", "demand_mw"]].set_index("datetime").sort_index()
        df = df.dropna()
        logger.info("Fetched %d hourly records for %s", len(df), region)
        return df

    @staticmethod
    def _aggregate_daily(hourly: pd.DataFrame, region: str) -> pd.DataFrame:
        """Aggregate hourly demand to daily metrics.

        Computes for each day:
            - ``{region}_total_mwh``: Total energy consumed (MWh).
            - ``{region}_peak_mw``:   Maximum hourly demand (MW).
            - ``{region}_min_mw``:    Minimum hourly demand (MW).
            - ``{region}_avg_mw``:    Mean hourly demand (MW).
            - ``{region}_load_factor``: avg / peak — grid utilisation metric.

        Args:
            hourly: Hourly demand DataFrame.
            region: Region code used as column prefix.

        Returns:
            Daily-indexed DataFrame.
        """
        prefix = region.lower()
        daily = hourly.resample("D").agg(
            total=("demand_mw", "sum"),
            peak=("demand_mw", "max"),
            minimum=("demand_mw", "min"),
            avg=("demand_mw", "mean"),
            hours=("demand_mw", "count"),
        )
        # Only keep days with ≥20 hours of data (filter partial days)
        daily = daily[daily["hours"] >= 20]
        daily[f"{prefix}_load_factor"] = daily["avg"] / daily["peak"]
        daily = daily.rename(columns={
            "total": f"{prefix}_total_mwh",
            "peak": f"{prefix}_peak_mw",
            "minimum": f"{prefix}_min_mw",
            "avg": f"{prefix}_avg_mw",
        }).drop(columns=["hours"])
        return daily

    def fetch_daily_demand(
        self,
        regions: list[str] | None = None,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily electricity demand for multiple regions.

        Fetches hourly data region-by-region, aggregates to daily, and
        merges into a single DataFrame.  Also computes a total Eastern
        Interconnection demand metric.

        Args:
            regions: List of ISO/RTO codes.  Defaults to all four Eastern
                regions (PJM, MISO, NYIS, ISNE).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.  Defaults to 5 days ago.

        Returns:
            Daily DataFrame with per-region columns plus aggregate totals.
        """
        if regions is None:
            regions = list(REGIONS.keys())
        if end is None:
            end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        combined = None
        for region in regions:
            logger.info("Fetching daily demand for %s (%s — %s)",
                        region, start, end)
            hourly = self.fetch_hourly_demand(region, start, end)
            if hourly.empty:
                continue
            daily = self._aggregate_daily(hourly, region)
            if combined is None:
                combined = daily
            else:
                combined = combined.join(daily, how="inner")  # inner = only days with ALL regions

        if combined is None:
            return pd.DataFrame()

        # Aggregate: total Eastern Interconnection demand
        total_cols = [c for c in combined.columns if c.endswith("_total_mwh")]
        peak_cols = [c for c in combined.columns if c.endswith("_peak_mw")]
        if total_cols:
            combined["east_total_mwh"] = combined[total_cols].sum(axis=1)
        if peak_cols:
            combined["east_peak_mw"] = combined[peak_cols].sum(axis=1)

        # Remove outliers: IQR-based filter on total demand
        if "east_total_mwh" in combined.columns:
            q1 = combined["east_total_mwh"].quantile(0.01)
            q99 = combined["east_total_mwh"].quantile(0.99)
            iqr = q99 - q1
            lower = q1 - 2 * iqr
            upper = q99 + 2 * iqr
            n_before = len(combined)
            combined = combined[
                (combined["east_total_mwh"] >= lower)
                & (combined["east_total_mwh"] <= upper)
            ]
            n_removed = n_before - len(combined)
            if n_removed > 0:
                logger.warning(
                    "Removed %d outlier days (demand outside [%.0f, %.0f] MWh)",
                    n_removed, lower, upper,
                )

        combined = combined.sort_index()
        logger.info(
            "Daily demand: %d days, %d columns, regions=%s",
            len(combined), len(combined.columns), regions,
        )
        return combined
