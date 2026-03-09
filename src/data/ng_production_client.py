"""Natural gas production and drilling activity client.

Aggregates data from two free sources to build a complete picture of
US natural gas supply dynamics:

  1. **EIA** — Monthly marketed NG production (``/natural-gas/prod/sum/data/``).
  2. **FRED** — Weekly Baker Hughes NG rotary rig count (``RNGWELHOL``),
     monthly marketed production (``GASPRODM``).

Baker Hughes rig counts are the single best leading indicator for
NG production (4–6 month lead).  A rising rig count signals future
production growth, which is bearish for NG prices long-term.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"


class NGProductionClient:
    """Fetches NG production fundamentals from EIA and FRED.

    Combines monthly production volumes with weekly rig counts
    to build features for production trend forecasting.

    Attributes:
        eia_api_key: EIA API key.
        fred_api_key: FRED API key.
        session: Persistent HTTP session.
    """

    # EIA endpoints
    EIA_PRODUCTION_ENDPOINT = "/natural-gas/prod/sum/data/"

    # FRED series
    FRED_SERIES = {
        "drilling_index": "IPN213111S",   # Industrial Production: Drilling Oil & Gas Wells (monthly)
    }

    def __init__(
        self,
        eia_api_key: str | None = None,
        fred_api_key: str | None = None,
    ) -> None:
        self.eia_api_key = eia_api_key or os.environ.get("EIA_API_KEY")
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        if not self.eia_api_key:
            raise ValueError("EIA_API_KEY required")
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY required")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        logger.info("NGProductionClient initialised")

    # ------------------------------------------------------------------ #
    #  FRED data (weekly rig counts + monthly production)
    # ------------------------------------------------------------------ #
    def _fetch_fred_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch a FRED series using fredapi library (preferred) or HTTP fallback."""
        # Try fredapi library first (more reliable)
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)
            series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            series.name = series_id
            series.index.name = "date"
            logger.info("Fetched FRED %s via fredapi: %d observations", series_id, len(series))
            return series
        except ImportError:
            logger.debug("fredapi not installed, using HTTP fallback")
        except Exception as exc:
            logger.warning("fredapi failed for %s: %s — trying HTTP", series_id, exc)

        # HTTP fallback
        return self._fetch_fred_series_http(series_id, start, end)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    def _fetch_fred_series_http(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch a FRED series via direct HTTP (fallback)."""
        url = "https://api.stlouisfed.org/fred/series/observations"
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        records = {
            obs["date"]: float(obs["value"])
            for obs in observations
            if obs["value"] != "."
        }
        series = pd.Series(records, name=series_id, dtype=float)
        series.index = pd.to_datetime(series.index)
        series.index.name = "date"
        logger.info("Fetched FRED %s: %d observations", series_id, len(series))
        return series

    def fetch_drilling_activity(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch drilling activity index from FRED.

        Uses Industrial Production: Drilling Oil & Gas Wells (IPN213111S),
        a monthly index that closely tracks rig counts and drilling intensity.
        This is the best freely available proxy for Baker Hughes rig counts.

        Returns a daily-frequency DataFrame (forward-filled from monthly)
        with drilling index levels and derived trend features.

        Args:
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.

        Returns:
            DataFrame with drilling activity features.
        """
        raw = self._fetch_fred_series(
            self.FRED_SERIES["drilling_index"], start, end,
        )
        if raw.empty:
            logger.warning("No drilling activity data from FRED")
            return pd.DataFrame()

        df = raw.to_frame(name="drilling_index")

        # Resample to daily, forward-fill (monthly data)
        df = df.resample("D").last().ffill()

        # Trend features
        df["drill_4w"] = df["drilling_index"].rolling(28).mean()
        df["drill_13w"] = df["drilling_index"].rolling(91).mean()
        df["drill_26w"] = df["drilling_index"].rolling(182).mean()
        df["drill_52w"] = df["drilling_index"].rolling(364).mean()

        # Momentum: change over various periods
        df["drill_change_3m"] = df["drilling_index"].diff(90)
        df["drill_change_6m"] = df["drilling_index"].diff(182)

        # Percentage change
        df["drill_pct_3m"] = df["drilling_index"].pct_change(90)
        df["drill_pct_6m"] = df["drilling_index"].pct_change(182)

        # Drilling index relative to its 52-week range (0–1)
        roll_min = df["drilling_index"].rolling(364).min()
        roll_max = df["drilling_index"].rolling(364).max()
        df["drill_52w_percentile"] = (
            (df["drilling_index"] - roll_min) / (roll_max - roll_min + 1e-6)
        ).clip(0, 1)

        # Acceleration: second derivative
        df["drill_acceleration"] = df["drill_change_3m"].diff(90)

        # Regime: is drilling trending up or down?
        df["drill_uptrend"] = (df["drill_4w"] > df["drill_26w"]).astype(int)

        logger.info(
            "Drilling activity features: %d days, %d columns",
            len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------ #
    #  EIA monthly production
    # ------------------------------------------------------------------ #
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    def _fetch_eia(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make a GET request to the EIA API."""
        params["api_key"] = self.eia_api_key
        url = f"{EIA_BASE_URL}{self.EIA_PRODUCTION_ENDPOINT}"
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def fetch_monthly_production(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch monthly US marketed NG production from EIA.

        Args:
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.

        Returns:
            Daily-frequency DataFrame (forward-filled from monthly) with
            production levels and derived trend features.
        """
        all_records: list[dict] = []
        offset = 0
        page_size = 5000

        while True:
            params = {
                "frequency": "monthly",
                "data[0]": "value",
                "facets[duoarea][]": "NUS",    # National US
                "facets[process][]": "VGM",     # Marketed Production (MMcf)
                "start": start[:7],             # YYYY-MM for monthly
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": page_size,
            }
            if end:
                params["end"] = end[:7]

            try:
                data = self._fetch_eia(params)
            except Exception as exc:
                logger.error("EIA production request failed (offset=%d): %s", offset, exc)
                break

            records = data.get("response", {}).get("data", [])
            if not records:
                break
            all_records.extend(records)
            total = int(data.get("response", {}).get("total", 0))
            offset += page_size
            if offset >= total:
                break

        if not all_records:
            logger.warning("No EIA production data retrieved")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["period"])
        df["production_mmcf"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "production_mmcf"]].set_index("date").sort_index()
        df = df.dropna()

        # Remove duplicates (keep last)
        df = df[~df.index.duplicated(keep="last")]

        # Resample to daily, forward-fill
        df = df.resample("D").last().ffill()

        # Production trend features
        df["prod_3m"] = df["production_mmcf"].rolling(90).mean()
        df["prod_6m"] = df["production_mmcf"].rolling(180).mean()
        df["prod_12m"] = df["production_mmcf"].rolling(365).mean()

        # Momentum
        df["prod_change_3m"] = df["production_mmcf"].diff(90)
        df["prod_change_6m"] = df["production_mmcf"].diff(180)
        df["prod_pct_3m"] = df["production_mmcf"].pct_change(90)
        df["prod_pct_6m"] = df["production_mmcf"].pct_change(180)

        # Production regime
        df["prod_uptrend"] = (df["prod_3m"] > df["prod_12m"]).astype(int)

        logger.info("EIA production features: %d days, %d columns", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------ #
    #  Combined production fundamentals
    # ------------------------------------------------------------------ #
    def fetch_production_fundamentals(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch and merge all NG production fundamental data.

        Combines the FRED drilling activity index with monthly EIA
        production to build a comprehensive supply-side feature set.

        Args:
            start: Start date.
            end: End date.

        Returns:
            Daily DataFrame with drilling activity features, production
            features, and cross-series derived features.
        """
        drill_df = self.fetch_drilling_activity(start, end)
        prod_df = self.fetch_monthly_production(start, end)

        if drill_df.empty and prod_df.empty:
            logger.warning("No production data available")
            return pd.DataFrame()

        if drill_df.empty:
            return prod_df
        if prod_df.empty:
            return drill_df

        # Merge on date
        combined = drill_df.join(prod_df, how="outer").sort_index()
        combined = combined.ffill()

        # Cross-series: drilling-to-production relationship
        if "drilling_index" in combined.columns and "production_mmcf" in combined.columns:
            # Lagged drilling index (to align with production response)
            combined["drill_lag_4m"] = combined["drilling_index"].shift(120)
            combined["drill_lag_6m"] = combined["drilling_index"].shift(180)

            # Production per drilling intensity (efficiency metric)
            drill_safe = combined["drilling_index"].clip(lower=0.01)
            combined["prod_per_drill"] = combined["production_mmcf"] / drill_safe
            combined["prod_per_drill_6m"] = combined["prod_per_drill"].rolling(180).mean()

            # Supply growth signal: rising drilling + rising production
            combined["supply_growth"] = (
                combined["drill_uptrend"].fillna(0)
                + combined["prod_uptrend"].fillna(0)
            ) / 2  # 0 = both declining, 0.5 = mixed, 1.0 = both growing

        combined = combined.dropna(how="all", axis=1)
        logger.info(
            "Production fundamentals: %d days, %d columns",
            len(combined), len(combined.columns),
        )
        return combined
