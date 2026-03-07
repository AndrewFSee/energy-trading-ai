"""FRED (Federal Reserve Economic Data) API client.

Retrieves macroeconomic indicators relevant to energy markets including
interest rates, CPI, GDP, industrial production, and US dollar strength.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class FREDClient:
    """Client for the FRED (Federal Reserve Economic Data) API.

    Uses the ``fredapi`` library as the underlying transport.  Falls back
    to a direct HTTP implementation if ``fredapi`` is unavailable.

    Attributes:
        api_key: FRED API key.
        fred: Underlying ``fredapi.Fred`` instance.
    """

    # Default macro series used in feature engineering
    DEFAULT_SERIES: dict[str, str] = {
        "fed_funds_rate": "FEDFUNDS",
        "us_cpi": "CPIAUCSL",
        "us_gdp": "GDP",
        "us_unemployment": "UNRATE",
        "industrial_production": "INDPRO",
        "ten_year_treasury": "GS10",
        "two_year_treasury": "GS2",
        "yield_spread_10y2y": None,  # computed from GS10 - GS2
        "trade_weighted_usd": "DTWEXBGS",
        "m2_money_supply": "M2SL",
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the FRED client.

        Args:
            api_key: FRED API key.  If ``None``, reads from the
                ``FRED_API_KEY`` environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key is required. Set the FRED_API_KEY environment variable "
                "or pass api_key to FREDClient."
            )
        self._init_client()

    def _init_client(self) -> None:
        """Initialise the underlying fredapi client."""
        try:
            from fredapi import Fred

            self.fred = Fred(api_key=self.api_key)
            logger.info("FREDClient initialised with fredapi")
        except ImportError:
            logger.warning("fredapi not installed — falling back to direct HTTP")
            self.fred = None

    def fetch_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series identifier (e.g. ``"FEDFUNDS"``).
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.

        Returns:
            ``pd.Series`` with a ``DatetimeIndex``.
        """
        if self.fred is not None:
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                series.name = series_id
                logger.info("Fetched FRED series %s (%d obs)", series_id, len(series))
                return series
            except Exception as exc:
                logger.error("Failed to fetch FRED series %s: %s", series_id, exc)
                return pd.Series(dtype=float, name=series_id)

        # Fallback: direct HTTP request
        return self._fetch_series_http(series_id, start, end)

    def _fetch_series_http(
        self,
        series_id: str,
        start: str | None,
        end: str | None,
    ) -> pd.Series:
        """Fetch a FRED series via direct HTTP (fallback).

        Args:
            series_id: FRED series identifier.
            start: Start date string.
            end: End date string.

        Returns:
            ``pd.Series`` with a ``DatetimeIndex``.
        """
        import requests

        url = "https://api.stlouisfed.org/fred/series/observations"
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            records = {
                obs["date"]: float(obs["value"]) for obs in observations if obs["value"] != "."
            }
            series = pd.Series(records, name=series_id)
            series.index = pd.to_datetime(series.index)
            logger.info("Fetched FRED series %s via HTTP (%d obs)", series_id, len(series))
            return series
        except Exception as exc:
            logger.error("HTTP fetch failed for FRED series %s: %s", series_id, exc)
            return pd.Series(dtype=float, name=series_id)

    def fetch_macro_features(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all default macro series and merge into a single DataFrame.

        Args:
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.

        Returns:
            ``pd.DataFrame`` with one column per macro series, merged on date.
        """
        frames: list[pd.Series] = []
        for name, series_id in self.DEFAULT_SERIES.items():
            if series_id is None:
                continue
            series = self.fetch_series(series_id, start=start, end=end)
            series.name = name
            frames.append(series)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        df.index.name = "date"

        # Compute derived features
        if "ten_year_treasury" in df.columns and "two_year_treasury" in df.columns:
            df["yield_spread_10y2y"] = df["ten_year_treasury"] - df["two_year_treasury"]

        # Forward-fill monthly/quarterly series to daily frequency
        df = df.resample("D").last().ffill()
        logger.info("Macro feature DataFrame: shape=%s", df.shape)
        return df
