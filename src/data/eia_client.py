"""EIA (Energy Information Administration) API client.

Fetches energy-specific fundamental data including crude oil and natural gas
storage levels, production figures, and import/export volumes from the
EIA Open Data API v2.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"


class EIAClient:
    """Client for the EIA Open Data API v2.

    Retrieves weekly petroleum and natural gas storage, production,
    and trade statistics.

    Attributes:
        api_key: EIA API key loaded from the ``EIA_API_KEY`` environment
            variable.
        base_url: Base URL for the EIA API.
        session: Persistent ``requests.Session`` for connection pooling.
    """

    # Common EIA series IDs
    SERIES = {
        "crude_storage": "PET.WCRSTUS1.W",
        "nat_gas_storage": "NG.NW2_EPG0_SWO_R48_BCF.W",
        "crude_production": "PET.WCRFPUS2.W",
        "crude_imports": "PET.WCRIMUS2.W",
        "crude_exports": "PET.WCREXUS2.W",
        "refinery_utilization": "PET.WPULEUS3.W",
        "gasoline_demand": "PET.WGFUPUS2.W",
        "distillate_demand": "PET.WDIRPUS2.W",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = EIA_BASE_URL,
    ) -> None:
        """Initialise the EIA client.

        Args:
            api_key: EIA API key.  If ``None``, reads from ``EIA_API_KEY``
                environment variable.
            base_url: Override for the EIA API base URL.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key is required. Set the EIA_API_KEY environment variable "
                "or pass api_key to EIAClient."
            )
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        logger.info("EIAClient initialised (base_url=%s)", self.base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a GET request to the EIA API with retry logic.

        Args:
            endpoint: API endpoint path (e.g. ``"/seriesid/PET.WCRSTUS1.W"``).
            params: Query parameters to include in the request.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            requests.HTTPError: On non-2xx HTTP status codes.
        """
        params["api_key"] = self.api_key
        url = f"{self.base_url}{endpoint}"
        logger.debug(
            "EIA GET %s params=%s", url, {k: v for k, v in params.items() if k != "api_key"}
        )
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def fetch_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        """Fetch a time series from the EIA API.

        Args:
            series_id: EIA series identifier (e.g. ``"PET.WCRSTUS1.W"``).
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.
            frequency: Data frequency (``"weekly"``, ``"monthly"``, ``"annual"``).

        Returns:
            ``pd.DataFrame`` with a ``date`` index and a ``value`` column.
        """
        params: dict[str, Any] = {
            "frequency": frequency,
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        endpoint = f"/seriesid/{series_id}"
        try:
            data = self._get(endpoint, params)
        except Exception as exc:
            logger.error("Failed to fetch EIA series %s: %s", series_id, exc)
            return pd.DataFrame()

        records = data.get("response", {}).get("data", [])
        if not records:
            logger.warning("No data returned for EIA series %s", series_id)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["period"])
        df = df[["date", "value"]].set_index("date").sort_index()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.columns = [series_id.split(".")[-1].lower()]
        logger.info("Fetched %d rows for EIA series %s", len(df), series_id)
        return df

    def fetch_crude_storage(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Fetch weekly US crude oil storage levels (thousand barrels).

        Args:
            start: Start date string.
            end: End date string.

        Returns:
            ``pd.DataFrame`` with weekly storage data.
        """
        return self.fetch_series(self.SERIES["crude_storage"], start=start, end=end)

    def fetch_nat_gas_storage(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Fetch weekly US natural gas storage levels (billion cubic feet).

        Args:
            start: Start date string.
            end: End date string.

        Returns:
            ``pd.DataFrame`` with weekly storage data.
        """
        return self.fetch_series(self.SERIES["nat_gas_storage"], start=start, end=end)

    def compute_storage_surprise(
        self,
        actual_storage: pd.DataFrame,
        rolling_window: int = 4,
    ) -> pd.DataFrame:
        """Compute storage surprise as actual minus rolling-average expectation.

        A positive surprise (higher-than-expected build) is typically bearish,
        while a negative surprise (draw larger than expected) is bullish.

        Args:
            actual_storage: DataFrame with storage levels from ``fetch_crude_storage``
                or ``fetch_nat_gas_storage``.
            rolling_window: Number of weeks to use for the expectation baseline.

        Returns:
            ``pd.DataFrame`` with additional ``expected`` and ``surprise`` columns.
        """
        col = actual_storage.columns[0]
        df = actual_storage.copy()
        df["change"] = df[col].diff()
        df["expected_change"] = df["change"].rolling(rolling_window).mean()
        df["surprise"] = df["change"] - df["expected_change"]
        logger.debug("Computed storage surprise with window=%d", rolling_window)
        return df
