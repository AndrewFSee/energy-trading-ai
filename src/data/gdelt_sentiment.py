"""GDELT-based energy news sentiment retrieval.

Uses the GDELT DOC 2.0 API to fetch historical daily average tone
(sentiment) of global news articles about energy and oil markets.
Coverage: ~February 2015 to present.  No API key required.

The GDELT Project (gdeltproject.org) tracks news from tens of thousands
of outlets worldwide — including Business Insider, Reuters, Bloomberg,
CNBC, etc. — and computes a tone score for every article.  Negative tone
is bearish, positive tone is bullish.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Energy-related search queries — GDELT supports Boolean operators
ENERGY_QUERIES = [
    '"crude oil" OR "oil price" OR WTI OR "Brent crude" OR OPEC',
    '"natural gas price" OR "energy market" OR petroleum OR "oil demand" OR "oil supply"',
]


class GDELTSentiment:
    """Fetches daily energy news sentiment from the GDELT Project.

    GDELT indexes articles from tens of thousands of news sources and
    computes a "tone" score per article (scale roughly -10 to +10).
    This class retrieves daily aggregated tone for energy-related
    articles, suitable for use as ML features.

    Attributes:
        queries: List of Boolean search queries for GDELT.
        request_delay: Seconds to wait between API calls.
    """

    EARLIEST_DATE = "2015-02-19"  # GDELT DOC 2.0 coverage start

    def __init__(
        self,
        queries: list[str] | None = None,
        request_delay: float = 3.0,
        max_retries: int = 3,
    ) -> None:
        """Initialise the GDELT sentiment fetcher.

        Args:
            queries: Energy-related search queries for GDELT.
            request_delay: Delay (seconds) between API requests to
                respect rate limits.
            max_retries: Maximum number of retries per failed request.
        """
        self.queries = queries or ENERGY_QUERIES
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnergyTradingAI/1.0"})

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _fetch_timeline_chunk(
        self,
        query: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Fetch one time-chunk of tone data from the GDELT DOC API.

        Includes retry logic with exponential backoff for rate limiting.

        Args:
            query: GDELT search query string.
            start_dt: Chunk start datetime.
            end_dt: Chunk end datetime.

        Returns:
            DataFrame with ``datetime`` and ``tone`` columns.
        """
        params = {
            "query": query,
            "mode": "timelinetone",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
            "format": "csv",
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(GDELT_DOC_API, params=params, timeout=120)
                if resp.status_code == 429:
                    wait = self.request_delay * (2 ** (attempt + 1))
                    logger.warning("GDELT 429 rate-limit, waiting %.0fs (attempt %d/%d)",
                                   wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

                text = resp.text.strip()
                if not text:
                    return pd.DataFrame(columns=["datetime", "tone"])

                return self._parse_timeline_csv(text)

            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    wait = self.request_delay * (2 ** (attempt + 1))
                    logger.warning("GDELT request error, retrying in %.0fs: %s", wait, exc)
                    time.sleep(wait)
                else:
                    raise

        return pd.DataFrame(columns=["datetime", "tone"])

    @staticmethod
    def _parse_timeline_csv(text: str) -> pd.DataFrame:
        """Parse the CSV response from GDELT's timeline mode.

        GDELT may return various CSV layouts depending on the query;
        this method handles the common formats robustly.

        Args:
            text: Raw CSV text from the API.

        Returns:
            DataFrame with ``datetime`` and ``tone`` columns.
        """
        records: list[dict] = []

        # Try pandas CSV first (works if response is well-formed)
        try:
            df = pd.read_csv(io.StringIO(text), header=None)
            if df.shape[1] >= 2:
                # Assume first column is datetime, second is numeric tone
                for _, row in df.iterrows():
                    try:
                        dt = pd.to_datetime(str(row.iloc[0]), errors="coerce")
                        tone = float(row.iloc[1])
                        if pd.notna(dt):
                            records.append({"datetime": dt, "tone": tone})
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass

        # Fallback: line-by-line parsing for space/tab separated data
        if not records:
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.replace("\t", " ").split()
                if len(parts) >= 2:
                    try:
                        dt = pd.to_datetime(parts[0], errors="coerce")
                        tone = float(parts[1])
                        if pd.notna(dt):
                            records.append({"datetime": dt, "tone": tone})
                    except (ValueError, TypeError):
                        continue

        if not records:
            return pd.DataFrame(columns=["datetime", "tone"])

        return pd.DataFrame(records)

    def _fetch_volume_chunk(
        self,
        query: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Fetch article volume timeline from GDELT.

        Args:
            query: GDELT search query string.
            start_dt: Chunk start datetime.
            end_dt: Chunk end datetime.

        Returns:
            DataFrame with ``datetime`` and ``volume`` columns.
        """
        params = {
            "query": query,
            "mode": "timelinevolraw",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
            "format": "csv",
        }
        try:
            resp = self.session.get(GDELT_DOC_API, params=params, timeout=120)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return pd.DataFrame(columns=["datetime", "volume"])

            records: list[dict] = []
            df = pd.read_csv(io.StringIO(text), header=None)
            if df.shape[1] >= 2:
                for _, row in df.iterrows():
                    try:
                        dt = pd.to_datetime(str(row.iloc[0]), errors="coerce")
                        vol = float(row.iloc[1])
                        if pd.notna(dt):
                            records.append({"datetime": dt, "volume": vol})
                    except (ValueError, TypeError):
                        continue
            return pd.DataFrame(records) if records else pd.DataFrame(columns=["datetime", "volume"])
        except Exception as exc:
            logger.debug("Volume fetch failed: %s", exc)
            return pd.DataFrame(columns=["datetime", "volume"])

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fetch_sentiment(
        self,
        start: str = "2015-02-19",
        end: str | None = None,
        chunk_months: int = 3,
    ) -> pd.DataFrame:
        """Fetch daily energy news sentiment from GDELT.

        Queries the GDELT DOC API in date chunks, aggregates results
        to daily frequency, and computes derived sentiment features.

        Args:
            start: Start date ``YYYY-MM-DD``.  Clamped to 2015-02-19.
            end: End date ``YYYY-MM-DD``.  Defaults to today.
            chunk_months: Size of each API request chunk in months.

        Returns:
            DataFrame indexed by date with columns:
            ``gdelt_tone_mean``, ``gdelt_tone_std``, ``gdelt_tone_min``,
            ``gdelt_tone_max``, ``gdelt_obs_count``, ``gdelt_tone_range``,
            ``gdelt_tone_ma7``, ``gdelt_tone_ma30``, ``gdelt_tone_momentum``,
            ``gdelt_tone_zscore``, ``gdelt_attention``.
        """
        end = end or datetime.now().strftime("%Y-%m-%d")
        start_dt = max(pd.to_datetime(start), pd.to_datetime(self.EARLIEST_DATE))
        end_dt = pd.to_datetime(end)

        if start_dt >= end_dt:
            logger.warning("GDELT: start >= end, returning empty DataFrame")
            return pd.DataFrame()

        logger.info(
            "Fetching GDELT energy sentiment (%s → %s) with %d queries...",
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            len(self.queries),
        )

        tone_chunks: list[pd.DataFrame] = []
        vol_chunks: list[pd.DataFrame] = []

        for qi, query in enumerate(self.queries):
            cursor = start_dt
            while cursor < end_dt:
                chunk_end = min(
                    cursor + pd.DateOffset(months=chunk_months), end_dt
                )
                # Tone timeline
                try:
                    tone_df = self._fetch_timeline_chunk(query, cursor, chunk_end)
                    if not tone_df.empty:
                        tone_df["query"] = query
                        tone_chunks.append(tone_df)
                    logger.info(
                        "  GDELT tone: query %d/%d, %s → %s: %d records",
                        qi + 1,
                        len(self.queries),
                        cursor.strftime("%Y-%m"),
                        chunk_end.strftime("%Y-%m"),
                        len(tone_df),
                    )
                except requests.RequestException as exc:
                    logger.warning(
                        "  GDELT tone request failed (%s → %s): %s",
                        cursor.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                        exc,
                    )

                time.sleep(self.request_delay)

                # Volume timeline
                try:
                    vol_df = self._fetch_volume_chunk(query, cursor, chunk_end)
                    if not vol_df.empty:
                        vol_chunks.append(vol_df)
                except requests.RequestException:
                    pass

                time.sleep(self.request_delay)
                cursor = chunk_end

        if not tone_chunks:
            logger.warning("No GDELT tone data retrieved")
            return pd.DataFrame()

        # ------ Aggregate to daily frequency ------
        raw = pd.concat(tone_chunks, ignore_index=True)
        raw["date"] = raw["datetime"].dt.normalize()

        daily = raw.groupby("date").agg(
            gdelt_tone_mean=("tone", "mean"),
            gdelt_tone_std=("tone", "std"),
            gdelt_tone_min=("tone", "min"),
            gdelt_tone_max=("tone", "max"),
            gdelt_obs_count=("tone", "count"),
        )
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"

        # Volume data (if available)
        if vol_chunks:
            vol_raw = pd.concat(vol_chunks, ignore_index=True)
            vol_raw["date"] = vol_raw["datetime"].dt.normalize()
            vol_daily = vol_raw.groupby("date")["volume"].sum()
            daily["gdelt_article_volume"] = vol_daily.reindex(daily.index).fillna(0)
        else:
            daily["gdelt_article_volume"] = daily["gdelt_obs_count"]

        # ------ Derived features ------
        daily["gdelt_tone_range"] = daily["gdelt_tone_max"] - daily["gdelt_tone_min"]
        daily["gdelt_tone_ma7"] = (
            daily["gdelt_tone_mean"].rolling(7, min_periods=1).mean()
        )
        daily["gdelt_tone_ma30"] = (
            daily["gdelt_tone_mean"].rolling(30, min_periods=1).mean()
        )
        daily["gdelt_tone_momentum"] = daily["gdelt_tone_ma7"] - daily["gdelt_tone_ma30"]

        roll_mean = daily["gdelt_tone_mean"].rolling(60, min_periods=10).mean()
        roll_std = daily["gdelt_tone_mean"].rolling(60, min_periods=10).std()
        daily["gdelt_tone_zscore"] = (daily["gdelt_tone_mean"] - roll_mean) / roll_std

        daily["gdelt_attention"] = np.log1p(daily["gdelt_article_volume"])

        # Fill NaN std for days with single observation
        daily["gdelt_tone_std"] = daily["gdelt_tone_std"].fillna(0)
        daily["gdelt_tone_zscore"] = daily["gdelt_tone_zscore"].fillna(0)

        logger.info(
            "GDELT sentiment complete: %d daily records (%s → %s), %d features",
            len(daily),
            daily.index.min().strftime("%Y-%m-%d"),
            daily.index.max().strftime("%Y-%m-%d"),
            daily.shape[1],
        )

        return daily
