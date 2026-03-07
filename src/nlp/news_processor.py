"""News article processing and deduplication.

Cleans raw news article DataFrames, removes duplicates using text similarity,
filters for relevance to energy markets, and structures articles for
downstream NLP processing.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Energy market keywords for relevance filtering
ENERGY_KEYWORDS = [
    "oil",
    "crude",
    "brent",
    "wti",
    "petroleum",
    "opec",
    "barrel",
    "natural gas",
    "lng",
    "pipeline",
    "refinery",
    "gasoline",
    "diesel",
    "energy",
    "fuel",
    "storage",
    "inventory",
    "production",
    "supply",
    "demand",
    "rig count",
    "shale",
    "fracking",
    "offshore",
    "eia",
    "iea",
    "commodit",
    "price forecast",
]


class NewsProcessor:
    """Cleans, deduplicates, and filters news articles for energy relevance.

    Attributes:
        min_title_length: Minimum character length for valid article titles.
        dedup_threshold: Cosine similarity threshold for near-duplicate detection.
        require_energy_relevance: Whether to filter out non-energy articles.
    """

    def __init__(
        self,
        min_title_length: int = 20,
        dedup_threshold: float = 0.85,
        require_energy_relevance: bool = True,
    ) -> None:
        """Initialise the news processor.

        Args:
            min_title_length: Minimum title length; shorter titles are dropped.
            dedup_threshold: Similarity threshold above which articles are
                considered duplicates (only the first is kept).
            require_energy_relevance: If ``True``, articles with no energy
                keywords are filtered out.
        """
        self.min_title_length = min_title_length
        self.dedup_threshold = dedup_threshold
        self.require_energy_relevance = require_energy_relevance

    def clean_text(self, text: str | None) -> str:
        """Normalise a text string for NLP processing.

        Args:
            text: Raw text string (may be ``None``).

        Returns:
            Cleaned string with normalised whitespace and removed HTML.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove special characters that add no semantic value
        text = re.sub(r"[^\w\s.,!?;:()\-\'\"$%]", "", text)
        return text

    def is_energy_relevant(self, text: str) -> bool:
        """Check if a text contains energy market keywords.

        Args:
            text: Article title or body text.

        Returns:
            ``True`` if any energy keyword is found (case-insensitive).
        """
        lower = text.lower()
        return any(kw in lower for kw in ENERGY_KEYWORDS)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full processing pipeline: clean, filter, deduplicate.

        Args:
            df: Raw news DataFrame from ``NewsFetcher`` with columns
                ``title``, ``description``, ``content``, ``published_at``.

        Returns:
            Processed DataFrame with cleaned text and relevance scores.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to NewsProcessor")
            return df

        logger.info("Processing %d raw articles", len(df))
        df = df.copy()

        # 1. Clean text fields
        for col in ["title", "description", "content"]:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)

        # 2. Drop articles with very short titles
        df = df[df["title"].str.len() >= self.min_title_length].copy()

        # 3. Drop articles with missing publication dates
        df = df.dropna(subset=["published_at"])

        # 4. Filter for energy relevance
        if self.require_energy_relevance:
            combined_text = (
                df["title"].fillna("")
                + " "
                + df.get("description", pd.Series("", index=df.index)).fillna("")
            )
            mask = combined_text.apply(self.is_energy_relevant)
            before = len(df)
            df = df[mask].copy()
            logger.info("Relevance filter: %d → %d articles", before, len(df))

        # 5. Deduplicate by title similarity (simple exact-title dedup first)
        df = df.drop_duplicates(subset=["title"])

        # 6. Build combined text for NLP
        df["full_text"] = (
            df["title"].fillna("")
            + ". "
            + df.get("description", pd.Series("", index=df.index)).fillna("")
        )

        # 7. Sort by publication date
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

        logger.info("Processing complete: %d articles remain", len(df))
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate processed articles to a daily granularity.

        Groups articles by calendar day and creates aggregated fields
        useful for daily sentiment calculations.

        Args:
            df: Processed articles DataFrame with ``published_at`` column.

        Returns:
            Daily aggregated DataFrame with ``article_count``,
            ``titles_combined``, and ``text_combined`` columns.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["published_at"]).dt.date
        daily = (
            df.groupby("date")
            .agg(
                article_count=("title", "count"),
                titles_combined=("title", lambda x: " | ".join(x.dropna())),
                text_combined=("full_text", lambda x: " ".join(x.dropna())),
            )
            .reset_index()
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date").sort_index()
        logger.info("Aggregated to %d daily records", len(daily))
        return daily
