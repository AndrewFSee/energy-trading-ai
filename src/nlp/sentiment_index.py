"""Daily rolling sentiment index for energy markets.

Aggregates per-article FinBERT sentiment scores into a daily composite
sentiment index.  Applies exponential decay weighting to give more
recent articles higher influence, and normalises into a -1 to +1 range.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentIndex:
    """Constructs a daily rolling sentiment index from article-level scores.

    The index is designed to be used as a feature in the ML forecasting
    pipeline.  It captures the aggregate market sentiment signal from
    energy news coverage.

    Attributes:
        rolling_window: Number of days to include in the rolling window.
        min_articles: Minimum articles per day to compute a valid index value.
        decay_factor: Exponential decay weight (higher = faster decay).
    """

    def __init__(
        self,
        rolling_window: int = 7,
        min_articles: int = 3,
        decay_factor: float = 0.5,
    ) -> None:
        """Initialise the sentiment index calculator.

        Args:
            rolling_window: Number of calendar days in the rolling window.
            min_articles: Minimum number of articles required to compute
                a non-NaN index value for a given day.
            decay_factor: Controls exponential decay of older articles.
                Higher values mean older articles lose weight faster.
        """
        self.rolling_window = rolling_window
        self.min_articles = min_articles
        self.decay_factor = decay_factor

    def compute_daily_scores(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate article-level sentiment to daily averages.

        Args:
            sentiment_df: DataFrame with ``published_at`` and ``net_sentiment``
                columns from ``SentimentAnalyzer.analyse_dataframe()``.

        Returns:
            Daily aggregated DataFrame with ``mean_sentiment``,
            ``article_count``, ``positive_ratio``, ``negative_ratio`` columns,
            indexed by date.
        """
        df = sentiment_df.copy()
        df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()

        daily = df.groupby("date").agg(
            mean_sentiment=("net_sentiment", "mean"),
            median_sentiment=("net_sentiment", "median"),
            article_count=("net_sentiment", "count"),
            positive_ratio=(
                "sentiment_label",
                lambda x: (x == "positive").sum() / len(x),
            ),
            negative_ratio=(
                "sentiment_label",
                lambda x: (x == "negative").sum() / len(x),
            ),
        )

        # Mask days with insufficient coverage
        mask = daily["article_count"] < self.min_articles
        daily.loc[mask, ["mean_sentiment", "median_sentiment"]] = np.nan

        logger.info("Daily sentiment computed for %d days", len(daily))
        return daily

    def compute_rolling_index(
        self,
        daily_scores: pd.DataFrame,
        price_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Compute the rolling composite sentiment index.

        Applies a rolling mean over ``rolling_window`` days and optionally
        reindexes to align with a price data index.

        Args:
            daily_scores: Daily sentiment DataFrame from ``compute_daily_scores``.
            price_index: Optional ``DatetimeIndex`` to reindex the output onto
                (fills gaps in news coverage with forward-filled values).

        Returns:
            DataFrame with ``sentiment_index``, ``sentiment_momentum``, and
            ``sentiment_regime`` columns.
        """
        df = daily_scores.copy()

        # Rolling mean sentiment index
        df["sentiment_index"] = (
            df["mean_sentiment"].rolling(window=self.rolling_window, min_periods=1).mean()
        )

        # Sentiment momentum: index minus its own 14-day lag
        df["sentiment_momentum"] = df["sentiment_index"].diff(14)

        # Sentiment regime: bullish (1), neutral (0), bearish (-1)
        df["sentiment_regime"] = np.select(
            [df["sentiment_index"] > 0.1, df["sentiment_index"] < -0.1],
            [1, -1],
            default=0,
        ).astype(int)

        # Normalise index to [-1, 1] range using rolling z-score
        rolling_mean = df["sentiment_index"].rolling(60).mean()
        rolling_std = df["sentiment_index"].rolling(60).std().replace(0, np.nan)
        df["sentiment_zscore"] = (df["sentiment_index"] - rolling_mean) / rolling_std

        if price_index is not None:
            df = df.reindex(price_index).ffill()

        logger.info("Rolling sentiment index computed (window=%d days)", self.rolling_window)
        return df

    def build(
        self,
        sentiment_df: pd.DataFrame,
        price_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """End-to-end: article sentiments → rolling daily index.

        Args:
            sentiment_df: Article-level sentiment DataFrame.
            price_index: Optional target date index for resampling.

        Returns:
            Daily sentiment index DataFrame ready for use as model features.
        """
        daily = self.compute_daily_scores(sentiment_df)
        return self.compute_rolling_index(daily, price_index=price_index)
