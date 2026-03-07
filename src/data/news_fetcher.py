"""News fetcher module for energy market headlines.

Retrieves energy-related news from the NewsAPI and RSS feeds from major
financial and energy news sources.  Articles are returned in a structured
format ready for downstream NLP processing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Public RSS feeds for energy market news (no API key required)
DEFAULT_RSS_FEEDS = [
    "https://www.eia.gov/rss/todayinenergy.xml",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.spglobal.com/commodityinsights/en/rss-feed/oil",
]

# NewsAPI search queries for energy topics
DEFAULT_QUERIES = [
    "crude oil price OPEC",
    "natural gas price storage",
    "energy market supply demand",
    "oil production pipeline",
    "WTI Brent futures",
]


class NewsFetcher:
    """Fetches energy-related news from NewsAPI and RSS feeds.

    Combines results from the NewsAPI (requires API key) and free RSS feeds
    into a unified DataFrame suitable for sentiment analysis.

    Attributes:
        api_key: NewsAPI key.
        session: HTTP session for direct requests.
    """

    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: str | None = None,
        rss_feeds: list[str] | None = None,
    ) -> None:
        """Initialise the news fetcher.

        Args:
            api_key: NewsAPI key.  If ``None``, reads from ``NEWS_API_KEY``
                environment variable.  RSS feeds work without an API key.
            rss_feeds: List of RSS feed URLs to scrape.  Defaults to
                ``DEFAULT_RSS_FEEDS``.
        """
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        self.rss_feeds = rss_feeds or DEFAULT_RSS_FEEDS
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnergyTradingAI/1.0"})
        if self.api_key:
            logger.info("NewsFetcher initialised with NewsAPI key")
        else:
            logger.info("NewsFetcher initialised (RSS only — no NewsAPI key)")

    def fetch_newsapi(
        self,
        queries: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        page_size: int = 100,
        language: str = "en",
    ) -> pd.DataFrame:
        """Fetch articles from NewsAPI for given search queries.

        Args:
            queries: List of search query strings.  Defaults to
                ``DEFAULT_QUERIES``.
            start: Start date in ``YYYY-MM-DD`` format.  Defaults to 7 days ago.
            end: End date in ``YYYY-MM-DD`` format.  Defaults to today.
            page_size: Number of articles per page (max 100).
            language: Article language code.

        Returns:
            ``pd.DataFrame`` with columns: ``title``, ``description``, ``content``,
            ``url``, ``published_at``, ``source``, ``query``, ``article_id``.
        """
        if not self.api_key:
            logger.warning("No NewsAPI key — skipping NewsAPI fetch")
            return pd.DataFrame()

        if queries is None:
            queries = DEFAULT_QUERIES
        if start is None:
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        all_articles: list[dict] = []
        for query in queries:
            params = {
                "q": query,
                "from": start,
                "to": end,
                "language": language,
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": self.api_key,
            }
            try:
                resp = self.session.get(self.NEWSAPI_URL, params=params, timeout=30)
                resp.raise_for_status()
                articles = resp.json().get("articles", [])
                for art in articles:
                    all_articles.append(
                        {
                            "title": art.get("title", ""),
                            "description": art.get("description", ""),
                            "content": art.get("content", ""),
                            "url": art.get("url", ""),
                            "published_at": art.get("publishedAt", ""),
                            "source": art.get("source", {}).get("name", ""),
                            "query": query,
                        }
                    )
                logger.info("NewsAPI: %d articles for query '%s'", len(articles), query)
            except Exception as exc:
                logger.error("NewsAPI fetch failed for query '%s': %s", query, exc)

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["article_id"] = df["url"].apply(
            lambda u: hashlib.md5(u.encode()).hexdigest()  # noqa: S324
        )
        return df

    def fetch_rss(self, feeds: list[str] | None = None) -> pd.DataFrame:
        """Fetch articles from RSS feeds.

        Args:
            feeds: List of RSS feed URLs.  Defaults to ``self.rss_feeds``.

        Returns:
            ``pd.DataFrame`` with columns: ``title``, ``description``, ``url``,
            ``published_at``, ``source``, ``article_id``.
        """
        feeds = feeds or self.rss_feeds
        all_articles: list[dict] = []

        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    pub = entry.get("published", entry.get("updated", ""))
                    all_articles.append(
                        {
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "content": entry.get("summary", ""),
                            "url": entry.get("link", ""),
                            "published_at": pub,
                            "source": feed.feed.get("title", feed_url),
                            "query": "rss",
                        }
                    )
                logger.info("RSS: %d articles from %s", len(feed.entries), feed_url)
            except Exception as exc:
                logger.error("RSS fetch failed for %s: %s", feed_url, exc)

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["article_id"] = df["url"].apply(
            lambda u: hashlib.md5(u.encode()).hexdigest()  # noqa: S324
        )
        return df

    def fetch_all(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch articles from all available sources (NewsAPI + RSS).

        Args:
            start: Start date string for NewsAPI.
            end: End date string for NewsAPI.

        Returns:
            Combined and deduplicated ``pd.DataFrame`` of articles.
        """
        frames = []
        newsapi_df = self.fetch_newsapi(start=start, end=end)
        if not newsapi_df.empty:
            frames.append(newsapi_df)

        rss_df = self.fetch_rss()
        if not rss_df.empty:
            frames.append(rss_df)

        if not frames:
            logger.warning("No news articles fetched from any source")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        # Deduplicate by article_id
        combined = combined.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
        combined = combined.sort_values("published_at", ascending=False)
        logger.info("Total unique articles fetched: %d", len(combined))
        return combined
