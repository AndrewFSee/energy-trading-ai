"""LLM-generated daily morning research note.

Generates a structured energy market morning note by combining:
- Latest price data and technical signals
- Sentiment index reading
- EIA storage data
- RAG-retrieved fundamental context
- LLM synthesis into a professional research note
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MORNING_NOTE_PROMPT = """Generate a professional energy market morning note for {date}.

MARKET DATA:
{market_summary}

SENTIMENT SNAPSHOT:
{sentiment_summary}

FUNDAMENTAL CONTEXT (from energy market knowledge base):
{fundamental_context}

Write a concise morning research note (350-500 words) structured as follows:

## Energy Market Morning Note — {date}

### Market Overview
[2-3 sentences on key price moves overnight and current positioning]

### Key Themes
[3 bullet points on the most important drivers for today]

### Technical Outlook
[WTI and Natural Gas technical levels, support/resistance, momentum]

### Fundamental Watch
[Storage, production, OPEC/geopolitical considerations]

### Risks to Watch
[Key downside and upside risks for the session]

### Trade Idea
[One specific actionable trade idea with entry, target, and stop levels]

---
*This morning note is AI-generated and for informational purposes only. Not investment advice.*"""


class MorningNoteGenerator:
    """Generates daily energy market morning research notes using RAG + LLM.

    Combines quantitative market data with retrieved qualitative context
    from the knowledge base to produce a professional research note.

    Attributes:
        llm_client: LLM interface for text generation.
        retriever: Optional RAG retriever for fundamental context.
    """

    def __init__(
        self,
        llm_client,  # type: ignore[type-arg]
        retriever=None,  # type: ignore[type-arg]
    ) -> None:
        """Initialise the morning note generator.

        Args:
            llm_client: ``LLMClient`` instance for text generation.
            retriever: Optional ``Retriever`` instance for RAG context.
        """
        self.llm_client = llm_client
        self.retriever = retriever
        logger.info("MorningNoteGenerator initialised")

    def _build_market_summary(
        self,
        prices: dict[str, float],
        price_changes: dict[str, float],
        technicals: dict[str, str] | None = None,
    ) -> str:
        """Format current price data into a market summary string.

        Args:
            prices: Dict of {instrument: current_price}.
            price_changes: Dict of {instrument: daily_pct_change}.
            technicals: Optional dict of {instrument: technical_signal}.

        Returns:
            Formatted market summary string.
        """
        lines = []
        for instrument, price in prices.items():
            chg = price_changes.get(instrument, 0.0)
            chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
            tech = technicals.get(instrument, "") if technicals else ""
            lines.append(f"  {instrument}: ${price:.2f} ({chg_str}) {tech}".strip())
        return "\n".join(lines)

    def _build_sentiment_summary(
        self, sentiment_score: float, article_count: int, sentiment_regime: str
    ) -> str:
        """Format sentiment data into a summary string.

        Args:
            sentiment_score: Current composite sentiment score [-1, +1].
            article_count: Number of articles used.
            sentiment_regime: Regime label (``"bullish"``, ``"neutral"``, ``"bearish"``).

        Returns:
            Formatted sentiment summary.
        """
        polarity = "+" if sentiment_score >= 0 else ""
        return (
            f"  Composite Sentiment Score: {polarity}{sentiment_score:.3f} "
            f"({sentiment_regime.upper()})\n"
            f"  Articles Analysed: {article_count}"
        )

    def _retrieve_context(self, date_str: str) -> str:
        """Retrieve relevant energy market context from the knowledge base.

        Args:
            date_str: Current date string for query context.

        Returns:
            Formatted context string.
        """
        if self.retriever is None:
            return "RAG knowledge base not available."

        queries = [
            "What are the key fundamental drivers of crude oil prices?",
            "How do OPEC production decisions affect oil markets?",
            "What seasonal patterns affect natural gas prices?",
        ]
        contexts = []
        for q in queries:
            try:
                result = self.retriever.retrieve(q, top_k=2)
                for chunk in result[:1]:
                    contexts.append(f"• {chunk['text'][:300]}...")
            except Exception as exc:
                logger.warning("Context retrieval failed for query '%s': %s", q, exc)
        return "\n".join(contexts) if contexts else "No relevant context found."

    def generate(
        self,
        prices: dict[str, float],
        price_changes: dict[str, float],
        sentiment_score: float = 0.0,
        article_count: int = 0,
        technicals: dict[str, str] | None = None,
        date: datetime | None = None,
    ) -> str:
        """Generate the daily morning note.

        Args:
            prices: Dict of {instrument: current_price}.
            price_changes: Dict of {instrument: daily_pct_change}.
            sentiment_score: Current sentiment index value.
            article_count: Number of news articles in the sentiment index.
            technicals: Optional technical signal labels per instrument.
            date: Report date.  Defaults to today.

        Returns:
            Formatted morning note as a markdown string.
        """
        date = date or datetime.utcnow()
        date_str = date.strftime("%A, %d %B %Y")

        if sentiment_score > 0.1:
            sentiment_regime = "bullish"
        elif sentiment_score < -0.1:
            sentiment_regime = "bearish"
        else:
            sentiment_regime = "neutral"

        market_summary = self._build_market_summary(prices, price_changes, technicals)
        sentiment_summary = self._build_sentiment_summary(
            sentiment_score, article_count, sentiment_regime
        )
        fundamental_context = self._retrieve_context(date_str)

        prompt = MORNING_NOTE_PROMPT.format(
            date=date_str,
            market_summary=market_summary,
            sentiment_summary=sentiment_summary,
            fundamental_context=fundamental_context,
        )

        logger.info("Generating morning note for %s", date_str)
        try:
            note = self.llm_client.complete(prompt)
        except Exception as exc:
            logger.error("Morning note generation failed: %s", exc)
            note = f"*Morning note generation failed: {exc}*"

        return note
